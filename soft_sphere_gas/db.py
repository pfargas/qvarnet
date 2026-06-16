"""Resumable SQLite store for the soft-sphere sweeps.

One row per ``(potential, x, N, seed, hp)`` run — that tuple is the natural key, so a run is
fully reproducible from its row. Re-running a sweep skips ``done`` rows and re-queues anything
left ``running`` (an interrupted process), exactly like claude-assist's ``status.db``.

Status values: ``todo`` -> ``running`` -> ``done`` | ``failed`` | ``skipped_box`` (R >= L/2).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

DEFAULT_DB = "outputs/soft_sphere.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY,
    potential_label TEXT NOT NULL,
    R               REAL NOT NULL,
    V0_paper        REAL NOT NULL,
    x               REAL NOT NULL,
    N               INTEGER NOT NULL,
    seed            INTEGER NOT NULL,
    hp_json         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'todo',
    e_per_n         REAL,
    err_per_n       REAL,
    sigma_e_per_n   REAL,
    acceptance      REAL,
    passed          INTEGER,
    verdict_json    TEXT,
    L               REAL,
    upper_bound     REAL,
    run_dir         TEXT,
    error           TEXT,
    started_at      TEXT,
    finished_at     TEXT,
    UNIQUE (potential_label, x, N, seed, hp_json)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: str = DEFAULT_DB) -> sqlite3.Connection:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def _key(potential, x: float, N: int, seed: int, hp) -> tuple:
    return (potential.label, x, N, seed, json.dumps(hp.to_dict(), sort_keys=True))


def status_of(conn, potential, x, N, seed, hp) -> str | None:
    label, x, N, seed, hp_json = _key(potential, x, N, seed, hp)
    row = conn.execute(
        "SELECT status FROM runs WHERE potential_label=? AND x=? AND N=? AND seed=? AND hp_json=?",
        (label, x, N, seed, hp_json),
    ).fetchone()
    return row["status"] if row else None


def enqueue(conn, potential, x, N, seed, hp) -> None:
    """Insert a ``todo`` row if this run is not already recorded."""
    label, x, N, seed, hp_json = _key(potential, x, N, seed, hp)
    conn.execute(
        "INSERT OR IGNORE INTO runs (potential_label, R, V0_paper, x, N, seed, hp_json, status) "
        "VALUES (?,?,?,?,?,?,?, 'todo')",
        (label, potential.R, potential.V0_paper, x, N, seed, hp_json),
    )
    conn.commit()


def mark_running(conn, potential, x, N, seed, hp) -> None:
    label, x, N, seed, hp_json = _key(potential, x, N, seed, hp)
    conn.execute(
        "UPDATE runs SET status='running', started_at=?, error=NULL "
        "WHERE potential_label=? AND x=? AND N=? AND seed=? AND hp_json=?",
        (_now(), label, x, N, seed, hp_json),
    )
    conn.commit()


def mark_skipped_box(conn, potential, x, N, seed, hp, L: float) -> None:
    enqueue(conn, potential, x, N, seed, hp)
    label, x, N, seed, hp_json = _key(potential, x, N, seed, hp)
    conn.execute(
        "UPDATE runs SET status='skipped_box', L=?, finished_at=? "
        "WHERE potential_label=? AND x=? AND N=? AND seed=? AND hp_json=?",
        (L, _now(), label, x, N, seed, hp_json),
    )
    conn.commit()


def save_result(conn, result, run_dir: str | None = None) -> None:
    """Persist a :class:`point.PointResult` as a ``done`` row.

    ``run_dir`` is the path (relative to the DB's ``outputs/``) of this run's artifact directory
    written by ``io.write_run_artifacts`` — it ties the scalar row to its history + best params.
    """
    label, x, N, seed, hp_json = _key(result.potential, result.x, result.N, result.seed, result.hp)
    conn.execute(
        "UPDATE runs SET status='done', e_per_n=?, err_per_n=?, sigma_e_per_n=?, acceptance=?, "
        "passed=?, verdict_json=?, L=?, upper_bound=?, run_dir=?, finished_at=? "
        "WHERE potential_label=? AND x=? AND N=? AND seed=? AND hp_json=?",
        (
            result.e_per_n, result.err_per_n, result.sigma_e_per_n, result.acceptance,
            int(result.passed), json.dumps(_json_safe(result.verdict)), result.L,
            result.upper_bound, run_dir, _now(),
            label, x, N, seed, hp_json,
        ),
    )
    conn.commit()


def mark_failed(conn, potential, x, N, seed, hp, error: str) -> None:
    label, x, N, seed, hp_json = _key(potential, x, N, seed, hp)
    conn.execute(
        "UPDATE runs SET status='failed', error=?, finished_at=? "
        "WHERE potential_label=? AND x=? AND N=? AND seed=? AND hp_json=?",
        (error[:2000], _now(), label, x, N, seed, hp_json),
    )
    conn.commit()


def requeue_interrupted(conn) -> int:
    """Reset any ``running`` rows (a crashed/killed process) back to ``todo``. Returns count."""
    cur = conn.execute("UPDATE runs SET status='todo' WHERE status='running'")
    conn.commit()
    return cur.rowcount


def claim_next(conn):
    """Atomically claim one ``todo`` row for execution (multi-worker safe). Returns the row or None.

    Uses ``BEGIN IMMEDIATE`` so concurrent workers on one WAL database serialise on the write
    lock — two GPU workers can never grab the same point. The returned row carries everything
    needed to reconstruct the run: ``potential_label, R, V0_paper, x, N, seed, hp_json``.
    """
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        "SELECT id, potential_label, R, V0_paper, x, N, seed, hp_json "
        "FROM runs WHERE status='todo' ORDER BY id LIMIT 1"
    ).fetchone()
    if row is None:
        conn.commit()
        return None
    conn.execute(
        "UPDATE runs SET status='running', started_at=?, error=NULL WHERE id=?",
        (_now(), row["id"]),
    )
    conn.commit()
    return row


def status_counts(conn) -> dict:
    """``{status: count}`` across all rows — for launcher/worker progress reporting."""
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM runs GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}


def fetch_done(conn, potential_label: str, N: int) -> list[sqlite3.Row]:
    """All completed, verdict-passing rows for one (potential, N), ordered by x then seed."""
    return conn.execute(
        "SELECT * FROM runs WHERE potential_label=? AND N=? AND status='done' "
        "ORDER BY x, seed",
        (potential_label, N),
    ).fetchall()


def _json_safe(d: dict) -> dict:
    """Drop non-serializable verdict entries (e.g. per-chain arrays) before storing."""
    out = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(type(v).__name__)
    return out
