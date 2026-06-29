"""Resumable SQLite store for the Calogero-Sutherland grid sweeps.

One row per ``(physics, seed, hp)`` run — that tuple is the natural key, so a run is fully
reproducible from its row. Re-running a sweep skips ``done`` rows and re-queues anything left
``running`` (an interrupted process).

**Grid-agnostic.** The physics and solver settings are stored as JSON (``physics_json``,
``hp_json``), not fixed columns, so you can grid over *any* axes (L, N, epsilon, lr, kind, …)
without a schema change. Query individual axes with SQLite's JSON1, e.g.
``json_extract(physics_json,'$.L')`` — or just use ``sweep.load_table`` which expands both
JSON blobs into DataFrame columns.

Status values: ``todo`` -> ``running`` -> ``done`` | ``failed``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime

DEFAULT_DB = "outputs/cs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY,
    physics_json  TEXT NOT NULL,
    seed          INTEGER NOT NULL,
    hp_json       TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'todo',
    e_total       REAL,
    e_per_n       REAL,
    err_total     REAL,
    err_per_n     REAL,
    sigma_e       REAL,
    acceptance    REAL,
    passed        INTEGER,
    e_exact       REAL,
    gap           REAL,
    verdict_json  TEXT,
    run_dir       TEXT,
    error         TEXT,
    started_at    TEXT,
    finished_at   TEXT,
    UNIQUE (physics_json, seed, hp_json)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: str = DEFAULT_DB) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def _key(physics, seed: int, hp) -> tuple:
    return (json.dumps(physics.to_dict(), sort_keys=True), int(seed),
            json.dumps(hp.to_dict(), sort_keys=True))


def status_of(conn, physics, seed, hp) -> str | None:
    pj, seed, hj = _key(physics, seed, hp)
    row = conn.execute(
        "SELECT status FROM runs WHERE physics_json=? AND seed=? AND hp_json=?",
        (pj, seed, hj),
    ).fetchone()
    return row["status"] if row else None


def enqueue(conn, physics, seed, hp) -> None:
    """Insert a ``todo`` row if this run is not already recorded (idempotent)."""
    pj, seed, hj = _key(physics, seed, hp)
    conn.execute(
        "INSERT OR IGNORE INTO runs (physics_json, seed, hp_json, status) VALUES (?,?,?, 'todo')",
        (pj, seed, hj),
    )
    conn.commit()


def mark_running(conn, physics, seed, hp) -> None:
    pj, seed, hj = _key(physics, seed, hp)
    conn.execute(
        "UPDATE runs SET status='running', started_at=?, error=NULL "
        "WHERE physics_json=? AND seed=? AND hp_json=?",
        (_now(), pj, seed, hj),
    )
    conn.commit()


def save_result(conn, result, run_dir: str | None = None) -> None:
    """Persist a :class:`point.CSResult` as a ``done`` row (links to its artifact dir)."""
    pj, seed, hj = _key(result.physics, result.seed, result.hp)
    conn.execute(
        "UPDATE runs SET status='done', e_total=?, e_per_n=?, err_total=?, err_per_n=?, "
        "sigma_e=?, acceptance=?, passed=?, e_exact=?, gap=?, verdict_json=?, run_dir=?, "
        "finished_at=? WHERE physics_json=? AND seed=? AND hp_json=?",
        (
            result.e_total, result.e_per_n, result.err_total, result.err_per_n,
            result.sigma_e, result.acceptance, int(result.passed), result.e_exact, result.gap,
            json.dumps(_json_safe(result.verdict)), run_dir, _now(),
            pj, seed, hj,
        ),
    )
    conn.commit()


def mark_failed(conn, physics, seed, hp, error: str) -> None:
    pj, seed, hj = _key(physics, seed, hp)
    conn.execute(
        "UPDATE runs SET status='failed', error=?, finished_at=? "
        "WHERE physics_json=? AND seed=? AND hp_json=?",
        (error[:2000], _now(), pj, seed, hj),
    )
    conn.commit()


def requeue_interrupted(conn) -> int:
    """Reset any ``running`` rows (a crashed/killed process) back to ``todo``. Returns count."""
    cur = conn.execute("UPDATE runs SET status='todo' WHERE status='running'")
    conn.commit()
    return cur.rowcount


def claim_next(conn):
    """Atomically claim one ``todo`` row for execution (multi-worker safe). Returns row or None.

    ``BEGIN IMMEDIATE`` serialises concurrent workers on the WAL write lock, so two GPU workers
    can never grab the same point. The returned row carries ``physics_json, seed, hp_json`` —
    enough to reconstruct the run via ``point.Physics.from_dict`` / ``point.HP.from_dict``.
    """
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        "SELECT id, physics_json, seed, hp_json FROM runs WHERE status='todo' ORDER BY id LIMIT 1"
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
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM runs GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}


def fetch_done(conn) -> list[sqlite3.Row]:
    """All completed rows, ordered by id."""
    return conn.execute("SELECT * FROM runs WHERE status='done' ORDER BY id").fetchall()


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
