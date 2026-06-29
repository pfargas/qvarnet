"""Merge several CS sweep DBs into one (for multi-machine runs with no shared filesystem).

Every row is keyed by ``(physics_json, seed, hp_json)``, so merging is just ``INSERT OR IGNORE``
followed by a done-precedence pass (a finished result never loses to a leftover ``todo``/``running``).
Copy each machine's ``outputs/runs/`` into a shared ``runs/`` too (run-dir names are unique per key).

    python merge_db.py merged.db pc1.db pc2.db pc3.db
"""

from __future__ import annotations

import argparse
import sqlite3

import db

_COLS = ("physics_json, seed, hp_json, status, e_total, e_per_n, err_total, err_per_n, "
         "sigma_e, acceptance, passed, e_exact, gap, verdict_json, run_dir, error, "
         "started_at, finished_at")
_RANK = {"done": 3, "failed": 2, "running": 1, "todo": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("sources", nargs="+")
    args = ap.parse_args()

    out = db.connect(args.dest)
    merged = 0
    for src_path in args.sources:
        src = sqlite3.connect(src_path)
        src.row_factory = sqlite3.Row
        for r in src.execute(f"SELECT {_COLS} FROM runs").fetchall():
            key = (r["physics_json"], r["seed"], r["hp_json"])
            existing = out.execute(
                "SELECT status FROM runs WHERE physics_json=? AND seed=? AND hp_json=?", key
            ).fetchone()
            if existing and _RANK.get(existing["status"], 0) >= _RANK.get(r["status"], 0):
                continue  # keep the better-status row we already have
            out.execute("DELETE FROM runs WHERE physics_json=? AND seed=? AND hp_json=?", key)
            cols = _COLS.replace(" ", "")
            out.execute(
                f"INSERT INTO runs ({_COLS}) VALUES ({','.join('?' * len(cols.split(',')))})",
                tuple(r[c] for c in cols.split(",")),
            )
            merged += 1
        src.close()
    out.commit()
    print(f"merged {merged} row(s) into {args.dest}; status={db.status_counts(out)}")


if __name__ == "__main__":
    main()
