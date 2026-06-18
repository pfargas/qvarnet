"""Merge per-node sweep DBs into one (multi-PC runs).

The shared-queue coordination is SQLite, so it is **same-node only**: give each PC its own ``--db``
and its own ``outputs/`` tree, then merge afterwards. Every run is keyed by
``(potential_label, x, N, seed, hp_json)`` (the DB's UNIQUE constraint), so rows merge cleanly.

Workflow
--------
1. **One HP set everywhere** — identical HP on every PC ⇒ identical ``hp_json`` hash ⇒ the rows are
   the same physics points and merge (and the per-run dir names match the physics axes). Any HP
   difference makes *different* keys, which then live side by side rather than merging.
2. **Partition the work so no two PCs run the same point** (no wasted GPU, trivial merge). Easiest:
   by seed — ``--seeds 0`` on PC1, ``--seeds 1`` on PC2, ``--seeds 2`` on PC3, each doing the full
   ``--N`` ladder. (Or by N, to put the heavy N=256 on the strongest GPU.) After merging, every
   ``(x, N, seed)`` is present, so ``analysis.extrapolate_thermodynamic`` works unchanged.
3. Each PC runs ``run_workers.py ... --db outputs/<host>.db``.
4. Collect the trees on one machine (rsync), then:

       python merge_db.py outputs/merged.db  pc1/soft_sphere.db pc2/soft_sphere.db pc3/soft_sphere.db
       # then union the per-run artifact dirs (names are unique by physics axes when partitioned):
       rsync -a pc1/runs/  outputs/runs/ ;  rsync -a pc2/runs/  outputs/runs/ ;  rsync -a pc3/runs/  outputs/runs/

5. Point any analysis (``check_db.py``, ``load_curve``, ``extrapolate_thermodynamic``) at
   ``outputs/merged.db``.

WAL note: if a source DB was copied live, its ``-wal`` sidecar carries the latest writes — either
copy it alongside the ``.db`` (rsync of the whole tree does), or run
``sqlite3 src.db "PRAGMA wal_checkpoint(TRUNCATE);"`` on the source first. Attaching reads the WAL.

Conflict policy: rows absent from the target are inserted; for a key present in *both* (only if you
did not partition), a terminal source row (``done``) overwrites a non-terminal target row, so a
finished result never loses to a leftover ``todo``/``running``. Two different ``done`` rows for the
same key (genuine duplicate compute) keep the target's — dedupe upstream by partitioning.
"""

from __future__ import annotations

import argparse

import db

# the UNIQUE key; everything else is payload copied on merge
_KEY = ("potential_label", "x", "N", "seed", "hp_json")


def _columns(conn) -> list[str]:
    """All ``runs`` columns except the autoincrement ``id`` (which the target reassigns)."""
    return [r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall() if r[1] != "id"]


def merge(target: str, sources: list[str]) -> None:
    conn = db.connect(target)  # creates schema if new
    cols = _columns(conn)
    collist = ", ".join(cols)
    payload = [c for c in cols if c not in _KEY]
    key_match = " AND ".join(f"t.{k} = s.{k}" for k in _KEY)

    for src in sources:
        n0 = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.execute("ATTACH DATABASE ? AS src", (src,))
        # 1) insert keys not yet in the target
        conn.execute(f"INSERT OR IGNORE INTO runs ({collist}) SELECT {collist} FROM src.runs")
        # 2) let a finished source row replace a still-pending target row for the same key
        set_clause = ", ".join(f"{c} = s.{c}" for c in payload)
        conn.execute(
            f"UPDATE runs AS t SET {set_clause} "
            f"FROM src.runs AS s "
            f"WHERE {key_match} AND s.status = 'done' AND t.status != 'done'"
        )
        conn.commit()
        conn.execute("DETACH DATABASE src")
        n1 = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        print(f"merged {src}: +{n1 - n0} new rows (target now {n1})")

    print(f"\nfinal status: {db.status_counts(conn)}")
    conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="output DB (created/extended); analyse this one")
    ap.add_argument("sources", nargs="+", help="per-node DBs to merge in")
    args = ap.parse_args()
    merge(args.target, args.sources)


if __name__ == "__main__":
    main()
