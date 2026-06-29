"""A single sweep worker: drain the shared SQLite queue until it is empty.

Run one worker per GPU (``CUDA_VISIBLE_DEVICES`` pinned by ``run_workers.py``); each calls
``db.claim_next`` to atomically grab a distinct ``(physics, seed, hp)`` point, so two workers never
run the same one. Reconstructs the typed ``Physics``/``HP`` from the row's JSON and executes it.

    CUDA_VISIBLE_DEVICES=0 python worker.py --db outputs/cs.db --out-root outputs
"""

from __future__ import annotations

import argparse
import json

import db
from point import HP, Physics
from sweep import execute_claimed


def main() -> None:
    ap = argparse.ArgumentParser(description="Drain the CS sweep queue (one worker).")
    ap.add_argument("--db", default=db.DEFAULT_DB)
    ap.add_argument("--out-root", default="outputs")
    args = ap.parse_args()

    conn = db.connect(args.db)
    n_done = 0
    while True:
        row = db.claim_next(conn)
        if row is None:
            break
        physics = Physics.from_dict(json.loads(row["physics_json"]))
        hp = HP.from_dict(json.loads(row["hp_json"]))
        seed = int(row["seed"])
        print(f"[claim] {physics.label} kind={hp.kind} seed={seed}", flush=True)
        try:
            execute_claimed(conn, physics, seed, hp, out_root=args.out_root)
            n_done += 1
            print(f"[done]  {physics.label} kind={hp.kind} seed={seed}", flush=True)
        except Exception as exc:  # already marked failed in execute_claimed; keep draining
            print(f"[fail]  {physics.label} seed={seed}: {exc}", flush=True)
    print(f"worker finished: {n_done} point(s) completed; queue empty.", flush=True)


if __name__ == "__main__":
    main()
