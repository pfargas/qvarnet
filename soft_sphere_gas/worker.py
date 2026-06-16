"""A single sweep worker: drain the shared DB queue, one point at a time, on one GPU.

Run several of these at once (one per GPU, each with its own ``CUDA_VISIBLE_DEVICES``), all
pointed at the same ``--db``. They coordinate purely through the SQLite work queue:
``db.claim_next`` hands each worker a distinct ``todo`` point atomically, so no two GPUs ever run
the same ``(potential, x, N, seed, hp)``. The launcher ``run_workers.py`` spawns them and fills
the queue first; you can also run one by hand for a single-GPU drain.

    CUDA_VISIBLE_DEVICES=0 python worker.py --db outputs/soft_sphere.db
"""

from __future__ import annotations

import argparse
import json
import os

import db
import sweep
from point import HP, Potential


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="outputs/soft_sphere.db")
    args = ap.parse_args()
    out_root = os.path.dirname(args.db) or "."

    conn = db.connect(args.db)

    import jax  # imported here so CUDA_VISIBLE_DEVICES (set by the launcher) is already in effect

    pid = os.getpid()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"[worker {pid}] CUDA_VISIBLE_DEVICES={cvd}  jax devices={jax.devices()}", flush=True)

    ran = 0
    while True:
        row = db.claim_next(conn)
        if row is None:
            break
        potential = Potential(R=row["R"], V0_paper=row["V0_paper"], label=row["potential_label"])
        hp = HP(**json.loads(row["hp_json"]))
        x, N, seed = row["x"], row["N"], row["seed"]
        print(f"[worker {pid}] running {potential.label} x={x:g} N={N} seed={seed}", flush=True)
        try:
            sweep.execute_claimed(conn, potential, x, N, seed, hp, out_root=out_root)
            print(f"[worker {pid}] done   {potential.label} x={x:g} N={N} seed={seed}", flush=True)
        except Exception as exc:  # recorded as 'failed' inside execute_claimed; keep draining
            print(f"[worker {pid}] FAILED {potential.label} x={x:g} N={N} seed={seed}: {exc!r}",
                  flush=True)
        ran += 1

    print(f"[worker {pid}] queue empty; ran {ran} point(s)", flush=True)


if __name__ == "__main__":
    main()
