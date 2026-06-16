"""Parallel sweep driver: fill the queue, then run one worker per GPU.

This is the multi-GPU entry point. It (1) requeues anything left ``running`` by a previous crash,
(2) enqueues every ``(SS10, x, N, seed, hp)`` point of the requested sweep as ``todo``, then
(3) spawns one ``worker.py`` subprocess per GPU — each pinned to a single device via
``CUDA_VISIBLE_DEVICES`` — all draining the same DB. Resumable and load-balancing: a worker that
finishes a point immediately claims the next, so faster GPUs do more work.

    # use all detected GPUs, default sweep (N=64, seeds 0/1/2, 7 x-points, 2000 epochs)
    python run_workers.py
    # pick GPUs and a finite-N ladder explicitly
    python run_workers.py --gpus 0,1 --N 32 64 128 --seeds 0 1 --epochs 1500 --chains 1024

After it finishes, build the curve/plots exactly as for the serial driver (see RUNNING.md).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import db
import sweep
from point import HP, SS10


def detect_gpus() -> list[str]:
    """GPU indices from ``nvidia-smi -L`` (e.g. ['0','1']); fall back to ['0']."""
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        ids = [line.split(":")[0].split()[-1] for line in out.strip().splitlines() if line.strip()]
        return ids or ["0"]
    except Exception:
        return ["0"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, nargs="+", default=[64], help="particle counts (N-ladder)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-x", type=int, default=7)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--chains", type=int, default=1024)
    p.add_argument("--gpus", type=str, default=None, help="comma list e.g. 0,1; default = all detected")
    p.add_argument("--db", default="outputs/soft_sphere.db")
    args = p.parse_args()

    conn = db.connect(args.db)
    requeued = db.requeue_interrupted(conn)
    if requeued:
        print(f"requeued {requeued} interrupted run(s)")

    hp = HP(n_epochs=args.epochs, n_chains=args.chains)
    n = sweep.enqueue_sweep(conn, SS10, args.N, args.seeds, hp, n_x=args.n_x)
    print(f"enqueued sweep: N={args.N} seeds={args.seeds} n_x={args.n_x} -> {n} new todo points")
    print(f"queue status: {db.status_counts(conn)}")
    conn.close()  # workers open their own connections

    gpus = args.gpus.split(",") if args.gpus else detect_gpus()
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"launching {len(gpus)} worker(s) on GPU(s) {gpus}")

    procs = []
    for g in gpus:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(here, "worker.py"), "--db", args.db], env=env
        ))
    for proc in procs:
        proc.wait()

    conn = db.connect(args.db)
    print(f"all workers done. final queue status: {db.status_counts(conn)}")


if __name__ == "__main__":
    main()
