"""Multi-GPU launcher: build a grid-agnostic sweep, enqueue it, and run one worker per GPU.

Axes are passed as repeatable ``--axis NAME=v1,v2,...`` (physics: L, N, epsilon, n_dim) and
``--hp-axis NAME=v1,v2,...`` (solver: kind, lr, n_epochs, n_chains, ...). Values are coerced to the
base field's type. Each GPU gets its own process with ``CUDA_VISIBLE_DEVICES`` pinned, all draining
one shared SQLite queue (``db.claim_next`` hands out distinct points atomically).

    # grid over L and N, two model kinds, seeds 0..2, on GPUs 0 and 1
    python run_workers.py --gpus 0,1 --seeds 0 1 2 \
        --axis L=0.5,0.8,1.0,1.5,2.0 --axis N=2,5,10 \
        --hp-axis kind=jastrow,mlp_jastrow --hp-axis n_epochs=2000

Fully resumable: re-run the same command to extend (add seeds / axes) — finished points are skipped.
For a single device by hand:  CUDA_VISIBLE_DEVICES=0 python worker.py --db <path>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import fields

import db
from point import HP, Physics
from sweep import build_grid, enqueue_grid

HERE = os.path.dirname(os.path.abspath(__file__))


def _coerce(value: str, like):
    """Coerce a CLI string to the type of the base field value ``like``."""
    if isinstance(like, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(value)
    if isinstance(like, float):
        return float(value)
    return value  # str (e.g. kind, lr_schedule)


def _parse_axes(specs, base) -> dict:
    """``["L=0.5,0.8", "N=2,5"]`` -> ``{"L": [0.5,0.8], "N": [2,5]}`` (typed off ``base``)."""
    field_types = {f.name: getattr(base, f.name) for f in fields(base)}
    axes = {}
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"bad axis {spec!r}; use NAME=v1,v2,...")
        name, raw = spec.split("=", 1)
        name = name.strip()
        if name not in field_types:
            raise SystemExit(f"unknown {type(base).__name__} field {name!r}")
        axes[name] = [_coerce(v, field_types[name]) for v in raw.split(",") if v != ""]
    return axes


def _detect_gpus() -> list[str]:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return [str(i) for i, line in enumerate(out.splitlines()) if line.strip()]
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Enqueue a CS grid sweep and run one worker per GPU.")
    ap.add_argument("--axis", action="append", default=[], help="physics axis NAME=v1,v2,...")
    ap.add_argument("--hp-axis", action="append", default=[], help="solver axis NAME=v1,v2,...")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--gpus", default=None, help="comma list, e.g. 0,1 (default: auto-detect)")
    ap.add_argument("--db", default=db.DEFAULT_DB)
    ap.add_argument("--out-root", default="outputs")
    # progress / logging: multiple workers sharing one terminal makes tqdm bars clobber each
    # other, so with >1 worker each is redirected to its own log file (tail -f to watch one).
    ap.add_argument("--log-dir", default=None, help="per-worker log dir (default: <out-root>/logs)")
    ap.add_argument("--progress", choices=("bar", "off"), default="bar",
                    help="'off' disables the tqdm bar in workers (TQDM_DISABLE)")
    ap.add_argument("--tqdm-mininterval", type=float, default=5.0,
                    help="min seconds between bar refreshes (throttles log growth)")
    args = ap.parse_args()

    physics_axes = _parse_axes(args.axis, Physics())
    hp_axes = _parse_axes(args.hp_axis, HP())
    grid = build_grid(physics_axes, hp_axes)

    conn = db.connect(args.db)
    requeued = db.requeue_interrupted(conn)
    n = enqueue_grid(conn, grid, args.seeds)
    print(f"grid: {len(grid)} point(s) × {len(args.seeds)} seed(s); "
          f"enqueued/ensured {n} row(s); requeued {requeued} interrupted. "
          f"status={db.status_counts(conn)}")

    gpus = args.gpus.split(",") if args.gpus else _detect_gpus()
    if not gpus:
        print("no GPU detected — running a single CPU worker.")
        gpus = [""]  # empty CUDA_VISIBLE_DEVICES

    # >1 worker on one terminal ⇒ tqdm bars clobber. Give each its own log file instead.
    redirect = len(gpus) > 1
    log_dir = args.log_dir or os.path.join(args.out_root, "logs")
    if redirect:
        os.makedirs(log_dir, exist_ok=True)

    procs, logs = [], []
    for g in gpus:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, MPLBACKEND="Agg",
                   TQDM_MININTERVAL=str(args.tqdm_mininterval))
        if args.progress == "off":
            env["TQDM_DISABLE"] = "1"
        cmd = [sys.executable, "-u", os.path.join(HERE, "worker.py"),
               "--db", args.db, "--out-root", args.out_root]
        if redirect:
            path = os.path.join(log_dir, f"worker_gpu{g or 'cpu'}.log")
            fh = open(path, "w")
            logs.append(fh)
            print(f"launch worker on GPU {g or '(cpu)'} -> {path}   (tail -f to watch)")
            procs.append(subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT))
        else:
            print(f"launch worker on GPU {g or '(cpu)'} (output below)")
            procs.append(subprocess.Popen(cmd, env=env))

    rc = 0
    for p in procs:
        rc |= p.wait()
    for fh in logs:
        fh.close()
    print(f"all workers exited. final status={db.status_counts(conn)}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
