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
from point import HP, Potential


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
    # soft-sphere potential: only R is free (V0 fixed by a=1). R=10 -> SS10, R=5 -> SS5; decreasing
    # R toward 1 stiffens the core toward the hard sphere (R=1 is the HS limit, V0 -> inf, NOT
    # representable — use R slightly above 1, e.g. 1.2, to approach it). Label becomes "SS<R>".
    p.add_argument("--R", type=float, nargs="+", default=[10.0],
                   help="soft-sphere core range(s) in units of a (R>1); pass several to sweep "
                        "multiple potentials at once, e.g. --R 10 5 2")
    p.add_argument("--N", type=int, nargs="+", default=[64], help="particle counts (N-ladder)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-x", type=int, default=7)
    # x-grid bounds (log-spaced, clipped to the box ceiling x<N/(8R^3) of the smallest N). For a
    # SINGLE x point set --n-x 1 and --x-lo to that x (geomspace returns the low end when n=1).
    p.add_argument("--x-lo", type=float, default=1e-5, help="low end of the x grid")
    p.add_argument("--x-hi", type=float, default=1e-2, help="high end of the x grid")
    p.add_argument("--epochs", type=int, default=2000, help="epoch ceiling (early-stop ends sooner)")
    p.add_argument("--chains", type=int, default=1024)
    # DeepSet pooled-latent width (the real capacity bottleneck; was hardcoded 20). Bump toward N
    # (e.g. 256) to test how far a one-body ansatz can go before a Jastrow is needed.
    p.add_argument("--hidden-internal-dim", type=int, default=20,
                   help="DeepSet pooled-latent size (default 20; try 256)")
    # analytic two-body soft-core Jastrow as the short-range correlation factor (jastrow.py).
    # The DeepSet alone is mean-field (sits at Eq.31); --jastrow adds the r_ij correlation hole.
    p.add_argument("--jastrow", action="store_true",
                   help="multiply in the analytic soft-core two-body Jastrow (Σ_{i<j} j(r_ij))")
    # bare-Jastrow baseline: drop the DeepSet entirely (no trainable params) -> lowest-order
    # two-body-correlated energy (IPC/SR analog). Use together with --jastrow.
    p.add_argument("--no-network", action="store_true",
                   help="no neural net — bare analytic Jastrow only (pair with --jastrow)")
    # learning rate + schedule (Adam). cosine/exponential decay lr → lr·lr-final-frac over n_epochs.
    p.add_argument("--lr", type=float, default=3e-3, help="initial Adam learning rate")
    p.add_argument("--lr-schedule", choices=["constant", "cosine", "exponential"],
                   default="constant", help="LR schedule over the epochs")
    p.add_argument("--lr-final-frac", type=float, default=0.1,
                   help="final LR as a fraction of --lr (cosine/exponential)")
    # sampler: per-epoch Laplacian batch = chains * n_eff, n_eff=(chain_len-therm)//thin.
    # Default 21/20/1 = keep 1 sample/chain/epoch (warm walkers carry the chain across epochs).
    p.add_argument("--chain-length", type=int, default=21)
    p.add_argument("--thermalization", type=int, default=20, help="must be < chain-length")
    p.add_argument("--thinning", type=int, default=1)
    p.add_argument("--gpus", type=str, default=None, help="comma list e.g. 0,1; default = all detected")
    p.add_argument("--cpu-per-worker", type=int, default=0,
                   help="CPU threads per worker (0 = auto = cores/n_workers); prevents multi-worker thrash")
    # early-stop tuning (n_epochs is a ceiling). plateau-rel > 0 stops when the tail-mean energy
    # improves by less than that over `es-check-every` epochs — needed because the strict verdict
    # rarely fires at very small x (per-epoch noise > error-of-mean). 0 = verdict-only.
    p.add_argument("--plateau-rel", type=float, default=0.0,
                   help="early-stop on energy plateau (e.g. 0.005 = <0.5%% improvement/check); 0=off")
    p.add_argument("--es-min-epochs", type=int, default=200)
    p.add_argument("--es-check-every", type=int, default=50)
    p.add_argument("--es-patience", type=int, default=2)
    p.add_argument("--no-early-stop", action="store_true", help="run the full --epochs (no early stop)")
    # kinetic-energy Laplacian estimator: forward_ad (exact, default) vs hutchinson (stochastic,
    # cheaper at large N, adds variance). --hutchinson-n-terms sets the number of probe vectors.
    p.add_argument("--laplacian", choices=["forward_ad", "hutchinson", "central_difference"],
                   default="forward_ad", help="kinetic-energy Laplacian estimator")
    p.add_argument("--hutchinson-n-terms", type=int, default=16,
                   help="probe vectors for --laplacian hutchinson (higher = less variance, more cost)")
    p.add_argument("--db", default="outputs/soft_sphere.db")
    args = p.parse_args()

    conn = db.connect(args.db)
    requeued = db.requeue_interrupted(conn)
    if requeued:
        print(f"requeued {requeued} interrupted run(s)")

    hp = HP(
        n_epochs=args.epochs, n_chains=args.chains,
        chain_length=args.chain_length,
        thermalization_steps=args.thermalization,
        thinning_factor=args.thinning,
        early_stop=not args.no_early_stop,
        es_plateau_rel=args.plateau_rel,
        es_min_epochs=args.es_min_epochs,
        es_check_every=args.es_check_every,
        es_patience=args.es_patience,
        laplacian_method=args.laplacian,
        hutchinson_n_terms=args.hutchinson_n_terms,
        hidden_internal_dim=args.hidden_internal_dim,
        use_jastrow=args.jastrow,
        use_network=not args.no_network,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_final_frac=args.lr_final_frac,
    )
    potentials = [Potential.from_R(R) for R in args.R]  # from_R raises if R <= 1 (HS limit, V0->inf)
    total = 0
    for potential in potentials:
        # feasible_x_grid clips x to this potential's own box ceiling x<N/(8R^3), so each R gets its
        # own feasible range from the shared --x-lo/--x-hi/--n-x (smaller R reaches higher x).
        n = sweep.enqueue_sweep(conn, potential, args.N, args.seeds, hp,
                                n_x=args.n_x, x_lo=args.x_lo, x_hi=args.x_hi)
        total += n
        print(f"  {potential.label}: R={potential.R:g} V0={potential.V0_paper:.6g} -> {n} new todo")
    print(f"enqueued {len(potentials)} potential(s) {[p.label for p in potentials]}: "
          f"{total} new todo points  (N={args.N} seeds={args.seeds} "
          f"x in [{args.x_lo:g}, {args.x_hi:g}] n_x={args.n_x})")
    print(f"queue status: {db.status_counts(conn)}")
    conn.close()  # workers open their own connections

    gpus = args.gpus.split(",") if args.gpus else detect_gpus()
    here = os.path.dirname(os.path.abspath(__file__))

    # Cap host-side CPU threads per worker. Running N unthrottled JAX processes on one node makes
    # them thrash each other on the shared CPU (measured ~28× slowdown with 2 workers); pinning
    # each to a slice of the cores fixes it. Default: cores / n_workers (>=1).
    n_cpu = os.cpu_count() or 8
    per = args.cpu_per_worker if args.cpu_per_worker > 0 else max(1, n_cpu // max(1, len(gpus)))
    print(f"launching {len(gpus)} worker(s) on GPU(s) {gpus}  ({per} CPU threads each of {n_cpu})")

    procs = []
    for g in gpus:
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(g),
            OMP_NUM_THREADS=str(per),
            MKL_NUM_THREADS=str(per),
            OPENBLAS_NUM_THREADS=str(per),
            NUMEXPR_NUM_THREADS=str(per),
        )
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(here, "worker.py"), "--db", args.db], env=env
        ))
    for proc in procs:
        proc.wait()

    conn = db.connect(args.db)
    print(f"all workers done. final queue status: {db.status_counts(conn)}")


if __name__ == "__main__":
    main()
