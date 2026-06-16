"""Milestone 1 driver: reproduce the SS10 E/N(x) curve.

Sweeps a box-feasible x-grid for SS10 at fixed N over a few seeds, then plots E/N(x) against the
lower bound, Lee-Yang, and the Eq.31 upper bound. Resumable: re-running skips completed points.

    uv run --project .. python run_ss10_curve.py            # default N=64, frozen HP
    uv run --project .. python run_ss10_curve.py --tune     # ASHA-tune at an anchor first
"""

from __future__ import annotations

import argparse
import os

import db
from analysis import plot_curve
from point import HP, SS10
from sweep import Frozen, feasible_x_grid, load_curve, sweep_x, tune_then_freeze


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-x", type=int, default=7)
    p.add_argument("--epochs", type=int, default=2000)
    # 1024 chains is the largest that fits an 8 GB GPU at N=64 (2048 OOMs); raise on bigger cards.
    p.add_argument("--chains", type=int, default=1024)
    # sampler: per-epoch Laplacian batch = chains * (chain-length-therm)//thinning.
    # 21/20/1 = keep 1 sample/chain/epoch (warm walkers carry the chain across epochs).
    p.add_argument("--chain-length", type=int, default=21)
    p.add_argument(
        "--thermalization", type=int, default=20, help="must be < chain-length"
    )
    p.add_argument("--thinning", type=int, default=1)
    p.add_argument(
        "--tune", action="store_true", help="ASHA-tune at an anchor x, then freeze"
    )
    # early-stop tuning (--epochs is a ceiling). See run_workers.py for the same flags.
    p.add_argument(
        "--plateau-rel",
        type=float,
        default=0.0,
        help="early-stop on energy plateau (e.g. 0.005); 0=verdict-only",
    )
    p.add_argument("--es-min-epochs", type=int, default=200)
    p.add_argument("--es-check-every", type=int, default=50)
    p.add_argument("--es-patience", type=int, default=2)
    p.add_argument("--no-early-stop", action="store_true", help="run the full --epochs")
    p.add_argument("--db", default="outputs/soft_sphere.db")
    args = p.parse_args()

    out_root = os.path.dirname(args.db) or "."
    conn = db.connect(args.db)
    db.requeue_interrupted(conn)

    xs = feasible_x_grid(SS10, args.N, x_lo=1e-5, x_hi=1e-2, n=args.n_x)
    print(
        f"SS10 feasible x-grid: {xs[0]:.2e} ... {xs[-1]:.2e}  (N={args.N}, n_x={args.n_x})"
    )
    print(
        f" chains={args.chains}, chain-length={args.chain_length}, thermalization={args.thermalization}, thinning={args.thinning}"
    )
    base = HP(
        n_epochs=args.epochs,
        n_chains=args.chains,
        chain_length=args.chain_length,
        thermalization_steps=args.thermalization,
        thinning_factor=args.thinning,
        early_stop=not args.no_early_stop,
        es_plateau_rel=args.plateau_rel,
        es_min_epochs=args.es_min_epochs,
        es_check_every=args.es_check_every,
        es_patience=args.es_patience,
    )

    if args.tune:
        space = [
            base,
            HP(n_epochs=args.epochs, n_chains=args.chains, lr=1e-3),
            HP(
                n_epochs=args.epochs,
                n_chains=args.chains,
                phi_hidden=(128,),
                F_hidden=(128,),
            ),
            HP(n_epochs=args.epochs, n_chains=args.chains, step_size=0.5),
        ]
        anchor = float(xs[len(xs) // 2])
        print(f"ASHA tuning at anchor x={anchor:g} ...")
        strategy = tune_then_freeze(SS10, anchor, args.N, space)
    else:
        strategy = Frozen(base)

    print(
        f"SS10 sweep: N={args.N}, x in [{xs[0]:.1e}, {xs[-1]:.1e}], seeds={args.seeds}"
    )
    sweep_x(conn, SS10, xs, args.N, args.seeds, strategy, out_root=out_root)

    curve = load_curve(conn, "SS10", args.N)
    print(f"\ndone points: {len(curve['x'])}  (seeds/x: {curve['n_per_x']})")
    for x, e, err in zip(curve["x"], curve["e_per_n"], curve["err"]):
        print(
            f"  x={x:.2e}  E/N={e:.6f} +/- {err:.6f}   E/N / 4pi x = {e/(4*3.14159265*x):.4f}"
        )
    plot_curve(curve, SS10, save=f"outputs/ss10_curve_N{args.N}.png")


if __name__ == "__main__":
    main()
