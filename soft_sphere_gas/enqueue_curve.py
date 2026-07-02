"""Enqueue an E/N(x) curve sweep: the box-feasible log-spaced x grid, per potential.

This is the physics-aware grid planner the generic ``runq enqueue`` can't do by itself:
the x grid must be clipped to the box ceiling ``x < N/(8R^3)`` — computed with the
*smallest* N of the ladder, so every x is feasible at every N (what makes the fixed-x
``1/N -> 0`` extrapolation possible). Solver settings ride along as ordinary axes.

    python enqueue_curve.py --R 10 --N 32 64 128 --seeds 0 1 2 --n-x 7 \
        --set lr=3e-3 --set n_epochs=2000 --set use_jastrow=true
    runq run point.py --db outputs/runq.db --gpus 0,1        # then drain

Idempotent: re-run to extend a sweep; existing points keep their status.
"""

from __future__ import annotations

import argparse

from point import Potential, feasible_x_grid, run_point
from runq import ParamSpace, build_grid, connect, key_json, run_label, store


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--R", type=float, nargs="+", default=[10.0],
                    help="soft-sphere core range(s), e.g. --R 10 5 (R>1; R=1 is the HS limit)")
    ap.add_argument("--N", type=int, nargs="+", default=[64], help="particle ladder")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-x", type=int, default=7)
    ap.add_argument("--x-lo", type=float, default=1e-5)
    ap.add_argument("--x-hi", type=float, default=1e-2)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="fix any solver parameter for the whole sweep (repeatable)")
    ap.add_argument("--db", default=store.DEFAULT_DB)
    args = ap.parse_args()

    space = ParamSpace.from_function(run_point)
    fixed = {}
    for spec in args.set:
        name, _, raw = spec.partition("=")
        fixed[name.strip()] = space.coerce(name.strip(), raw)

    conn = connect(args.db)
    total_new = 0
    for R in args.R:
        potential = Potential.from_R(R)  # fails fast on R <= 1
        # shared x grid, clipped by the smallest N (most restrictive box ceiling)
        xs = [float(v) for v in feasible_x_grid(potential, min(args.N),
                                                x_lo=args.x_lo, x_hi=args.x_hi, n=args.n_x)]
        axes = {"R": [R], "x": xs, "N": args.N, "seed": args.seeds,
                **{k: [v] for k, v in fixed.items()}}
        swept = ["R", "x", "N", "seed"]
        n = 0
        for params in build_grid(space, axes):
            n += store.enqueue(conn, key_json(params), run_label(params, swept))
        total_new += n
        print(f"  SS{R:g}: {len(xs)} x-points in [{xs[0]:.3e}, {xs[-1]:.3e}] -> {n} new todo")

    print(f"enqueued {total_new} new point(s); status={store.status_counts(conn)}")
    print(f"drain with:  runq run point.py --db {args.db}   (add --gpus 0,1 ...)")
    conn.close()


if __name__ == "__main__":
    main()
