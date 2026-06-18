"""Sweep orchestration: nest shape -> x -> N -> seed over ``run_point``, resumably.

The HP axis is pluggable (``HPStrategy``): ``Frozen`` (tune-then-freeze) or ``ASHA`` (adaptive
per point, the default intent). The recommended pattern for a curve is ``tune_then_freeze`` — run
ASHA once at an anchor ``x`` and freeze the winner across the sweep — cheap and good enough.

Milestone 1: ``sweep_x`` over a box-feasible x-grid for one potential at fixed ``N``; ``load_curve``
aggregates seeds (verdict-gated) into one ``E/N(x)`` point each.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import replace
from typing import Protocol

import artifacts
import db
import numpy as np
from dilute_gas import box_side_for_gas_parameter
from point import HP, Potential, box_fits_interaction, run_point

# ── HP strategy ────────────────────────────────────────────────────────────────────


class HPStrategy(Protocol):
    def best_hp(self, potential: Potential, x: float, N: int) -> HP: ...


class Frozen:
    """Use one fixed HP everywhere (tune-then-freeze, once the anchor tune is done)."""

    def __init__(self, hp: HP):
        self.hp = hp

    def best_hp(self, potential, x, N) -> HP:
        return self.hp


class ASHA:
    """Successive halving over a list of candidate HPs (re-trains at increasing epoch budgets).

    Selection metric (lower=better): ``e_per_n + err_per_n`` among verdict-passing trials — the
    variational principle (all energies are upper bounds) with an uncertainty penalty. No
    checkpoint resume, so each rung re-trains from scratch; fine for the budgets we use.
    """

    def __init__(self, space: list[HP], rung_epochs=(60, 180, 400), eta: int = 3, seed: int = 0):
        self.space = space
        self.rung_epochs = rung_epochs
        self.eta = eta
        self.seed = seed

    def best_hp(self, potential, x, N) -> HP:
        survivors = list(self.space)
        for budget in self.rung_epochs:
            scored = []
            for hp in survivors:
                r = run_point(potential, x, N, self.seed, replace(hp, n_epochs=budget))
                penalty = 0.0 if r.passed else 1e3  # demote unconverged trials
                scored.append((r.e_per_n + r.err_per_n + penalty, hp))
            scored.sort(key=lambda t: t[0])
            keep = max(1, len(scored) // self.eta)
            survivors = [hp for _, hp in scored[:keep]]
            if len(survivors) == 1:
                break
        return survivors[0]


def tune_then_freeze(potential, anchor_x, N, space, **asha_kw) -> Frozen:
    """ASHA-tune at one anchor x, then freeze the winner for the whole sweep."""
    best = ASHA(space, **asha_kw).best_hp(potential, anchor_x, N)
    return Frozen(best)


# ── the runs ──────────────────────────────────────────────────────────────────────


def execute_claimed(conn, potential, x, N, seed, hp, *, out_root: str) -> None:
    """Train one already-claimed (status ``running``) point, write artifacts, save the result.

    Shared by the serial driver (`run_one`) and the parallel `worker.py`. Assumes the row is
    already marked ``running`` (by `mark_running` or `db.claim_next`); does not re-check status.
    Writes the run dir (history/params/meta/verdict) under ``out_root`` and links it via ``run_dir``.
    On failure the row is marked ``failed`` (with the traceback) and the exception re-raised.
    """
    rel = os.path.join(artifacts.RUNS_SUBDIR, artifacts.run_id(potential, x, N, seed, hp))
    run_path = os.path.join(out_root, rel)
    os.makedirs(run_path, exist_ok=True)
    try:
        result = run_point(potential, x, N, seed, hp, checkpoint_dir=run_path)
        artifacts.write_run_artifacts(out_root, result)
        db.save_result(conn, result, run_dir=rel)
    except Exception:
        db.mark_failed(conn, potential, x, N, seed, hp, traceback.format_exc())
        raise


def run_one(conn, potential, x, N, seed, hp, *, out_root: str, force: bool = False) -> None:
    """Run a single point unless already done (resumable, serial). Records status + artifacts."""
    if not force and db.status_of(conn, potential, x, N, seed, hp) == "done":
        return
    if not box_fits_interaction(potential, x, N):
        db.mark_skipped_box(conn, potential, x, N, seed, hp, box_side_for_gas_parameter(x, 1.0, N))
        return
    db.enqueue(conn, potential, x, N, seed, hp)
    db.mark_running(conn, potential, x, N, seed, hp)
    execute_claimed(conn, potential, x, N, seed, hp, out_root=out_root)


def sweep_x(conn, potential, xs, N, seeds, strategy: HPStrategy, *,
            out_root: str = ".", force: bool = False) -> None:
    """Sweep gas parameter x at fixed (potential, N), over seeds, with a HP strategy per x."""
    os.makedirs(out_root, exist_ok=True)
    for x in xs:
        if not box_fits_interaction(potential, x, N):
            for seed in seeds:
                db.mark_skipped_box(conn, potential, x, N, seed, strategy.best_hp(potential, x, N),
                                    box_side_for_gas_parameter(x, 1.0, N))
            print(f"[skip] {potential.label} x={x:g} N={N}: R>=L/2 (box too small)")
            continue
        hp = strategy.best_hp(potential, x, N)
        for seed in seeds:
            run_one(conn, potential, x, N, seed, hp, out_root=out_root, force=force)
            row = db.status_of(conn, potential, x, N, seed, hp)
            print(f"[{row}] {potential.label} x={x:g} N={N} seed={seed}")


# ── aggregation: seeds -> one E/N(x) point ──────────────────────────────────────────


def load_curve(conn, potential_label: str, N: int, require_passed: bool = True) -> dict:
    """Aggregate verdict-passing seeds into one E/N per x. Returns sorted arrays.

    Per x: combine seeds. If >1 seed, the error is the larger of (spread across seeds)/sqrt(n)
    and the mean within-seed error — so it can't understate the uncertainty.
    """
    rows = db.fetch_done(conn, potential_label, N)
    by_x: dict[float, list] = {}
    for r in rows:
        if require_passed and not r["passed"]:
            continue
        by_x.setdefault(r["x"], []).append(r)

    xs, e, err, ub = [], [], [], []
    for x in sorted(by_x):
        seeds = by_x[x]
        evals = np.array([s["e_per_n"] for s in seeds])
        within = np.array([s["err_per_n"] for s in seeds])
        mean = float(evals.mean())
        across = float(evals.std(ddof=1) / np.sqrt(len(evals))) if len(evals) > 1 else 0.0
        xs.append(x)
        e.append(mean)
        err.append(max(across, float(within.mean())))
        ub.append(seeds[0]["upper_bound"])
    return {"x": np.array(xs), "e_per_n": np.array(e), "err": np.array(err),
            "upper_bound": np.array(ub), "n_per_x": [len(by_x[x]) for x in sorted(by_x)]}


def feasible_x_grid(potential, N, x_lo=1e-5, x_hi=1e-2, n=8) -> np.ndarray:
    """Log-spaced x grid clipped to the box-feasible range R < L/2  <=>  x < N/(8 R^3)."""
    x_max_box = N / (8.0 * potential.R**3)
    grid = np.geomspace(x_lo, min(x_hi, 0.999 * x_max_box), n)
    return grid


# ── work-queue planning (for the parallel, worker-per-GPU runner) ────────────────────


def enqueue_sweep(conn, potential, N_list, seeds, hp, *, n_x=7, x_lo=1e-5, x_hi=1e-2) -> int:
    """Insert all ``todo`` rows for a (potential, N-ladder, seeds, hp) sweep; return #enqueued.

    One row per ``(potential, x, N, seed, hp)`` over a **single shared x-grid** used for every N in
    the ladder. The grid is clipped to the box ceiling ``x < N/(8R^3)`` of the *smallest* N (the most
    restrictive), so every x is feasible at every N — this is what makes the fixed-x, multi-N data a
    clean set for the ``1/N -> 0`` finite-size extrapolation (the whole point of an N-ladder). A
    per-N grid (the old behaviour) would put each N on different x and leave nothing to extrapolate.
    Idempotent (``INSERT OR IGNORE``), so it is safe to re-run to extend a sweep — already present
    points keep their status. Workers then drain the queue via :func:`db.claim_next`.
    """
    n = 0
    xs = feasible_x_grid(potential, min(N_list), x_lo=x_lo, x_hi=x_hi, n=n_x)  # shared across all N
    for N in N_list:
        for x in (float(v) for v in xs):
            for seed in seeds:
                if not box_fits_interaction(potential, x, N):  # safety net (grid is clipped to min N)
                    db.mark_skipped_box(conn, potential, x, N, seed, hp,
                                        box_side_for_gas_parameter(x, 1.0, N))
                    continue
                db.enqueue(conn, potential, x, N, seed, hp)
                n += 1
    return n
