"""Plot an E/N(x) curve against the paper's benchmarks (all in paper units, a = 1).

Two layers:

* ``plot_curve`` — one (potential, N) E/N(x) curve vs the bounds (finite-N, single ladder rung);
* ``extrapolate_thermodynamic`` / ``plot_extrapolations`` — the finite-size step: at each *fixed* x,
  combine seeds and fit E/N vs 1/N to get the N→∞ value (what actually lands on the paper's curve).

The two-stage reduction is: seeds → one (E/N, err) per (x, N)  [``aggregate_seeds``];
then the N-ladder at fixed x → one (E∞, err) per x  [``extrapolate_thermodynamic``].
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from dilute_gas import first_order_energy_upper_bound, lee_yang_energy_per_particle


def plot_curve(curve: dict, potential, *, scaled: bool = True, ax=None, save: str | None = None):
    """Overlay VMC E/N(x) on lower bound (4πx), Lee-Yang (Eq.1), and Eq.31 upper bound.

    ``scaled=True`` plots E/N / 4πx (the paper's Fig. 1 / Fig. 12 axis), where the rigorous
    lower bound is the horizontal line 1 and shape/correlation effects are the deviation above it.
    """
    x = curve["x"]
    if len(x) == 0:
        raise ValueError("empty curve — no done/passed runs in the DB yet")

    xs = np.geomspace(x.min(), x.max(), 200)
    lb = 4 * math.pi * xs
    ly = np.array([lee_yang_energy_per_particle(xx) for xx in xs])
    ub = np.array([first_order_energy_upper_bound(xx, potential.V0_paper, potential.R) for xx in xs])

    norm = (4 * math.pi * x) if scaled else 1.0
    norm_s = (4 * math.pi * xs) if scaled else 1.0

    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ax.plot(xs, lb / norm_s, "k:", lw=1, label="lower bound  4πx")
    ax.plot(xs, ly / norm_s, "k--", lw=1, label="Lee-Yang (Eq.1)")
    ax.plot(xs, ub / norm_s, color="tab:red", lw=1, label="upper bound (Eq.31)")
    ax.errorbar(x, curve["e_per_n"] / norm, yerr=curve["err"] / norm, fmt="o",
                color="tab:blue", capsize=3, label=f"VMC ({potential.label})")
    ax.set_xscale("log")
    ax.set_xlabel("x = ρa³")
    ax.set_ylabel("E/N / 4πx" if scaled else "E/N  [ℏ²/2ma²]")
    ax.set_title(f"{potential.label}: energy per particle vs gas parameter")
    ax.legend(fontsize=8)
    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=140)
        print(f"saved {save}")
    return ax


# ── finite-size extrapolation: seeds → (x, N) point → N→∞ value ─────────────────────


def aggregate_seeds(rows) -> tuple[float, float, int]:
    """Combine the seeds of one (x, N) into a single (E/N, err, n_seeds), paper units.

    Each seed's ``e_per_n`` is its tail-energy estimate — an independent *variational upper
    bound* on the same E/N. We report their **mean** (the unbiased central estimate; taking the
    single lowest would bias the bound downward and throw away two-thirds of the data). The error
    is the larger of the across-seed scatter ``std/√n`` and the mean within-seed error, so seed
    disagreement (optimisation noise) can never be hidden by small per-run error bars. This is the
    same rule ``sweep.load_curve`` uses, factored out so the extrapolation reuses it.
    """
    e = np.array([r["e_per_n"] for r in rows], float)
    within = np.array([r["err_per_n"] or 0.0 for r in rows], float)
    mean = float(e.mean())
    across = float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else 0.0
    return mean, max(across, float(within.mean())), len(e)


def collect_ladder(conn, potential_label: str, N_list, require_passed: bool = True) -> dict:
    """Group all done runs into ``{x: [(N, e_per_n, err, n_seeds), ...]}`` (seeds already merged).

    ``x`` keys come straight from the DB; for an aligned (shared-grid) sweep they match bit-for-bit
    across N, so the same physical x lands in one bucket. Off-grid legacy points (a different x per
    N) simply land in singleton buckets and are skipped by the fit (needs ≥2 N).
    """
    from collections import defaultdict

    import db

    per_xn: dict = defaultdict(list)
    for N in N_list:
        rows = [r for r in db.fetch_done(conn, potential_label, N)
                if not (require_passed and not r["passed"])]
        by_x: dict = defaultdict(list)
        for r in rows:
            by_x[r["x"]].append(r)
        for x, seed_rows in by_x.items():
            e, err, n = aggregate_seeds(seed_rows)
            per_xn[x].append((N, e, err, n))
    return {x: sorted(per_xn[x]) for x in sorted(per_xn)}


def _wls_intercept(t, y, s):
    """Weighted linear fit y = a + b·t with weights 1/s²; return (a, σ_a, b, σ_b).

    ``a`` is the value at t=0 (here t=1/N, so a = E∞). Errors propagate the supplied s only
    (Gaussian), so they are only as trustworthy as the per-point error bars feeding them.
    """
    s = np.where(s > 0, s, np.nanmedian(s[s > 0]) if np.any(s > 0) else 1.0)
    w = 1.0 / s**2
    S, St, Stt = w.sum(), (w * t).sum(), (w * t * t).sum()
    Sy, Sty = (w * y).sum(), (w * t * y).sum()
    D = S * Stt - St * St
    a = (Stt * Sy - St * Sty) / D
    b = (S * Sty - St * Sy) / D
    return a, math.sqrt(Stt / D), b, math.sqrt(S / D)


def extrapolate_thermodynamic(conn, potential_label: str, N_list,
                              require_passed: bool = True, min_N: int = 2) -> dict:
    """N→∞ extrapolation of E/N at each fixed x: weighted linear fit in 1/N.

    Returns ``{x: fit}`` where each ``fit`` has::

        N, e, err     — the ladder points used (arrays, sorted by N)
        e_inf, e_inf_err  — E/N at 1/N = 0 (the thermodynamic value) and its error
        slope         — coefficient of 1/N (sign/size = strength of the finite-size correction)
        n_points      — how many N went into the fit

    The leading periodic-box finite-size correction to E/N of a homogeneous gas at fixed density
    goes as 1/N, so a straight line in 1/N is the first-order extrapolation; swap ``t`` for a
    different power here if a curvature term is needed. Points with fewer than ``min_N`` rungs are
    skipped (you can't extrapolate one N).
    """
    ladder = collect_ladder(conn, potential_label, N_list, require_passed)
    fits = {}
    for x, pts in ladder.items():
        if len(pts) < min_N:
            continue
        N = np.array([p[0] for p in pts], float)
        e = np.array([p[1] for p in pts], float)
        err = np.array([p[2] for p in pts], float)
        t = 1.0 / N
        a, a_err, b, _ = _wls_intercept(t, e, err)
        fits[x] = {"N": N, "e": e, "err": err, "e_inf": float(a),
                   "e_inf_err": float(a_err), "slope": float(b), "n_points": len(pts)}
    return fits


def plot_extrapolations(fits: dict, potential=None, *, scaled: bool = False,
                        ncols: int = 4, save: str | None = None):
    """One panel per x: E/N (or E/N / 4πx) vs 1/N with the line and the 1/N→0 extrapolate.

    ``fits`` comes from :func:`extrapolate_thermodynamic`. ``scaled=True`` divides by 4πx so the
    Lieb-Yngvason lower bound is 1 (the paper's axis) — handy to see whether the extrapolated point
    is heading for Lee-Yang. Returns the Matplotlib ``Figure``; call from the notebook directly.
    """
    xs = sorted(fits)
    if not xs:
        raise ValueError("no x has ≥2 N — nothing to extrapolate (run the aligned N-ladder first)")
    nrows = math.ceil(len(xs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, x in enumerate(xs):
        f = fits[x]
        ax = axes.flat[k]
        ax.set_visible(True)
        norm = 4 * math.pi * x if scaled else 1.0
        t = 1.0 / f["N"]
        tt = np.linspace(0.0, t.max() * 1.05, 50)
        ax.errorbar(t, f["e"] / norm, yerr=f["err"] / norm, fmt="o", color="tab:blue", capsize=3)
        ax.plot(tt, (f["e_inf"] + f["slope"] * tt) / norm, "-", color="tab:gray", lw=1)
        ax.errorbar([0.0], [f["e_inf"] / norm], yerr=[f["e_inf_err"] / norm], fmt="s",
                    color="tab:red", capsize=4, label="N→∞")
        if scaled and potential is not None:
            ax.axhline(lee_yang_energy_per_particle(x) / norm, ls="--", color="k", lw=0.8)
        ax.set_title(f"x = {x:.3e}", fontsize=8)
        ax.set_xlabel("1/N", fontsize=8)
        ax.set_ylabel("E/N / 4πx" if scaled else "E/N", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Finite-size extrapolation E/N vs 1/N"
                 + (f"  ({potential.label})" if potential is not None else ""))
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140)
        print(f"saved {save}")
    return fig
