"""The three-referee verdict (roadmap §3) — host/numpy.

A run is "done" only when three independent referees agree on the tail of the run:

1. **Stationary** — energy trace has no residual drift (Heidelberger-Welch + Geweke).
2. **At the Monte-Carlo floor** — the residual fluctuation is sampling-limited, i.e.
   comparable to the MC error of the mean, not optimiser-limited.
3. **Chains mixed** — within-run split-R̂ on the per-chain energies ≤ 1.1 (no walkers trapped
   in different modes). The *seed-safe* referee proper (multi-seed R̂) is the step-6
   ``multi_seed_run``; this is its within-run cousin.
"""

import numpy as np

from .mcmc import split_rhat
from .stationarity import geweke_z, heidelberger_welch_t


def v_score(history, n_particles: int, e_inf: float = 0.0, tail_frac: float = 0.5) -> float:
    """Dimensionless V-score = N·Var(E_loc) / (Ē − E_∞)² (arXiv:2302.04919), tail-averaged.

    Comparable across systems; smaller is better (→ 0 at an eigenstate)."""
    energy = history.get("energy")
    std = history.get("std")
    n = len(energy)
    k = max(int(tail_frac * n), 1)
    e_tail = float(np.mean(energy[-k:]))
    var_tail = float(np.mean(std[-k:] ** 2))
    return n_particles * var_tail / ((e_tail - e_inf) ** 2 + 1e-12)


def three_referee_verdict(
    history,
    tail_frac: float = 0.5,
    rhat_threshold: float = 1.1,
    z_thr: float = 3.0,
    t_thr: float = 2.0,
    mc_floor_factor: float = 2.0,
) -> dict:
    """Run all three referees on the tail of a ``MetricsHistory``; return a verdict dict."""
    energy = history.get("energy")
    err = history.get("error_of_mean")
    n = len(energy)
    k = max(int(tail_frac * n), min(n, 8))
    e_tail = energy[-k:]
    err_tail = err[-k:]

    # 1. stationarity
    z = geweke_z(e_tail)
    t = heidelberger_welch_t(e_tail)
    stationary = bool(abs(z) < z_thr and abs(t) < t_thr)

    # 2. MC floor: tail residual to the best energy vs the MC error of the mean
    e_best = float(np.min(energy))
    residual = float(np.mean(np.abs(e_tail - e_best)))
    err_mean = float(np.mean(err_tail))
    at_mc_floor = bool(residual <= mc_floor_factor * err_mean)

    # 3. within-run split-R̂ on per-chain energies (chains mixed?)
    rhat = None
    chains_mixed = None
    try:
        E_chain = history.get("E_chain")  # (n_epochs, n_chains)
        if E_chain.ndim == 2 and E_chain.shape[1] >= 2:
            tail_chains = E_chain[-k:].T  # (n_chains, window)
            rhat = split_rhat(tail_chains)
            chains_mixed = bool(rhat <= rhat_threshold)
    except (KeyError, ValueError):
        pass

    passed = stationary and at_mc_floor and (chains_mixed is not False)
    return {
        "stationary": stationary,
        "geweke_z": z,
        "hw_t": t,
        "at_mc_floor": at_mc_floor,
        "tail_energy": float(np.mean(e_tail)),
        "tail_residual": residual,
        "tail_error_of_mean": err_mean,
        "chains_mixed": chains_mixed,
        "split_rhat": rhat,
        "passed": bool(passed),
    }


def format_verdict(v: dict) -> str:
    """Human-readable one-block summary of a verdict dict (the end-of-run artifact)."""

    def mark(ok):
        return "PASS" if ok else ("n/a" if ok is None else "FAIL")

    rhat = "n/a" if v["split_rhat"] is None else f"{v['split_rhat']:.3f}"
    return (
        "three-referee verdict\n"
        f"  1. stationary    : {mark(v['stationary'])}  (|z|={abs(v['geweke_z']):.2f} < z_thr, "
        f"|t|={abs(v['hw_t']):.2f} < t_thr)\n"
        f"  2. at MC floor   : {mark(v['at_mc_floor'])}  (tail |E-E_best|={v['tail_residual']:.2e} "
        f"vs err={v['tail_error_of_mean']:.2e})\n"
        f"  3. chains mixed  : {mark(v['chains_mixed'])}  (split-R̂={rhat})\n"
        f"  tail energy      : {v['tail_energy']:.6f}\n"
        f"  => {'CONVERGED' if v['passed'] else 'NOT converged'}"
    )
