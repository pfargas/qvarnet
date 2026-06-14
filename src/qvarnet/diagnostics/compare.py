"""Honest design comparison (roadmap §5.3) — Welch's unequal-variance t-test.

When you change anything (architecture, lr, sampler) and want to claim an improvement: take one
tail-mean energy per seed for designs A and B, then Welch's t-test. A small two-sided p means
the difference is real, not sampling luck. This is the referee for the whole hyperparameter
campaign — the same statistics the §8e queue uses to promote configs.
"""

import numpy as np
from scipy import stats


def welch_t_test(a, b) -> dict:
    """Welch's t-test between samples ``a`` and ``b`` (e.g. per-seed tail energies).

    Returns ``{t, dof, p, mean_a, mean_b}`` with a two-sided p-value (Welch-Satterthwaite dof).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        raise ValueError("Welch's t-test needs at least 2 samples per group")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / na + vb / nb
    t = (a.mean() - b.mean()) / (np.sqrt(se2) + 1e-300)
    dof = se2**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1) + 1e-300)
    p = float(2.0 * stats.t.sf(abs(t), dof))
    return {
        "t": float(t),
        "dof": float(dof),
        "p": p,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
    }
