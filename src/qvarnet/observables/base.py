"""Observable infrastructure (roadmap §6) — host/numpy.

Observables are evaluated **post-training** from a batch of configurations drawn from |ψ|²
(use the IAT machinery to set thinning). For an operator diagonal in position space the
estimator is just a sample average ⟨Ô⟩ = (1/M) Σ_i O(R_i).

Error bars on correlated data come from **blocking** (Flyvbjerg-Petersen): recursively average
adjacent pairs and take the error where it plateaus vs block level. One implementation, reused
by every observable.
"""

import numpy as np


def blocking_error(x) -> tuple[float, list]:
    """Flyvbjerg-Petersen blocking error of the mean of a 1-D correlated series.

    Returns ``(error, errors_by_level)``. The reported error is the maximum over block
    levels — the plateau value the standard error converges to as correlations are averaged
    out. (For i.i.d. data it is flat ≈ σ/√N from the first level.)
    """
    x = np.asarray(x, dtype=float).ravel()
    errors = []
    while x.shape[0] >= 2:
        n = x.shape[0]
        errors.append(float(np.sqrt(np.var(x, ddof=1) / n)))
        if n % 2:
            x = x[:-1]
        x = 0.5 * (x[0::2] + x[1::2])
    if not errors:
        return 0.0, errors
    return max(errors), errors


def mean_and_error(x) -> tuple[float, float]:
    """Sample mean and blocking error of a 1-D series."""
    x = np.asarray(x, dtype=float).ravel()
    return float(x.mean()), blocking_error(x)[0]
