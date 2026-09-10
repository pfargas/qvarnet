"""MCMC convergence diagnostics (roadmap §3.1, §3.4) — host/numpy.

These consume the host-side traces in ``MetricsHistory`` (energy over epochs, per-chain
energies), so they are plain numpy — no JIT constraints, data-dependent truncation is easy.
A JIT'd offline IAT/ESS for raw chains already lives in ``samplers/diagnostics.py``.
"""

import numpy as np


def autocorr(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation ρ_k (ρ_0 = 1) of a 1-D trace, via FFT."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.shape[0]
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f), n=2 * n)[:n].real
    return acf / (acf[0] + 1e-12)


def iat_geyer(x: np.ndarray) -> float:
    """Integrated autocorrelation time with Geyer's initial-positive truncation.

    τ_int = 1 + 2 Σ_{k≥1} ρ_k, summed until the first k with ρ_k ≤ 0 (the fixed-window
    estimator adds noise for short traces — §3.1). For AR(1) with parameter φ this recovers
    the exact τ = (1+φ)/(1-φ).
    """
    rho = autocorr(x)
    tau = 1.0
    for k in range(1, rho.shape[0]):
        if rho[k] <= 0.0:
            break
        tau += 2.0 * rho[k]
    return max(float(tau), 1.0)


def ess(x: np.ndarray) -> float:
    """Effective sample size N / τ_int."""
    x = np.asarray(x, dtype=float)
    return x.shape[0] / iat_geyer(x)


def split_rhat(chains: np.ndarray) -> float:
    """Split Gelman-Rubin R̂ for chains of shape ``(m, n)`` (m chains, length n).

    Each chain is split in half (catches a chain that was stuck then unstuck), giving 2m
    half-chains. R̂ ≈ 1 means the chains are indistinguishable; **pass: R̂ ≤ 1.1** (§3.4).
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError(f"chains must be 2-D (m, n), got shape {chains.shape}")
    m, n = chains.shape
    half = n // 2
    if half < 2:
        raise ValueError(f"chains too short to split: n={n}")
    split = np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)  # (2m, half)
    _, n2 = split.shape
    means = split.mean(axis=1)
    variances = split.var(axis=1, ddof=1)
    W = variances.mean()
    B = n2 * means.var(ddof=1)
    var_hat = (n2 - 1) / n2 * W + B / n2
    return float(np.sqrt(var_hat / (W + 1e-12)))
