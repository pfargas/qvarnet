"""Stationarity referees (roadmap §3.2, §3.3) — host/numpy.

Two independent tests on a single trace, both deflating their sample count by the integrated
autocorrelation time (correlated steps are not independent observations):

- **Geweke** split test — early vs late segment means agree if stationary.
- **Heidelberger-Welch** slope test — OLS trend with the standard error inflated by τ_int;
  catches slow monotone drift Geweke can miss.

A trace is "stationary" when both pass (|z| < z_thr and |t| < t_thr).
"""

import numpy as np

from .mcmc import iat_geyer


def geweke_z(x: np.ndarray, first: float = 0.1, last: float = 0.5) -> float:
    """Geweke z-score between the first ``first`` and last ``last`` fractions of the trace.

    Under stationarity |z| = O(1); **|z| ≳ 3 ⇒ still drifting**. Each segment's sample count
    is deflated by its own τ_int.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    a = x[: max(int(first * n), 2)]
    b = x[n - max(int(last * n), 2) :]
    na = len(a) / iat_geyer(a)
    nb = len(b) / iat_geyer(b)
    num = a.mean() - b.mean()
    den = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    return float(num / (den + 1e-12))


def heidelberger_welch_t(x: np.ndarray) -> float:
    """HW slope t-statistic: OLS slope / SE, with SE inflated by τ_int of the residuals.

    **|t| < 2 ⇒ no significant trend.** Sign of the slope tells direction (negative = still
    improving → keep training).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    t = np.arange(n, dtype=float)
    tbar, xbar = t.mean(), x.mean()
    Stt = np.sum((t - tbar) ** 2)
    slope = np.sum((t - tbar) * (x - xbar)) / (Stt + 1e-12)
    resid = x - (xbar + slope * (t - tbar))
    s2 = np.sum(resid**2) / max(n - 2, 1)
    se_ols = np.sqrt(s2 / (Stt + 1e-12))
    se = se_ols * np.sqrt(iat_geyer(resid))
    return float(slope / (se + 1e-12))


def is_stationary(x: np.ndarray, z_thr: float = 3.0, t_thr: float = 2.0) -> bool:
    """Both referees agree the trace is stationary."""
    return abs(geweke_z(x)) < z_thr and abs(heidelberger_welch_t(x)) < t_thr
