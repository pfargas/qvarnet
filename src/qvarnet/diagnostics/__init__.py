"""Diagnostics package (roadmap step 4): the three-referee convergence suite.

Host/numpy functions over ``MetricsHistory`` traces:
- ``mcmc``: Geyer IAT, ESS, split-R̂.
- ``stationarity``: Geweke and Heidelberger-Welch referees.
- ``stopper``: ``StationarityStopper`` early-stop callback.
- ``verdict``: the combined three-referee verdict + V-score.

Still to come (step-4 follow-up): gradient norms/SNR, θ-step dead-region maps, QGT spectrum.
"""

from .compare import welch_t_test
from .dashboard import plot_dashboard
from .gradients import global_grad_norm, gradient_snr, per_layer_grad_norms
from .mcmc import autocorr, ess, iat_geyer, split_rhat
from .parameters import dead_fraction, global_theta_ratio, theta_ratios
from .qgt_spectrum import d_eff, d_part, qgt_eigenvalues
from .stationarity import geweke_z, heidelberger_welch_t, is_stationary
from .stopper import StationarityStopper
from .verdict import format_verdict, three_referee_verdict, v_score

__all__ = [
    "autocorr",
    "iat_geyer",
    "ess",
    "split_rhat",
    "geweke_z",
    "heidelberger_welch_t",
    "is_stationary",
    "StationarityStopper",
    "three_referee_verdict",
    "format_verdict",
    "v_score",
    "global_grad_norm",
    "per_layer_grad_norms",
    "gradient_snr",
    "global_theta_ratio",
    "theta_ratios",
    "dead_fraction",
    "qgt_eigenvalues",
    "d_eff",
    "d_part",
    "welch_t_test",
    "plot_dashboard",
]
