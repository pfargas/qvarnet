"""Post-run analysis: convergence diagnostics and property estimators.

Host/numpy functions over ``MetricsHistory`` traces:
- ``mcmc``: Geyer IAT, ESS, split-R̂.
- ``stationarity``: Geweke and Heidelberger-Welch referees.
- ``stopper``: ``StationarityStopper`` early-stop callback.
- ``verdict``: the combined three-referee verdict + V-score.

Still to come (step-4 follow-up): gradient norms/SNR, θ-step dead-region maps, QGT spectrum.
"""

from qvarnet.analysis.compare import welch_t_test
from qvarnet.analysis.dashboard import plot_dashboard
from qvarnet.analysis.gradients import global_grad_norm, gradient_snr, per_layer_grad_norms
from qvarnet.analysis.mcmc import autocorr, ess, iat_geyer, split_rhat
from qvarnet.analysis.parameters import dead_fraction, global_theta_ratio, theta_ratios
from qvarnet.analysis.qgt_spectrum import d_eff, d_part, qgt_eigenvalues
from qvarnet.analysis.stationarity import geweke_z, heidelberger_welch_t, is_stationary
from qvarnet.analysis.verdict import format_verdict, three_referee_verdict, v_score

__all__ = [
    "autocorr",
    "iat_geyer",
    "ess",
    "split_rhat",
    "geweke_z",
    "heidelberger_welch_t",
    "is_stationary",
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
