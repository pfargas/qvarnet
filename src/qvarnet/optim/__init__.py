"""Optimisation geometry: the quantum geometric tensor, TDVP, auxiliary losses.

The same S-matrix is the stochastic-reconfiguration preconditioner (VMC) and the
TDVP metric, so it is shared rather than owned by either.
"""

from qvarnet.optim.qgt import (
    DEFAULT_QGT_CONFIG,
    LARGE_SYSTEM_QGT_CONFIG,
    MEMORY_EFFICIENT_QGT_CONFIG,
    QGTConfig,
    compute_log_derivatives,
    compute_natural_gradient,
    compute_natural_gradient_minsr,
    compute_qgt,
)
from qvarnet.optim.tdvp import imaginary_time_step, tdvp_force, tdvp_residual

__all__ = [
    "QGTConfig",
    "DEFAULT_QGT_CONFIG",
    "MEMORY_EFFICIENT_QGT_CONFIG",
    "LARGE_SYSTEM_QGT_CONFIG",
    "compute_qgt",
    "compute_log_derivatives",
    "compute_natural_gradient",
    "compute_natural_gradient_minsr",
    "tdvp_force",
    "imaginary_time_step",
    "tdvp_residual",
]
