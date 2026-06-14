"""Geometry package: the quantum geometric tensor (QGT / S-matrix).

Shared between methods because the same S is the stochastic-reconfiguration
preconditioner (VMC) and the TDVP metric (future t-VMC). Solver registry:
cholesky / direct / gmres / diagonal today; minSR (M×M Gram dual) later.
"""

from .qgt import (
    DEFAULT_QGT_CONFIG,
    LARGE_SYSTEM_QGT_CONFIG,
    MEMORY_EFFICIENT_QGT_CONFIG,
    QGTConfig,
    compute_log_derivatives,
    compute_natural_gradient,
    compute_natural_gradient_minsr,
    compute_qgt,
)
from .tdvp import imaginary_time_step, tdvp_force, tdvp_residual

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
