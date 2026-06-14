"""Backward-compat shim — QGT moved to ``qvarnet.geometry.qgt``."""

from .geometry.qgt import (  # noqa: F401
    DEFAULT_QGT_CONFIG,
    LARGE_SYSTEM_QGT_CONFIG,
    MEMORY_EFFICIENT_QGT_CONFIG,
    QGTConfig,
    compute_log_derivatives,
    compute_natural_gradient,
    compute_natural_gradient_minsr,
    compute_qgt,
    solve_qgt_cholesky,
    solve_qgt_diagonal,
    solve_qgt_direct,
    solve_qgt_gmres,
)
