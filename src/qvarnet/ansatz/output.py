"""Model output convention: every model returns ``(log_abs, phase)``.

    ψ(x) = exp(log_abs(x) + i·phase(x)),   log_abs, phase: (...,)

Ground state with a real Hamiltonian: phase ∈ {0, π}, set by the ansatz's structure
(e.g. the sign of a Slater determinant), never trained. Since that sign is piecewise
constant, sampling, kinetic energy, O_k and the QGT need only ``log_abs``; the phase
enters through ratios ψ(x')/ψ(x) at different configurations (overlaps, OBDM, tests).

A model returning a single array is treated as **bosonic**: that array is log|ψ| and
phase ≡ 0. Allowed so custom/legacy models keep working, but it warns.
"""

import warnings

import jax.numpy as jnp


def split_log_psi(out):
    """Model output → ``(log_abs, phase)``, each ``(...,)``."""
    if isinstance(out, tuple):
        log_abs, phase = out
        assert log_abs.shape == phase.shape, (
            f"log_abs {log_abs.shape} and phase {phase.shape} must have the same shape"
        )
        return log_abs, phase
    warnings.warn(
        "Model returns a single array: treating it as bosonic log|ψ| with phase 0. "
        "Return (log_abs, phase) to silence this.",
        stacklevel=2,
    )
    # Legacy outputs are (..., 1) or (...); drop the trailing singleton.
    # ponytail: a (...,) output with batch size 1 also loses that axis; callers squeeze anyway.
    log_abs = out[..., 0] if out.ndim >= 1 and out.shape[-1] == 1 else out
    return log_abs, jnp.zeros_like(log_abs)


def log_abs_of(model_apply):
    """Wrap ``model_apply`` so it returns only ``log_abs`` ``(...,)``."""

    def apply(params, x):
        return split_log_psi(model_apply(params, x))[0]

    return apply
