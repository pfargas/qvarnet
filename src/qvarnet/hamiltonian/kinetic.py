"""Kinetic energy functions for VMC.

Two physically distinct formulas, both exact:

  kinetic_log    — for log-space models (model outputs log|psi|)
                   T = -1/2 (Delta log|psi| + |grad log|psi||^2)

  kinetic_direct — for direct models (model outputs psi)
                   T = -1/2  Delta psi / psi

The Laplacian is pluggable: pass any function from laplacian.py as laplacian_fn.
Default is laplacian_forward_ad (O(DoF), exact, recommended).
"""

import jax
import jax.numpy as jnp

from .laplacian import laplacian_forward_ad


def kinetic_log(params, samples, model_apply, laplacian_fn=laplacian_forward_ad):
    """T = -1/2 (Delta log|psi| + |grad log|psi||^2).

    For models that output log|psi| directly.

    samples: (batch, DoF)
    returns: (batch,)
    """
    def log_psi(x):
        # x: (DoF,) — model expects (1, DoF)
        return model_apply(params, x[None]).squeeze()  # scalar

    grad_log_psi = jax.vmap(jax.grad(log_psi))(samples)   # (batch, DoF)
    lap = laplacian_fn(log_psi, samples)                    # (batch,)
    return -0.5 * (lap + jnp.sum(grad_log_psi**2, axis=-1))


def kinetic_direct(params, samples, model_apply, laplacian_fn=laplacian_forward_ad):
    """T = -1/2 Delta psi / psi.

    For models that output psi directly (non-log).

    samples: (batch, DoF)
    returns: (batch,)
    """
    def psi(x):
        # x: (DoF,) — model expects (1, DoF)
        return model_apply(params, x[None]).squeeze()  # scalar

    lap = laplacian_fn(psi, samples)          # (batch,)
    psi_vals = jax.vmap(psi)(samples)         # (batch,)
    return -0.5 * lap / psi_vals


# ---------------------------------------------------------------------------
# DEPRECATED — kept for reference only, not used in the training workflow.
# This is NOT the kinetic energy: it is missing the Delta log|psi| term.
# ---------------------------------------------------------------------------
def kinetic_term_divergence_theorem(params, xs, model_apply):
    """DEPRECATED. Gradient-only approximation: T ~ +1/2 |grad log psi|^2.

    Omits the Laplacian term — physically incomplete.
    """
    def log_psi_fn(x):
        x = jnp.atleast_1d(x).reshape(1, -1)
        psi = model_apply(params, x).squeeze()
        return jnp.log(jnp.abs(psi) + 1e-12)

    grad_val = jax.vmap(jax.grad(log_psi_fn))(xs)    # (batch, DoF)
    return 0.5 * jnp.sum(grad_val**2, axis=-1)
