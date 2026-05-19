"""Kinetic energy for VMC — log-space models only.

  T = -1/2 (Delta log|psi| + |grad log|psi||^2)

The Laplacian is pluggable: pass any function from laplacian.py as laplacian_fn.
Default is laplacian_forward_ad (O(dof), exact, recommended).
"""

import jax
import jax.numpy as jnp

from .laplacian import laplacian_forward_ad


def kinetic_log(params, samples, model_apply, laplacian_fn=laplacian_forward_ad, key=None):
    """T = -1/2 (Delta log|psi| + |grad log|psi||^2).

    samples: (batch, dof)
    key:     JAX PRNGKey — only used when laplacian_fn is laplacian_hutchinson; ignored otherwise
    returns: (batch,)
    """
    def log_psi(x):
        return model_apply(params, x[None]).squeeze()  # x: (dof,) -> scalar

    grad_log_psi = jax.vmap(jax.grad(log_psi))(samples)   # (batch, dof)
    lap = laplacian_fn(log_psi, samples, key)               # (batch,)
    return -0.5 * (lap + jnp.sum(grad_log_psi**2, axis=-1))
