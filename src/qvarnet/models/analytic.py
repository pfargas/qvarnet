from typing import Callable

import jax
from flax import linen as nn
from jax import numpy as jnp

from .base import BaseModel
from .layers.custom_dense import CustomDense
from .mlp import MLP
from .registry import register_model

@register_model("CS-analytic")
class CalogeroSutherlandAnalyticModel(BaseModel):
    """Exact log-wavefunction for the Calogero model (IS_LOG_MODEL=True).

    log|ψ₀| = λ · Σᵢ<ⱼ log|xᵢ-xⱼ|  −  ω/2 · Σᵢ xᵢ²

    Exact ground-state energy: E₀ = Nω/2 + ωλN(N-1)/2

    Input:  (..., n_particles)  — 1D particles, batched or unbatched
    Output: (...)               — log|ψ₀|, one scalar per configuration
    """
    @nn.compact
    def __call__(self, x):
        # x: (..., n_particles)  — works for (batch, N) and bare (N,)
        lam   = self.param("lam",   nn.initializers.constant(2.0), ())
        omega = self.param("omega", nn.initializers.constant(1.0), ())

        n = x.shape[-1]  # n_particles lives in the LAST dim, not dim 0

        # Pairwise |xᵢ - xⱼ|: (..., N, 1) vs (..., 1, N) → (..., N, N)
        diffs = jnp.abs(x[..., :, None] - x[..., None, :]) + 1e-12

        # Upper-triangle mask: selects i<j only (excludes diagonal and lower half)
        mask = jnp.triu(jnp.ones((n, n)), k=1)  # (N, N)

        # Jastrow in log-space: λ · Σᵢ<ⱼ log|xᵢ-xⱼ|  →  (...)
        log_jastrow = lam * jnp.sum(mask * jnp.log(diffs), axis=(-1, -2))

        # Harmonic envelope: -ω/2 · Σᵢ xᵢ²  →  (...)
        log_gaussian = -omega / 2 * jnp.sum(x**2, axis=-1)

        return log_jastrow + log_gaussian  # (...) — log|ψ₀|