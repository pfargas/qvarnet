from flax import linen as nn
from jax import numpy as jnp


class LogJastrow(nn.Module):
    """Bosonic Jastrow factor in log space: λ · Σᵢ<ⱼ log|xᵢ − xⱼ|.

    This is the 1D Calogero-Sutherland-type Jastrow. Applied to raw
    coordinates; combine with a network via ``LogWavefunction``.

    Parameters
    ----------
    n_particles:
        Number of particles.  Used to build the upper-triangle mask.
    """

    n_particles: int
    lambda_init: float = 1.0

    @nn.compact
    def __call__(self, x):
        # x: (..., n_particles)  — 1D coordinates (raw, pre-transform)
        lam = self.param("lambda", nn.initializers.constant(self.lambda_init), ())
        xi = x[..., :, None]   # (..., n, 1)
        xj = x[..., None, :]   # (..., 1, n)
        mask = jnp.triu(jnp.ones((self.n_particles, self.n_particles), bool), k=1)
        log_r = jnp.where(mask, jnp.log(jnp.abs(xi - xj) + 1e-10), 0.0)
        return (lam * jnp.sum(log_r, axis=(-2, -1)))[..., None]
