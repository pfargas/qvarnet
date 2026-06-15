from flax import linen as nn
from jax import numpy as jnp


class LogJastrow(nn.Module):
    """Bosonic Jastrow factor in log space.

    Open boundary (``L is None``, default):
        log J = λ · Σᵢ<ⱼ log|xᵢ − xⱼ|         (1D Calogero-Sutherland-type)

    Periodic box of side ``L`` (``L`` set):
        log J = λ · Σᵢ<ⱼ log|sin(π (xᵢ − xⱼ) / L)|   (Sutherland, on a ring)

    The Sutherland form is the *exactly L-periodic* analogue: it is invariant under
    xₖ → xₖ + L and smooth everywhere except the physical coincidence cusp — unlike a
    minimum-image ``log|xᵢ−xⱼ|`` which would acquire a spurious derivative kink at L/2.

    Applied to raw coordinates; combine with a network via ``LogWavefunction``.

    Parameters
    ----------
    n_particles:
        Number of particles.  Used to build the upper-triangle mask.
    L:
        Box length.  ``None`` (default) = open boundary; a float enables the
        periodic Sutherland form.
    """

    n_particles: int
    lambda_init: float = 1.0
    L: float | None = None

    @nn.compact
    def __call__(self, x):
        # x: (..., n_particles)  — 1D coordinates (raw, pre-transform)
        lam = self.param("lambda", nn.initializers.constant(self.lambda_init), ())
        xi = x[..., :, None]   # (..., n, 1)
        xj = x[..., None, :]   # (..., 1, n)
        dx = xi - xj
        if self.L is None:
            r = jnp.abs(dx)
        else:
            r = jnp.abs(jnp.sin(jnp.pi * dx / self.L))  # exactly L-periodic
        mask = jnp.triu(jnp.ones((self.n_particles, self.n_particles), bool), k=1)
        log_r = jnp.where(mask, jnp.log(r + 1e-10), 0.0)
        return (lam * jnp.sum(log_r, axis=(-2, -1)))[..., None]
