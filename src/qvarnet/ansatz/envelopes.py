from flax import linen as nn
from jax import numpy as jnp


class PolynomialEnvelope(nn.Module):
    """Log-space polynomial envelope: -α² · Σᵢ xᵢᵖ.

    Applied to the *raw* (pre-transform) coordinates.
    Combine with any network via ``LogWavefunction``.
    """

    power: int = 4
    init: float = 0.1

    @nn.compact
    def __call__(self, x):
        alpha = self.param("alpha", nn.initializers.constant(self.init), ())
        return -(alpha**2) * jnp.sum(x**self.power, axis=-1, keepdims=True)


class GaussianEnvelope(nn.Module):
    """Log-space Gaussian envelope: -α² · Σᵢ xᵢ².

    Equivalent to ``PolynomialEnvelope(power=2)``; kept as a named alias for clarity.
    """

    init: float = 0.1

    @nn.compact
    def __call__(self, x):
        alpha = self.param("alpha", nn.initializers.constant(self.init), ())
        return -(alpha**2) * jnp.sum(x**2, axis=-1, keepdims=True)
