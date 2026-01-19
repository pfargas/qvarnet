from flax import linen as nn
from jax import numpy as jnp


class ExponentialWavefunction(nn.Module):

    @nn.compact
    def __call__(self, x):
        alpha = self.param("alpha", nn.initializers.constant(1.0), ())
        return jnp.exp(-alpha * jnp.sum(x**2, axis=-1))
