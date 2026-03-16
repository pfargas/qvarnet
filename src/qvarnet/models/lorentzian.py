from flax import linen as nn
from jax import numpy as jnp


class WavefunctionOneParameter(nn.Module):

    @nn.compact
    def __call__(self, x):
        alpha = self.param("alpha", nn.initializers.constant(0.5), ())
        return 1 / (alpha**2 + x**2)
