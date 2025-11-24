from flax import linen as nn
from jax import numpy as jnp


# Define a simple MLP using Flax Linen
class MLP(nn.Module):
    architecture: list
    hidden_activation: callable = nn.tanh
    alpha: float = 1.0  # parameter for the final wavefunction

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.architecture) - 1):
            x = nn.Dense(features=self.architecture[i + 1])(x)
            if i < len(self.architecture) - 2:
                x = self.hidden_activation(x)

        return x


class WavefunctionOneParameter(nn.Module):

    @nn.compact
    def __call__(self, x):
        alpha = self.param("alpha", nn.initializers.constant(0.5), ())
        return 1 / (alpha**2 + x**2)


class ExponentialWavefunction(nn.Module):

    @nn.compact
    def __call__(self, x):
        alpha = self.param("alpha", nn.initializers.constant(1.0), ())
        return jnp.exp(-alpha * jnp.sum(x**2, axis=-1))
