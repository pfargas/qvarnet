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
        x = jnp.exp(-0.5 * (x**2))
        return x