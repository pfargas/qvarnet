from flax import linen as nn
import jax.numpy as jnp
from typing import Callable


class CustomDense(nn.Module):
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    beta: float = 1.0  # scale factor for kernel

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )
        y = jnp.dot(inputs, self.beta * kernel)
        bias = self.param("bias", self.bias_init, (self.features,))
        return y + bias
