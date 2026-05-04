from flax import linen as nn
import jax.numpy as jnp
from typing import Callable


class CustomDense(nn.Module):
    """Linear layer that broadcasts over arbitrary leading dimensions.

    Input:  (..., in_features)   — any number of leading batch/particle dims
    Kernel: (in_features, features)
    Output: (..., features)      — same leading dims, last dim replaced
    """

    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    beta: float = 1.0  # scale factor for kernel

    @nn.compact
    def __call__(self, inputs):
        # inputs: (..., in_features)
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )
        y = jnp.dot(inputs, self.beta * kernel)  # (..., features)
        bias = self.param("bias", self.bias_init, (self.features,))
        return y + bias  # (..., features)
