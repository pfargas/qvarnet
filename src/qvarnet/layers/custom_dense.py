from flax import linen as nn
import jax.numpy as jnp


class CustomDense(nn.Module):
    features: int
    kernel_init: callable = nn.initializers.lecun_normal()
    bias_init: callable = nn.initializers.zeros
    beta: float = 1.0  # scale factor for kernel

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )
        y = jnp.dot(inputs, self.beta * kernel)
        bias = self.param("bias", self.bias_init, (self.features,))
        return y + bias
