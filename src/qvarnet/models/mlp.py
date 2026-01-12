from qvarnet.layers import CustomDense
from flax import linen as nn


class MLP(nn.Module):
    architecture: list
    hidden_activation: callable = nn.tanh
    kernel_init: callable = nn.initializers.constant(0.5)
    bias_init: callable = nn.initializers.zeros
    beta: float = 1.0  # scale factor for kernel

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.architecture) - 1):
            x = CustomDense(
                features=self.architecture[i + 1],
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                beta=self.beta,
            )(x)
            if i < len(self.architecture) - 2:
                x = self.hidden_activation(x)
        return x
