from flax import linen as nn
from jax import numpy as jnp
from pyparsing import Callable

from qvarnet.layers.custom_dense import CustomDense
from .mlp import MLP
from .base import BaseModel


class ExponentialWavefunction(nn.Module):

    @nn.compact
    def __call__(self, x):
        alpha = self.param("alpha", nn.initializers.constant(1.0), ())
        return jnp.exp(-alpha * jnp.sum(x**2, axis=-1))


class ExponentialMLPwithPenalty(BaseModel):
    architecture: list
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        mlp = MLP(
            architecture=self.architecture,
            hidden_activation=self.hidden_activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        envelope_param = self.param("envelope_param", nn.initializers.constant(1.0), ())
        mlp_output = mlp(x)
        exp_wf = jnp.exp(
            mlp_output - envelope_param * jnp.sum(x**4, axis=-1, keepdims=True)
        )
        return exp_wf

    def build_from_params(self, params):
        pass
