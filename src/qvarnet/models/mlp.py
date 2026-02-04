from .layers import CustomDense
from flax import linen as nn
from typing import Callable
from .base import BaseModel

from .registry import register_model


@register_model("mlp")
class MLP(BaseModel):
    architecture: list
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
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

    def build_from_params(self, params):
        architecture = []
        layers = params["params"]
        for layer_name in layers:
            layer_params = layers[layer_name]
            if "kernel" in layer_params:
                architecture.append(layer_params["kernel"].shape[0])
        # Append output layer size
        last_layer = list(layers.keys())[-1]
        output_size = layers[last_layer]["kernel"].shape[1]
        architecture.append(output_size)
        return MLP(architecture=architecture)
