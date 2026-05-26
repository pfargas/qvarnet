from collections.abc import Callable

from flax import linen as nn

from .base import BaseModel
from .layers import CustomDense
from .registry import register_model


@register_model("mlp")
class MLP(BaseModel):
    """Multi-layer perceptron using CustomDense layers.

    Two interfaces (exactly one must be provided):
      hidden=[h1, h2, ...], output_dim=1  — new, lazy; no input dim needed
      architecture=[in, h1, ..., out]     — legacy; first element is documentation only

    Input:  (..., any)   — arbitrary leading dims preserved; input width inferred lazily
    Output: (..., output_dim / architecture[-1])
    """

    architecture: list = None
    hidden: list = None
    output_dim: int = 1
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    beta: float = 1.0
    has_output_activation: bool = False

    @nn.compact
    def __call__(self, x):
        if self.hidden is not None:
            arch = list(self.hidden) + [self.output_dim]
        elif self.architecture is not None:
            arch = list(self.architecture)[1:]  # first elem is documentation only
        else:
            raise ValueError("MLP requires either 'hidden' or 'architecture'")
        for i, features in enumerate(arch):
            x = CustomDense(
                features=features,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                beta=self.beta,
            )(x)
            if i < len(arch) - 1:
                x = self.hidden_activation(x)
        if self.has_output_activation:
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

    @classmethod
    def from_config(cls, model_args: dict):
        return cls(architecture=model_args["architecture"])

    @classmethod
    def get_input_shape(cls, model_args: dict, batch_size: int) -> tuple:
        return (batch_size, model_args["architecture"][0])
