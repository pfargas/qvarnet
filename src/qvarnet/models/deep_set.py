from flax import linen as nn
from jax import numpy as jnp
from pyparsing import Callable

from .layers.custom_dense import CustomDense
from .mlp import MLP
from .base import BaseModel

from .registry import register_model


from flax import linen as nn
from jax import numpy as jnp
from pyparsing import Callable

from .layers.custom_dense import CustomDense
from .mlp import MLP
from .base import BaseModel

from .registry import register_model

import jax


@register_model("exponential-deep-set")
class ExponentialDeepSet(BaseModel):
    phi_hidden_architecture: list
    F_hidden_architecture: list
    n_particles: int
    phi_hidden_activation: Callable = nn.tanh
    F_hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    n_dim: int = 1
    hidden_internal_dimension: int = 20

    def setup(self):
        # 1. Cast everything to lists to ensure '+' works
        phi_hidden = list(self.phi_hidden_architecture)
        f_hidden = list(self.F_hidden_architecture)

        # 2. Handle the internal dimension
        # If it's an int, wrap it: [20]. If it's a list/tuple, cast it: [20]
        if isinstance(self.hidden_internal_dimension, (list, tuple)):
            internal_dim_list = list(self.hidden_internal_dimension)
        else:
            internal_dim_list = [self.hidden_internal_dimension]

        # 3. Construct architectures safely
        phi_architecture = [self.n_dim] + phi_hidden + internal_dim_list
        F_architecture = internal_dim_list + f_hidden + [1]
        self.phi = MLP(
            architecture=phi_architecture,
            hidden_activation=self.phi_hidden_activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        self.F = MLP(
            architecture=F_architecture,
            hidden_activation=self.F_hidden_activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def __call__(self, x):
        # x input shape is either:
        #   (Batch, N*Dim) -> e.g., (1000, 3) # in the energy evaluation
        #   (N*Dim,)       -> e.g., (3,) # in the sampler

        # 1. ROBUST RESHAPE
        # We take all leading dimensions (*x.shape[:-1]) as the batch/grouping dims
        # and explicitly split ONLY the last dimension into (N, Dim).
        # This works for shape (1000, 3) -> (1000, N, Dim)
        # AND for shape (3,) -> (N, Dim)

        n_dim = getattr(self, "n_dim", 1)  # Default to 1D if not specified
        h = x.reshape(*x.shape[:-1], self.n_particles, n_dim)
        # *shape[:-1] captures all leading dimensions,
        # then we split the last dim into (N, Dim) where N is n_particles and Dim is n_dim.

        # 2. RUN NETWORK
        # CustomDense automatically broadcasts over the extra dimensions.
        h = self.phi(h)  # Shape: (Batch, N, PhiDim)
        h = jnp.sum(
            h, axis=-2
        )  # Sum over the N dimension, resulting in shape (Batch, PhiDim)
        output = self.F(h)  # Shape: (Batch, F_out_dim)
        # in theory, the output dimension of F should be 1 for energy evaluation
        assert (
            output.shape[-1] == 1
        ), "Output dimension of F should be 1 for energy evaluation."
        # Exponential output for positive values, shape (Batch, F_out_dim)
        return jnp.exp(output)

    @classmethod
    def from_config(cls, model_args: dict):
        return cls(
            phi_hidden_architecture=model_args["phi_hidden_architecture"],
            F_hidden_architecture=model_args["F_hidden_architecture"],
            n_dim=model_args.get("n_dim", 1),
            n_particles=model_args.get("n_particles", 10),
        )

    @classmethod
    def get_input_shape(cls, model_args: dict, batch_size: int) -> tuple:
        return (
            batch_size,
            model_args.get("n_dim", 1) * model_args.get("n_particles", 10),
        )
