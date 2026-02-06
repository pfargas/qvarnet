import jax.numpy as jnp
from .layers import CustomDense
from flax import linen as nn
from typing import Callable
from .base import BaseModel

from .registry import register_model


@register_model("fermionic-mlp")
class FermionicMLP(BaseModel):
    architecture: list
    n_fermions: int
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    n_dim: int = 1

    def setup(self):
        # 1. DEFINITION PHASE
        # We create the layers here. They are assigned to 'self',
        # so Flax knows these are the parameters of the model.

        self.hidden_layers = [
            CustomDense(
                features=feat,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"hidden_{i}",  # Optional, but good for debugging
            )
            for i, feat in enumerate(self.architecture)
        ]

        self.output_layer = CustomDense(
            features=self.n_fermions,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="orbital_output",
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
        h = x.reshape(*x.shape[:-1], self.n_fermions, n_dim)
        # *shape[:-1] captures all leading dimensions,
        # then we split the last dim into (N, Dim) where N is n_fermions and Dim is n_dim.

        # 2. RUN NETWORK
        # CustomDense automatically broadcasts over the extra dimensions.
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.hidden_activation(h)

        # Output shape: (..., N, N)
        # (Batch, Particle i, Orbital j)
        orbitals = self.output_layer(h)

        # 3. DETERMINANT
        # jnp.linalg.det handles the last two dimensions (NxN) correctly
        # regardless of how many batch dimensions precede them.
        psi = jnp.linalg.det(orbitals)

        return psi


@register_model("fermionic-mlp-2")
class FermionicMLP2ferms(BaseModel):
    architecture: list
    n_fermions: int = 2
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    def setup(self):
        # 1. DEFINITION PHASE
        # We create the layers here. They are assigned to 'self',
        # so Flax knows these are the parameters of the model.

        self.hidden_layers = [
            CustomDense(
                features=feat,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"hidden_{i}",  # Optional, but good for debugging
            )
            for i, feat in enumerate(self.architecture)
        ]

        self.output_layer = CustomDense(
            features=self.n_fermions,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="orbital_output",
        )

    def __call__(self, x):
        # 2. COMPUTATION PHASE
        # We just use the layers we defined above.

        def orbital_net(pos):
            y = pos
            for layer in self.hidden_layers:
                y = layer(y)
                y = self.hidden_activation(y)
            y = self.output_layer(y)
            return y

        # Apply the exact same network functions to both inputs
        # (This is mathematically guaranteed to use shared weights)
        orb_1 = orbital_net(x[..., :1])
        orb_2 = orbital_net(x[..., 1:])

        # Determinant Logic
        phi1_A = orb_1[..., 0]
        phi1_B = orb_1[..., 1]

        phi2_A = orb_2[..., 0]
        phi2_B = orb_2[..., 1]

        return phi1_A * phi2_B - phi1_B * phi2_A
