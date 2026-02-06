import jax
from .layers import CustomDense
from flax import linen as nn
from typing import Callable
from .base import BaseModel

from .registry import register_model


@register_model("fermionic-mlp")
class FermionicMLP(BaseModel):
    architecture: list  # List of hidden layer sizes. ONLY HIDDEN LAYERS
    n_fermions: int = 2
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x):

        def orbital(position):
            y = position
            for i in range(len(self.architecture)):
                y = CustomDense(
                    features=self.architecture[i],
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                )(y)
                y = self.hidden_activation(y)
            y = CustomDense(
                features=self.n_fermions,  # Output two values for the orbital
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(y)
            return y

        # Apply orbital function to each fermion's position
        orbital_1 = orbital(x[..., :1])  # First position for orbital 1
        orbital_2 = orbital(x[..., 1:])  # Second position for orbital 2
        # Combine orbitals to get the final output

        # 2. Slice outputs safely using '...'
        # orbital_1 is [phi_A, phi_B]
        # use [..., 0] to get phi_A regardless of batch size
        phi1_A = orbital_1[..., 0]
        phi1_B = orbital_1[..., 1]

        phi2_A = orbital_2[..., 0]
        phi2_B = orbital_2[..., 1]

        # 3. Determinant: A(1)B(2) - B(1)A(2)
        antisym = phi1_A * phi2_B - phi1_B * phi2_A
        return antisym

    def build_from_params(self, params):
        pass
