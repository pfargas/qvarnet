from collections.abc import Callable

from flax import linen as nn
from jax import numpy as jnp

from .base import BaseModel
from .mlp import MLP
from .registry import register_model


@register_model("deep-set")
class DeepSet(BaseModel):
    """Permutation-invariant log-wavefunction: log|ψ| = F(mean_i φ(rᵢ)).

    Dimension contract:
        Input:  (..., n_particles, per_particle_dim)  — pre-reshaped by LogWavefunction
        After φ: (..., n_particles, hidden_internal_dim)
        After mean: (..., hidden_internal_dim)
        Output: (..., 1)  — log|ψ|, no envelope

    The reshape from flat (..., N*ppd) to (..., N, ppd) is handled by
    ``LogWavefunction`` when ``n_particles`` is set there. No geometry fields
    needed here — just specify the hidden layer widths.
    """

    phi_hidden: list
    F_hidden: list
    hidden_internal_dim: int = 20
    phi_activation: Callable = nn.tanh
    F_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    def setup(self):
        self.phi = MLP(
            hidden=list(self.phi_hidden),
            output_dim=self.hidden_internal_dim,
            hidden_activation=self.phi_activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            has_output_activation=True,
        )
        self.F = MLP(
            hidden=list(self.F_hidden),
            output_dim=1,
            hidden_activation=self.F_activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def __call__(self, x):
        # x: (..., n_particles, per_particle_dim)
        h = self.phi(x)  # (..., n_particles, hidden_internal_dim)
        h = jnp.mean(h, axis=-2)  # (..., hidden_internal_dim)
        return self.F(h)  # (..., 1)

    @classmethod
    def from_config(cls, model_args: dict):
        return cls(
            phi_hidden=model_args["phi_hidden"],
            F_hidden=model_args["F_hidden"],
        )

    @classmethod
    def get_input_shape(cls, model_args: dict, batch_size: int) -> tuple:
        raise NotImplementedError("DeepSet input shape depends on LogWavefunction geometry.")


# Backward-compat aliases — old code that imports these names still works.
DeepSetNoEnvelope = DeepSet
