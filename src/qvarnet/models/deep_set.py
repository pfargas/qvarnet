from typing import Callable

import jax
from flax import linen as nn
from jax import numpy as jnp

from .base import BaseModel
from .layers.custom_dense import CustomDense
from .mlp import MLP
from .registry import register_model


@register_model("exponential-deep-set")
class ExponentialDeepSet(BaseModel):
    """DeepSet wavefunction ansatz: ψ = exp(F(Σ_i φ(r_i))).

    Dimension contract:
        get_input_shape → (batch, n_particles * n_dim)

        __call__ input:  (batch, n_particles * n_dim)  [energy eval]
                         (n_particles * n_dim,)          [sampler, single config]

        Internal reshape: (..., n_particles * n_dim) → (..., n_particles, n_dim)
        After phi:        (..., n_particles, hidden_internal_dimension)
        After sum:        (..., hidden_internal_dimension)          ← permutation invariant
        After F:          (..., 1)
        Output:           exp of that → (..., 1)   [caller must squeeze if needed]

    phi arch:  [n_dim] + phi_hidden + [hidden_internal_dimension]
    F arch:    [hidden_internal_dimension] + F_hidden + [1]
    """

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
        # x: (..., n_particles * n_dim)  — batch or single config
        n_dim = getattr(self, "n_dim", 1)
        h = x.reshape(*x.shape[:-1], self.n_particles, n_dim)
        # h: (..., n_particles, n_dim)

        h = self.phi(h)
        # h: (..., n_particles, hidden_internal_dimension)

        h = jnp.sum(h, axis=-2)
        # h: (..., hidden_internal_dimension)  — summed over particles → permutation invariant

        output = self.F(h)
        # output: (..., 1)
        assert (
            output.shape[-1] == 1
        ), "Output dimension of F should be 1 for energy evaluation."

        return jnp.exp(output)
        # return: (..., 1)

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
        # → (batch, n_particles * n_dim)
        return (
            batch_size,
            model_args.get("n_dim", 1) * model_args.get("n_particles", 10),
        )


@register_model("deep-set")
class DeepSet(BaseModel):
    """DeepSet log-wavefunction ansatz: log|ψ| = F(Σ_i φ(r_i)) - envelope·Σ x⁴.

    Dimension contract: identical to ExponentialDeepSet except output is in log-space.

        get_input_shape → (batch, n_particles * n_dim)

        __call__ input:  (..., n_particles * n_dim)
        Internal:        (..., n_particles, n_dim)  [after reshape]
        After phi:       (..., n_particles, hidden_internal_dimension)
        After sum:       (..., hidden_internal_dimension)
        After F:         (..., 1)
        Output:          (..., 1)  [log|ψ| minus envelope term]

    phi arch:  [n_dim] + phi_hidden + [hidden_internal_dimension]
    F arch:    [hidden_internal_dimension] + F_hidden + [1]
    """

    phi_hidden_architecture: list
    F_hidden_architecture: list
    n_particles: int
    phi_hidden_activation: Callable = nn.tanh
    F_hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    n_dim: int = 1
    hidden_internal_dimension: int = 20
    envelope_init: float = 0.1

    def setup(self):
        # 1. Cast everything to lists to ensure '+' works
        phi_hidden = list(self.phi_hidden_architecture)
        f_hidden = list(self.F_hidden_architecture)
        # define envelope_param as a trainable parameter

        self.envelope_param = self.param(
            "envelope_param", nn.initializers.constant(self.envelope_init), ()
        )

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
            has_output_activation=True,  # Apply activation to output of phi for better expressivity
        )

        self.F = MLP(
            architecture=F_architecture,
            hidden_activation=self.F_hidden_activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def __call__(self, x):
        # x: (..., n_particles * n_dim)
        n_dim = getattr(self, "n_dim", 1)
        h = x.reshape(*x.shape[:-1], self.n_particles, n_dim)
        # h: (..., n_particles, n_dim)

        h = self.phi(h)
        # h: (..., n_particles, hidden_internal_dimension)

        h = jnp.sum(h, axis=-2) / self.n_particles
        # h: (..., hidden_internal_dimension)  — permutation invariant

        output = self.F(h)
        # output: (..., 1)
        assert (
            output.shape[-1] == 1
        ), "Output dimension of F should be 1 for energy evaluation."

        return output - self.envelope_param**2 * jnp.sum(x**4, axis=-1, keepdims=True)

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
