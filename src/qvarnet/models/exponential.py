"""Log-space MLP + analytic-envelope ansätze (the paper's ``mlp-*-decay`` family).

All models here output **log|ψ|** = MLP(x) − α·(envelope), with a trainable envelope
strength α. The psi-space originals from before the log-only engine migration were
removed 2026-07-02 (git history has them).
"""

from collections.abc import Callable

from flax import linen as nn
from jax import numpy as jnp

from .base import BaseModel
from .mlp import MLP
from .registry import register_model


@register_model("log-analytic")
class LogAnalyticWavefunction(BaseModel):

    @nn.compact
    def __call__(self, x):
        alpha = self.param("alpha", nn.initializers.constant(1.0), ())
        return -alpha * jnp.sum(x**2, axis=-1)

    @classmethod
    def from_config(cls, model_args: dict):
        return cls()
    
    @classmethod
    def get_input_shape(cls, model_args: dict, batch_size: int) -> tuple:
        return (batch_size, model_args["input_dim"])


@register_model("mlp-fourth-decay")
class LogExponentialMLPwithPenalty(BaseModel):
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
        log_wf = mlp_output - envelope_param * jnp.sum(x**4, axis=-1, keepdims=True)
        return log_wf

    def build_from_params(self, params):
        pass

    @classmethod
    def from_config(cls, model_args: dict):
        return cls(architecture=model_args["architecture"])

    @classmethod
    def get_input_shape(cls, model_args: dict, batch_size: int) -> tuple:
        return (batch_size, model_args["architecture"][0])

@register_model("mlp-gaussian-decay")
class LogExponentialMLPwithGaussianPenalty(BaseModel):
    architecture: list
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.normal(1.0)
    # kernel_init: Callable = nn.initializers.lecun_normal()
    # bias_init: Callable = nn.initializers.zeros_init()
    bias_init: Callable = nn.initializers.normal(stddev=1.0)

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
        log_wf = mlp_output - envelope_param * jnp.sum(x**2, axis=-1, keepdims=True)
        return log_wf

    def build_from_params(self, params):
        pass

    @classmethod
    def from_config(cls, model_args: dict):
        return cls(architecture=model_args["architecture"])

    @classmethod
    def get_input_shape(cls, model_args: dict, batch_size: int) -> tuple:
        return (batch_size, model_args["architecture"][0])
    
@register_model("j-mlp-gaussian-decay")
class JastrowLogExponentialMLPwithGaussianPenalty(BaseModel):
    architecture: list
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    lambda_init: float = 0.5

    @nn.compact
    def __call__(self, x):
        # x: (..., n_particles)  — works for (batch, N) and bare (N,)
        lam   = self.param("lam",   nn.initializers.constant(self.lambda_init), ())
        # omega = self.param("omega", nn.initializers.constant(1.0), ())

        n = x.shape[-1]  # n_particles lives in the LAST dim, not dim 0

        # Pairwise |xᵢ - xⱼ|: (..., N, 1) vs (..., 1, N) → (..., N, N)
        diffs = jnp.abs(x[..., :, None] - x[..., None, :]) + 1e-6

        # Upper-triangle mask: selects i<j only (excludes diagonal and lower half)
        mask = jnp.triu(jnp.ones((n, n)), k=1)  # (N, N)

        # Jastrow in log-space: λ · Σᵢ<ⱼ log|xᵢ-xⱼ|  →  (...)
        log_jastrow = lam * jnp.sum(mask * jnp.log(diffs), axis=(-1, -2))[..., None] # the dimensions of sum were (N,) and we need (N,1) :)
        mlp = MLP(
            architecture=self.architecture,
            hidden_activation=self.hidden_activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        envelope_param = self.param("envelope_param", nn.initializers.constant(1.0), ())
        mlp_output = mlp(x)
        log_wf = mlp_output - envelope_param * jnp.sum(x**2, axis=-1, keepdims=True) + log_jastrow
        return log_wf

    def build_from_params(self, params):
        pass

    @classmethod
    def from_config(cls, model_args: dict):
        return cls(architecture=model_args["architecture"])

    @classmethod
    def get_input_shape(cls, model_args: dict, batch_size: int) -> tuple:
        return (batch_size, model_args["architecture"][0])