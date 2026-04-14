"""Probability function builders for different model types."""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Callable


def build_prob_fn(model_apply: Callable, is_log_model: bool) -> Callable:
    """
    Factory function to create probability density function matching model type.

    The probability function computes |ψ(x; θ)|² which is proportional to
    the probability of configuration x in Variational Monte Carlo.

    Args:
        model_apply: Flax model apply function with signature (params, x) -> ψ
        is_log_model: If True, model outputs log(|ψ|). If False, outputs ψ.

    Returns:
        Callable: Probability function (x, params) -> ℝ with shape (batch,)

    Examples:
        >>> # For direct model outputting ψ
        >>> prob_fn_direct = build_prob_fn(model.apply, is_log_model=False)
        >>> prob = prob_fn_direct(x, params)  # shape (batch,)
        >>>
        >>> # For log model outputting log(|ψ|)
        >>> prob_fn_log = build_prob_fn(model.apply, is_log_model=True)
        >>> log_prob = prob_fn_log(x, params)  # shape (batch,)
    """
    if not is_log_model:
        return _build_prob_fn_direct(model_apply)
    else:
        return _build_prob_fn_log(model_apply)


def _build_prob_fn_direct(model_apply: Callable) -> Callable:
    """
    Build probability function for models outputting ψ directly.

    Returns |ψ(x)|² by squaring the model output.
    """

    @partial(jax.jit, static_argnames=[])
    def prob_fn_direct(x, params):
        """
        Compute probability |ψ(x)|² for direct model.

        Args:
            x: Configuration(s), shape (batch, DoF) or (DoF,)
            params: Model parameters (PyTree)

        Returns:
            Probability density, shape (batch,)
        """
        forward = model_apply(params, x).flatten()
        psi_squared = jnp.square(forward)
        return jnp.squeeze(psi_squared)

    return prob_fn_direct


def _build_prob_fn_log(model_apply: Callable) -> Callable:
    """
    Build probability function for models outputting log(|ψ|).

    Returns |ψ(x)|² = e^(2*log|ψ|) using the log output.
    This is numerically more stable than taking exp(log(ψ))².
    """

    @partial(jax.jit, static_argnames=[])
    def prob_fn_log(x, params):
        """
        Compute probability |ψ(x)|² for log-space model.

        For a model outputting log(|ψ|), we have:
        |ψ|² = e^(2 * log(|ψ|))

        Args:
            x: Configuration(s), shape (batch, DoF) or (DoF,)
            params: Model parameters (PyTree)

        Returns:
            Log probability density log(|ψ|²) = 2*log|ψ|, shape (batch,)
            Note: This is returned in log-space for use in log-space MCMC kernels.
        """
        forward = model_apply(params, x).flatten()
        log_psi_squared = 2 * forward  # log(ψ²) = 2*log(ψ)
        return jnp.squeeze(log_psi_squared)

    return prob_fn_log
