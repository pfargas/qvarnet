"""Probability function builders for different model types."""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Callable


def build_prob_fn(model_apply: Callable, is_log_model: bool) -> Callable:
    """
    Factory function to create probability density function matching model type.

    The probability function computes :math:`|\\psi(x; \\theta)|^2` which is
    proportional to the probability of configuration x in Variational Monte Carlo.

    Args:
        model_apply: Flax model apply function with signature ``(params, x) -> psi``.
        is_log_model: If True, model outputs ``log(|psi|)``. If False, outputs ``psi``.

    Returns:
        Callable: Probability function ``(x, params) -> R`` with shape ``(batch,)``.

    Examples:
        >>> # For direct model outputting psi
        >>> prob_fn_direct = build_prob_fn(model.apply, is_log_model=False)
        >>> prob = prob_fn_direct(x, params)  # shape (batch,)
        >>>
        >>> # For log model outputting log(|psi|)
        >>> prob_fn_log = build_prob_fn(model.apply, is_log_model=True)
        >>> log_prob = prob_fn_log(x, params)  # shape (batch,)
    """
    if not is_log_model:
        return _build_prob_fn_direct(model_apply)
    else:
        return _build_prob_fn_log(model_apply)


def _build_prob_fn_direct(model_apply: Callable) -> Callable:
    """
    Build probability function for models outputting psi directly.

    Returns :math:`|\\psi(x)|^2` by squaring the model output.
    """

    @partial(jax.jit, static_argnames=[])
    def prob_fn_direct(x, params):
        """
        Compute probability :math:`|\\psi(x)|^2` for direct model.

        Args:
            x: Configuration(s), shape ``(batch, DoF)`` or ``(DoF,)``.
            params: Model parameters (PyTree).

        Returns:
            Probability density, shape ``(batch,)``.
        """
        forward = model_apply(params, x).flatten()
        psi_squared = jnp.square(forward)
        return jnp.squeeze(psi_squared)

    return prob_fn_direct


def _build_prob_fn_log(model_apply: Callable) -> Callable:
    """
    Build probability function for models outputting ``log(|psi|)``.

    Returns :math:`|\\psi(x)|^2 = e^{2\\log|\\psi|}` using the log output.
    This is numerically more stable than squaring the exponentiated output.
    """

    @partial(jax.jit, static_argnames=[])
    def prob_fn_log(x, params):
        """
        Compute log-probability :math:`\\log|\\psi(x)|^2` for log-space model.

        For a model outputting :math:`\\log|\\psi|`:

        .. math::

            \\log|\\psi|^2 = 2 \\log|\\psi|

        Args:
            x: Configuration(s), shape ``(batch, DoF)`` or ``(DoF,)``.
            params: Model parameters (PyTree).

        Returns:
            Log probability density :math:`2\\log|\\psi|`, shape ``(batch,)``.
            Returned in log-space for use with log-space MCMC kernels.
        """
        forward = model_apply(params, x).flatten()
        log_psi_squared = 2 * forward  # log(psi^2) = 2*log(psi)
        return jnp.squeeze(log_psi_squared)

    return prob_fn_log
