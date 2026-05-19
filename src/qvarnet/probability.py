"""Probability function for log-space VMC models.

The model always outputs log|psi(x)|. The probability function returns
2*log|psi| = log(|psi|^2), used directly by the log-space MH kernel.
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Callable


def build_prob_fn(model_apply: Callable) -> Callable:
    """Build the log-probability function for MCMC sampling.

    Args:
        model_apply: Flax model apply function ``(params, x) -> log|psi|``.

    Returns:
        Callable ``(x, params) -> 2*log|psi(x)|``, shape ``(batch,)``.
    """

    @partial(jax.jit, static_argnames=[])
    def prob_fn(x, params):
        forward = model_apply(params, x).flatten()
        return jnp.squeeze(2 * forward)  # log(|psi|^2) = 2*log|psi|

    return prob_fn
