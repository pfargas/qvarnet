from functools import partial
import jax
from jax import numpy as jnp


@partial(jax.jit, static_argnames=("prob_fn", "uniform"))
def mh_kernel(
    step_rand,
    prob_fn,
    prob_params,
    position,
    prob,
    step_size,
    PBC,
    uniform=False,
):
    """Single Metropolis-Hastings step in probability space.

    Proposes a new configuration and accepts or rejects it with probability
    :math:`A = \\min(1, P(x') / P(x))`.

    Args:
        step_rand: Pre-drawn random numbers for this step, shape ``(DoF + 1,)``.
            ``step_rand[:-1]`` is the proposal noise (Gaussian or uniform).
            ``step_rand[-1]`` is the uniform accept/reject draw.
        prob_fn: Callable ``(x, params) -> P(x)``, unnormalised probability density.
        prob_params: Parameters passed to ``prob_fn``.
        position: Current configuration, shape ``(DoF,)``.
        prob: Current probability :math:`P(x)`.
        step_size: Proposal standard deviation.
        PBC: Periodic boundary size (reserved, currently unused).
        uniform: If ``True``, use a uniform proposal instead of Gaussian.

    Returns:
        new_position: Accepted or current configuration, shape ``(DoF,)``.
        new_prob: Probability at ``new_position``.
        accept: Boolean indicating whether the proposal was accepted.
    """
    proposal_noise = step_rand[:-1]
    accept_draw = step_rand[-1]

    if uniform:
        proposal = position + step_size * (2 * proposal_noise - 1)
    else:
        proposal = position + step_size * proposal_noise  # standard normal
    # proposal = ((proposal + 0.5 * PBC) % PBC) - 0.5 * PBC # apply PBC in the samples
    proposal_prob = prob_fn(proposal, prob_params)
    accept_prob = jnp.minimum(1.0, proposal_prob / prob)
    accept = accept_draw < accept_prob
    new_position = jnp.where(accept, proposal, position)
    new_prob = jnp.where(accept, proposal_prob, prob)
    return new_position, new_prob, accept


@partial(jax.jit, static_argnames=("prob_fn", "uniform"))
def mh_kernel_log(
    step_rand,
    prob_fn,
    prob_params,
    position,
    prob,
    step_size,
    PBC,
    uniform=False,
):
    """Single Metropolis-Hastings step in log-probability space.

    Numerically more stable than :func:`mh_kernel` when the model outputs
    :math:`\\log|\\psi|`. Accepts with probability
    :math:`A = \\min(1, e^{\\log P(x') - \\log P(x)})`.

    Args:
        step_rand: Pre-drawn random numbers for this step, shape ``(DoF + 1,)``.
            ``step_rand[:-1]`` is the proposal noise (Gaussian or uniform).
            ``step_rand[-1]`` is the uniform accept/reject draw.
        prob_fn: Callable ``(x, params) -> log P(x)``, log-unnormalised probability.
        prob_params: Parameters passed to ``prob_fn``.
        position: Current configuration, shape ``(DoF,)``.
        prob: Current log-probability :math:`\\log P(x)`.
        step_size: Proposal standard deviation.
        PBC: Periodic boundary size (reserved, currently unused).
        uniform: If ``True``, use a uniform proposal instead of Gaussian.

    Returns:
        new_position: Accepted or current configuration, shape ``(DoF,)``.
        new_log_prob: Log-probability at ``new_position``.
        accept: Boolean indicating whether the proposal was accepted.
    """
    proposal_noise = step_rand[:-1]
    accept_draw = step_rand[-1]

    if uniform:
        proposal = position + step_size * (2 * proposal_noise - 1)
    else:
        proposal = position + step_size * proposal_noise  # standard normal
    proposal_log_prob = prob_fn(proposal, prob_params)
    accept_log_prob = jnp.minimum(0.0, proposal_log_prob - prob)
    accept = jnp.log(accept_draw) < accept_log_prob
    new_position = jnp.where(accept, proposal, position)
    new_log_prob = jnp.where(accept, proposal_log_prob, prob)
    return new_position, new_log_prob, accept


@partial(jax.jit, static_argnames=("prob_fn", "is_log_prob", "uniform"))
def mh_chain(
    random_values,
    PBC,
    prob_fn,
    prob_params,
    init_position,
    step_size,
    is_log_prob=False,
    uniform=False,
):
    """Run a single Metropolis-Hastings chain over ``n_steps`` steps.

    Uses :func:`jax.lax.scan` for efficient JIT-compilation. Random numbers
    must be pre-generated externally; dispatch to :func:`mh_kernel` or
    :func:`mh_kernel_log` depending on ``is_log_prob``.

    Args:
        random_values: Pre-generated random numbers, shape ``(n_steps, DoF + 1)``.
            Each row is one ``step_rand`` buffer: first ``DoF`` values are proposal
            noise, last value is the accept/reject draw.
        PBC: Periodic boundary size passed through to the kernel.
        prob_fn: Probability (or log-probability) function ``(x, params) -> P(x)``.
        prob_params: Parameters for ``prob_fn``.
        init_position: Initial configuration, shape ``(DoF,)``.
        step_size: Proposal step size.
        is_log_prob: If ``True``, use :func:`mh_kernel_log`; otherwise :func:`mh_kernel`.
        uniform: If ``True``, use uniform proposals.

    Returns:
        positions: All sampled positions, shape ``(n_steps, DoF)``.
        acceptance_rate: Fraction of accepted proposals over all steps.
    """
    init_prob = prob_fn(init_position, prob_params)
    carry0 = (init_position, init_prob, step_size, 0)

    if is_log_prob:
        mh_kernel_fn = mh_kernel_log
    else:
        mh_kernel_fn = mh_kernel

    def body_fn(carry, step_rand):
        position, prob, step_size, count = carry
        new_position, new_prob, accepted = mh_kernel_fn(
            step_rand=step_rand,
            prob_fn=prob_fn,
            prob_params=prob_params,
            position=position,
            prob=prob,
            step_size=step_size,
            PBC=PBC,
            uniform=uniform,
        )
        new_count = count + accepted
        return (new_position, new_prob, step_size, new_count), (new_position, accepted)

    (_, _, _, counts), (positions, accepted) = jax.lax.scan(
        body_fn, carry0, random_values
    )
    acceptance_rate = counts / random_values.shape[0]
    return positions, acceptance_rate
