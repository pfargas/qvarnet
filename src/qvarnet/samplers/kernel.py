from functools import partial

import jax
from jax import numpy as jnp


@partial(jax.jit, static_argnames=("prob_fn", "uniform"))
def mh_kernel_log(
    step_rand,
    prob_fn,
    prob_params,
    position,
    prob,
    step_size,
    uniform=False,
    box_L=0.0,
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
        uniform: If ``True``, use a uniform proposal instead of Gaussian.
        box_L: Periodic box side length. ``> 0`` wraps each proposed coordinate into
            ``[0, L)`` (PBC sampler). The Gaussian/uniform proposal is symmetric on the
            torus, so detailed balance and the acceptance ratio are unchanged. ``0``
            (default) disables wrapping. Passed as a traced value, not a static arg.

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
    # PBC sampler: fold proposal into [0, L) when box_L > 0 (no-op when box_L == 0).
    wrapped = proposal - box_L * jnp.floor(proposal / jnp.where(box_L > 0, box_L, 1.0))
    proposal = jnp.where(box_L > 0, wrapped, proposal)
    proposal_log_prob = prob_fn(proposal, prob_params)
    accept_log_prob = jnp.minimum(0.0, proposal_log_prob - prob)
    accept = jnp.log(accept_draw) < accept_log_prob
    new_position = jnp.where(accept, proposal, position)
    new_log_prob = jnp.where(accept, proposal_log_prob, prob)
    return new_position, new_log_prob, accept


@partial(jax.jit, static_argnames=("prob_fn", "uniform"))
def mh_chain(
    random_values,
    prob_fn,
    prob_params,
    init_position,
    step_size,
    uniform=False,
    box_L=0.0,
):
    """Run a single Metropolis-Hastings chain over ``n_steps`` steps.

    Uses :func:`jax.lax.scan` for efficient JIT-compilation. Random numbers
    must be pre-generated externally. Always uses log-space MH kernel.

    Args:
        random_values: Pre-generated random numbers, shape ``(n_steps, dof + 1)``.
            Each row: first ``dof`` values are proposal noise, last is accept/reject draw.
        prob_fn: Log-probability function ``(x, params) -> log P(x)``.
        prob_params: Parameters for ``prob_fn``.
        init_position: Initial configuration, shape ``(dof,)``.
        step_size: Proposal step size.
        uniform: If ``True``, use uniform proposals.
        box_L: Periodic box side length; ``> 0`` wraps proposals into ``[0, L)``.

    Returns:
        positions: All sampled positions, shape ``(n_steps, dof)``.
        acceptance_rate: Fraction of accepted proposals over all steps.
    """
    init_prob = prob_fn(init_position, prob_params)
    carry0 = (init_position, init_prob, step_size, 0)

    def body_fn(carry, step_rand):
        position, prob, step_size, count = carry
        new_position, new_prob, accepted = mh_kernel_log(
            step_rand=step_rand,
            prob_fn=prob_fn,
            prob_params=prob_params,
            position=position,
            prob=prob,
            step_size=step_size,
            uniform=uniform,
            box_L=box_L,
        )
        new_count = count + accepted
        return (new_position, new_prob, step_size, new_count), (new_position, accepted)

    (_, _, _, counts), (positions, accepted) = jax.lax.scan(body_fn, carry0, random_values)
    acceptance_rate = counts / random_values.shape[0]
    return positions, acceptance_rate
