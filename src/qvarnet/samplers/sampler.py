from functools import partial
import jax
from jax import random
from jax import numpy as jnp
from matplotlib.pyplot import hist


@partial(jax.jit, static_argnames=("prob_fn", "uniform"))
def mh_kernel(
    uniform_random_numbers,
    prob_fn,
    prob_params,
    position,
    prob,
    step_size,
    PBC,
    uniform=False,
):
    if uniform:
        proposal = position + step_size * (2 * uniform_random_numbers[:-1] - 1)
    else:
        proposal = position + step_size * random.normal(
            random.PRNGKey(0), shape=position.shape
        )
        jax.debug.print("Proposal normal")
    # proposal = ((proposal + 0.5 * PBC) % PBC) - 0.5 * PBC # apply PBC in the samples
    proposal_prob = prob_fn(proposal, prob_params)
    accept_prob = jnp.minimum(1.0, proposal_prob / (prob))  # + 1e-12))
    accept = uniform_random_numbers[-1] < accept_prob
    new_position = jnp.where(accept, proposal, position)
    new_prob = jnp.where(accept, proposal_prob, prob)
    return new_position, new_prob, accept


@partial(jax.jit, static_argnames=("prob_fn", "uniform"))
def mh_kernel_log(
    uniform_random_numbers,
    prob_fn,
    prob_params,
    position,
    prob,
    step_size,
    PBC,
    uniform=False,
):
    if uniform:
        proposal = position + step_size * (2 * uniform_random_numbers[:-1] - 1)
    else:
        proposal = position + step_size * uniform_random_numbers[:-1]  # standard normal
        jax.debug.print("Proposal normal")
    proposal_log_prob = prob_fn(proposal, prob_params)
    accept_log_prob = jnp.minimum(
        0.0, proposal_log_prob - prob
    )  # log(accept_prob) = min(0, log(proposal_prob) - log(current_prob))
    accept = jnp.log(uniform_random_numbers[-1]) < accept_log_prob
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
    """
    Single MH chain using pre-generated step keys.
    random_values: shape (n_steps, DoF + 1)
    init_position: shape (DoF,)
    """

    init_prob = prob_fn(init_position, prob_params)
    carry0 = (init_position, init_prob, step_size, 0)

    if is_log_prob:
        mh_kernel_fn = mh_kernel_log
    else:
        mh_kernel_fn = mh_kernel

    def body_fn(carry, random_values):
        position, prob, step_size, count = carry
        new_position, new_prob, accepted = mh_kernel_fn(
            uniform_random_numbers=random_values,
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
