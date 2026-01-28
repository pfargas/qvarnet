from functools import partial
import jax
from jax import random
from jax import numpy as jnp
from matplotlib.pyplot import hist


@partial(jax.jit, static_argnames=("prob_fn"))
def mh_kernel(
    uniform_random_numbers, prob_fn, prob_params, position, prob, step_size, PBC
):
    proposal = position + step_size * (2 * uniform_random_numbers[:-1] - 1)
    proposal = ((proposal + 0.5 * PBC) % PBC) - 0.5 * PBC
    proposal_prob = prob_fn(proposal, prob_params)
    accept_prob = jnp.minimum(1.0, proposal_prob / prob)
    accept = uniform_random_numbers[-1] < accept_prob
    new_position = jnp.where(accept, proposal, position)
    new_prob = jnp.where(accept, proposal_prob, prob)
    # acceptance_rate = jnp.mean(accept.astype(jnp.float32))
    # acceptance_rate = accept.astype(jnp.float32)
    acceptance_rate = 0.0
    return new_position, new_prob, acceptance_rate


@partial(jax.jit, static_argnames=("prob_fn"))
def mh_chain(random_values, PBC, prob_fn, prob_params, init_position, step_size):
    """
    Single MH chain using pre-generated step keys.
    random_values: shape (n_steps, DoF + 1)
    init_position: shape (DoF,)
    """

    init_prob = prob_fn(init_position, prob_params)
    carry0 = (init_position, init_prob, step_size)

    def body_fn(carry, random_values):
        position, prob, step_size = carry
        new_position, new_prob, _ = mh_kernel(
            uniform_random_numbers=random_values,
            prob_fn=prob_fn,
            prob_params=prob_params,
            position=position,
            prob=prob,
            step_size=step_size,
            PBC=PBC,
        )
        return (new_position, new_prob, step_size), new_position

    (_, _, _), positions = jax.lax.scan(body_fn, carry0, random_values)
    return positions
