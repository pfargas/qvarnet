from functools import partial
import jax
from jax import random
from jax import numpy as jnp


@partial(jax.jit, static_argnums=(1,))
def mh_kernel(rng_key, prob_fn, prob_params, position, prob, PBC=None):
    key1, key2 = random.split(rng_key)
    # proposal = position + random.normal(key1, shape=position.shape) * 0.5
    proposal = position + random.uniform(
        key1, shape=position.shape, minval=-0.5, maxval=0.5
    )
    # ensure PBC
    proposal = jax.lax.cond(
        PBC is None,
        lambda p: p,
        lambda p: ((p + 0.5 * PBC) % PBC) - 0.5 * PBC,
        proposal,
    )
    proposal_prob = prob_fn(proposal, prob_params)
    accept_prob = jnp.minimum(1.0, proposal_prob / prob)
    accept = jax.random.uniform(key2) < accept_prob
    # accept = jax.random.uniform(key2) < (proposal_prob / prob)
    new_position = jnp.where(accept, proposal, position)
    new_prob = jnp.where(accept, proposal_prob, prob)
    return new_position, new_prob


@partial(jax.jit, static_argnums=(1, 2, 3))
def mh_chain(rng_key, n_steps, PBC, prob_fn, prob_params, init_position):
    def body_fn(i, val):
        key, position, prob = val
        key, subkey = random.split(key)
        new_position, new_prob = mh_kernel(
            subkey, prob_fn, prob_params, position, prob, PBC=PBC
        )
        return key, new_position, new_prob

    init_prob = prob_fn(init_position, prob_params)
    init_val = (rng_key, init_position, init_prob)
    _, positions, _ = jax.lax.fori_loop(0, n_steps, body_fn, init_val)
    return positions