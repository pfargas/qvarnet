from functools import partial
import jax
from jax import random
from jax import numpy as jnp
from matplotlib.pyplot import hist


@partial(jax.jit, static_argnames=("prob_fn"))
def mh_kernel(key, prob_fn, prob_params, position, prob, step_size, PBC=10.0):
    uniform_random_numbers = random.uniform(key, shape=(position.shape[0] + 1,))
    proposal = position + step_size * (2 * uniform_random_numbers[:-1] - 1)
    proposal = ((proposal + 0.5 * PBC) % PBC) - 0.5 * PBC
    proposal_prob = prob_fn(proposal, prob_params)
    accept_prob = jnp.minimum(1.0, proposal_prob / prob)
    accept = uniform_random_numbers[-1] < accept_prob
    new_position = jnp.where(accept, proposal, position)
    new_prob = jnp.where(accept, proposal_prob, prob)
    return new_position, new_prob


@partial(jax.jit, static_argnames=("prob_fn", "n_steps"))
def mh_chain(key, PBC, prob_fn, prob_params, init_position, step_size, n_steps):
    """
    Single MH chain using pre-generated step keys.
    random_values: shape (n_steps, DoF + 1)
    init_position: shape (DoF,)
    """

    init_prob = prob_fn(init_position, prob_params)
    carry0 = (init_position, init_prob)
    keys = random.split(key, n_steps)

    def body_fn(carry, step_key):
        position, prob = carry
        new_position, new_prob = mh_kernel(
            key=step_key,
            prob_fn=prob_fn,
            prob_params=prob_params,
            position=position,
            prob=prob,
            step_size=step_size,
            PBC=PBC,
        )
        return (new_position, new_prob), new_position

    (_, _), positions = jax.lax.scan(body_fn, carry0, keys)
    return positions


if __name__ == "__main__":
    print("This is a module for Metropolis-Hastings sampling.")

    sampler = jax.vmap(
        mh_chain,
        in_axes=(
            0,  # key
            None,  # PBC
            None,  # prob_fn
            None,  # prob_params
            0,  # init_position
            None,  # step_size
            None,  # n_steps
        ),
        out_axes=0,
    )

    n_chains = 4
    DoF = 2
    n_steps = 10000
    key = random.PRNGKey(0)
    keys = random.split(key, n_chains)
    init_positions = jnp.zeros((n_chains, DoF))
    step_size = 1.0
    PBC = 10.0
    prob_fn = lambda x, params: jnp.exp(-jnp.sum(x**2))  # Example: Gaussian
    prob_params = None
    samples = sampler(
        keys,
        PBC,
        prob_fn,
        prob_params,
        init_positions,
        step_size,
        n_steps,
    )
    print("Samples shape:", samples.shape)  # (n_chains, n_steps, DoF)
    hist(samples.reshape(-1, DoF)[:, 0], bins=30)
    import matplotlib.pyplot as plt

    plt.show()
