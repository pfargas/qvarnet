from functools import partial
import jax
import jax.numpy as jnp
from jax import random


@partial(jax.jit, static_argnums=(1,))
def mh_kernel(rng_key, prob_fn, position, prob):
    key1, key2 = random.split(rng_key)
    proposal = position + random.normal(key1, shape=position.shape)
    proposal_prob = prob_fn(proposal)
    accept = jax.random.uniform(key2) < (proposal_prob / prob)
    new_position = jnp.where(accept, proposal, position)
    new_prob = jnp.where(accept, proposal_prob, prob)
    return new_position, new_prob


@partial(jax.jit, static_argnums=(1, 2))
def mh_chain(rng_key, n_steps, prob_fn, init_position):
    def body_fn(i, val):
        key, position, prob = val
        _, key = random.split(key)
        new_position, new_prob = mh_kernel(key, prob_fn, position, prob)
        return key, new_position, new_prob

    init_prob = prob_fn(init_position)
    init_val = (rng_key, init_position, init_prob)
    _, positions, _ = jax.lax.fori_loop(0, n_steps, body_fn, init_val)
    return positions


if __name__ == "__main__":
    rng_key = random.PRNGKey(0)
    n_chains = 100000
    n_steps = 100
    init_position = jnp.zeros(n_chains)  # (nchains,)
    rng_keys = jax.random.split(rng_key, n_chains)  # (nchains,)
    print(rng_keys.shape)
    run_mh_chain = jax.vmap(mh_chain, in_axes=(0, None, None, 0), out_axes=0)
    samples = run_mh_chain(
        rng_keys, n_steps, lambda x: jnp.exp(-0.5 * jnp.sum(x**2)), init_position
    )
    import matplotlib.pyplot as plt
    import numpy as np

    samples.block_until_ready()

    # Convert JAX arrays to numpy before plotting
    samples_np = np.array(samples.flatten())
    print(samples_np.shape)

    plt.hist(samples_np, bins=50, density=True, alpha=0.7, label="MCMC samples")

    x = jnp.linspace(-4, 4, 100)
    x_np = np.array(x)
    y_np = np.array(jnp.exp(-0.5 * x**2) / jnp.sqrt(2 * jnp.pi))

    plt.plot(x_np, y_np, "r-", linewidth=2, label="True distribution")
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("MCMC Sampling Results")
    plt.show()
