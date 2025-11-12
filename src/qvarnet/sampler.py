from functools import partial
import jax
from jax import random
from jax import numpy as jnp
from matplotlib.pyplot import hist


@partial(jax.jit, static_argnums=(1,))
def mh_kernel(rng_key, prob_fn, prob_params, position, prob, PBC=None):
    key1, key2 = random.split(rng_key)
    proposal = position + random.uniform(
        key1, shape=position.shape, minval=-1, maxval=1
    )
    if PBC is None:
        raise ValueError("PBC must be provided for periodic boundary conditions")
    # ensure PBC
    # proposal = jax.lax.cond(
    #     PBC is None,
    #     lambda p: p,
    #     lambda p: ((p + 0.5 * PBC) % PBC) - 0.5 * PBC,
    #     proposal,
    # )
    proposal = ((proposal + 0.5 * PBC) % PBC) - 0.5 * PBC
    proposal_prob = prob_fn(proposal, prob_params)
    accept_prob = jnp.minimum(1.0, proposal_prob / prob)
    accept = jax.random.uniform(key2) < accept_prob
    new_position = jnp.where(accept, proposal, position)
    new_prob = jnp.where(accept, proposal_prob, prob)
    return new_position, new_prob


@partial(jax.jit, static_argnums=(1, 2, 3))
def mh_chain(rng_key, n_steps, PBC, prob_fn, prob_params, init_position):
    """MH single chain.

    If this chain is executed alone, then you obtain 1 single sample taken after n_steps.

    To obtain multiple samples, run multiple chains in parallel:

    sampler = jax.vmap(mh_chain, in_axes=(0, None, None, None, None, 0), out_axes=0)

    That means:
    - in_axes:
        - 0: different rng_key per chain
        - None: same n_steps for all chains
        - None: same PBC for all chains
        - None: same prob_fn for all chains
        - None: same prob_params per chain
        - 0: different init_position per chain

    """

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


if __name__ == "__main__":
    print("This is the qvarnet.sampler module.")
    print("TESTING PLAYGROUND FOR THE SAMPLER MODULE")

    key = random.PRNGKey(0)

    def test_2d_prob_fn(x, params):
        return jnp.exp(-0.5 * jnp.sum((x) ** 2, axis=-1))

    sampler = jax.vmap(mh_chain, in_axes=(0, None, None, None, 0, 0), out_axes=0)
    n_chains = 500_000
    DoF = 2
    n_steps = 500
    PBC = 5
    print("CONFIG USED:")
    print(f"n_chains: {n_chains}, DoF: {DoF}, n_steps: {n_steps}")
    rng_keys = random.split(random.PRNGKey(872643), n_chains)
    init_positions = jax.random.normal(random.PRNGKey(0), (n_chains, DoF)) * 2.0
    samples = sampler(rng_keys, n_steps, PBC, test_2d_prob_fn, None, init_positions)
    print("Samples shape:", samples.shape)  # (n_chains, DoF)

    import matplotlib.pyplot as plt

    histo_data = jnp.histogram2d(
        samples.reshape(-1, DoF)[:, 0],
        samples.reshape(-1, DoF)[:, 1],
        # bins=jnp.ceil(jnp.sqrt(n_chains)).astype(int),
        bins=100,
        range=[[-PBC / 2, PBC / 2], [-PBC / 2, PBC / 2]],
        density=False,
    )
    # plt.figure(figsize=(8, 6))

    # plt.pcolor(histo_data[1], histo_data[2], histo_data[0].T, shading="auto")

    # 3d plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = jnp.meshgrid(histo_data[1][:-1], histo_data[2][:-1])
    Z = histo_data[0].T

    # set z axis to something small
    ax.set_zlim(0, jnp.max(Z) * 1.1)
    ax.set_xlim(-PBC / 2, PBC / 2)
    ax.set_ylim(-PBC / 2, PBC / 2)
    ax.plot_surface(X, Y, Z, cmap="viridis")

    # set axes squared
    plt.axis("equal")
    plt.show()
