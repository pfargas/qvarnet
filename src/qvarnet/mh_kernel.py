from functools import partial
import jax
import jax.numpy as jnp
from jax import random


@partial(jax.jit, static_argnums=(1,))
def mh_kernel(rng_key, prob_fn, prob_params, position, prob, PBC=None):
    key1, key2 = random.split(rng_key)
    proposal = position + random.normal(key1, shape=position.shape)
    # ensure PBC
    # if PBC is not None:
    #     proposal = ((proposal + 0.5 * PBC) % (PBC)) - 0.5 * PBC

    proposal = jax.lax.cond(
        PBC is None,
        lambda p: p,
        lambda p: ((p + 0.5 * PBC) % PBC) - 0.5 * PBC,
        proposal,
    )
    proposal_prob = prob_fn(proposal, prob_params)
    accept = jax.random.uniform(key2) < (proposal_prob / prob)
    new_position = jnp.where(accept, proposal, position)
    new_prob = jnp.where(accept, proposal_prob, prob)
    return new_position, new_prob


@partial(jax.jit, static_argnums=(1, 2))
def mh_chain(rng_key, n_steps, prob_fn, prob_params, init_position):
    def body_fn(i, val):
        key, position, prob = val
        _, key = random.split(key)
        new_position, new_prob = mh_kernel(
            key, prob_fn, prob_params, position, prob, PBC=5.0
        )
        return key, new_position, new_prob

    init_prob = prob_fn(init_position, prob_params)
    init_val = (rng_key, init_position, init_prob)
    _, positions, _ = jax.lax.fori_loop(0, n_steps, body_fn, init_val)
    return positions


if __name__ == "__main__":
    from flax import linen as nn

    rng_key = random.PRNGKey(42)
    n_chains = 10000
    n_steps = 100
    s = 5.0
    init_position = jnp.zeros(n_chains)  # (nchains,)
    rng_keys = jax.random.split(rng_key, n_chains)  # (nchains,)
    print(rng_keys.shape)
    run_mh_chain = jax.vmap(mh_chain, in_axes=(0, None, None, None, 0), out_axes=0)

    def prob_fn(x, s):
        return jnp.exp(-0.5 * (x / s) ** 2) / (s * jnp.sqrt(2 * jnp.pi))

    # Define a simple MLP using Flax Linen
    class MLP(nn.Module):
        architecture: list
        hidden_activation: callable = nn.tanh

        @nn.compact
        def __call__(self, x):
            for i in range(len(self.architecture) - 1):
                x = nn.Dense(features=self.architecture[i + 1])(x)
                if i < len(self.architecture) - 2:
                    x = self.hidden_activation(x)
            return x

    # Initialize model and parameters
    model = MLP(architecture=[1, 50, 1])
    print(model)
    rng = jax.random.PRNGKey(0)
    input_shape = (1000, 1)  # Batch size of 1000, input dimension
    params = model.init(rng, jnp.ones(input_shape))  # Initialize parameters

    def prob_fn(x, params):
        # Ensure x has a batch dimension, run the MLP, and return non-negative per-input values.
        x = jnp.atleast_1d(x).reshape(-1, 1)  # (batch, 1)
        forward = model.apply(params, x).flatten()  # (batch,)
        out = jnp.square(forward)  # non-negative density proxy
        return jnp.squeeze(out)  # scalar for scalar input, (batch,) for batch

    samples = run_mh_chain(rng_keys, n_steps, prob_fn, params, init_position)
    import matplotlib.pyplot as plt
    import numpy as np

    print("computing samples...")
    samples.block_until_ready()  # stop the code until samples are ready
    print("samples computed")

    # Convert JAX arrays to numpy before plotting
    samples_np = np.array(samples.flatten())
    print(samples_np.shape)

    plt.hist(
        samples_np,
        bins=int(np.sqrt(n_chains)),
        density=True,
        alpha=0.7,
        label="MCMC samples",
    )

    x = jnp.linspace(min(samples_np), max(samples_np), 500)
    x_np = np.array(x)
    y_np = np.array(prob_fn(x, params))
    norm = np.trapezoid(y_np, x_np)
    y_np = y_np / norm  # normalize the density

    plt.plot(x_np, y_np, "r-", linewidth=2, label="True distribution", alpha=0.7)
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("MCMC Sampling Results")
    plt.show()
