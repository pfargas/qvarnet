"""Refactored statistical tests script.

- Clear structure: utilities, model builders, sampler, and main.
- mh_chain_with_all_random_nums takes `prob_fn` and `params` separately so params are not treated as static by JAX.
- Renamed the Greek beta parameter to `beta` for readability.
"""

import time
from functools import partial
from typing import Callable

import jax
from jax import random
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt

# Device selection
device = "cuda"  # 'cpu' or 'cuda'
jax.config.update("jax_platform_name", device)
print("Using devices:", jax.devices())


# -----------------------------
# Metropolis-Hastings sampler
# -----------------------------
def mh_chain_with_all_random_nums(random_values, prob_fn, params, init_pos):
    """Run a single MH chain using a pre-supplied sequence of random values.

    Args:
        random_values: array of shape (n_steps, DoF+1). For each step the first DoF values
                       are proposals in [0,1), and the last value is a uniform for acceptance.
        prob_fn: callable(prob_inputs, params) -> nonnegative density (or unnormalized).
        params: parameters passed to prob_fn (kept dynamic so not static/hashing issues).
        init_pos: initial position (array of shape (DoF,) or (DoF,)).

    Returns:
        positions: array of shape (n_steps, DoF) with the positions visited by the chain.
    """

    def mh_kernel(carry, rv):
        pos, old_prob = carry
        # proposal: map rv[0:DoF] from [0,1) to proposal perturbation in [-1,1)
        proposal = pos + (2 * rv[:-1] - 1)
        proposal_prob = prob_fn(proposal, params)
        accept_prob = jnp.minimum(1.0, proposal_prob / old_prob)
        accept = rv[-1] < accept_prob
        new_pos = jnp.where(accept, proposal, pos)
        new_prob = jnp.where(accept, proposal_prob, old_prob)
        new_carry = (new_pos, new_prob)
        return new_carry, new_pos

    init_prob = prob_fn(init_pos, params)
    carry = (init_pos, init_prob)
    positions, _ = jax.lax.scan(mh_kernel, carry, random_values)
    return positions


# Vectorize across multiple chains
sampler = jax.vmap(mh_chain_with_all_random_nums, in_axes=(0, None, None, 0))


# -----------------------------
# Small MLP implementations
# -----------------------------
class CustomDense(nn.Module):
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    beta: float = 1.0  # scale factor for kernel

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )
        y = jnp.dot(inputs, self.beta * kernel)
        bias = self.param("bias", self.bias_init, (self.features,))
        return y + bias


class MLPExplicit(nn.Module):
    architecture: list
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.constant(0.5)
    bias_init: Callable = nn.initializers.zeros
    beta: float = 1.0

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.architecture) - 1):
            x = CustomDense(
                features=self.architecture[i + 1],
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                beta=self.beta,
            )(x)
            if i < len(self.architecture) - 2:
                x = self.hidden_activation(x)
        return x


class MLP(nn.Module):
    architecture: list
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.constant(0.5)
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.architecture) - 1):
            x = nn.Dense(
                features=self.architecture[i + 1],
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(x)
            if i < len(self.architecture) - 2:
                x = self.hidden_activation(x)
        return x


# -----------------------------
# Utilities for building prob functions
# -----------------------------


def make_prob_fn_from_model(model: nn.Module, params):
    """Return a prob_fn(x, params) that evaluates model and returns nonnegative values.

    The model is assumed to output a real scalar per input; we square it to ensure nonnegativity
    (this mirrors psi^2 in the original script).
    """

    def prob_fn(x, p):
        x_arr = jnp.atleast_1d(x).reshape(-1, x.shape[-1] if x.ndim > 1 else 1)
        out = model.apply(p, x_arr).squeeze()
        # If out is an array of shape (batch, ) return it; if scalar return scalar
        return jnp.squeeze(jnp.square(out)) + 1e-12

    return prob_fn


# -----------------------------
# Main experiment
# -----------------------------


def main():
    DoF = 1
    hidden_units = 3
    batch_size = 10
    N_CHAINS = 100
    N_STEPS_PER_CHAIN = 500

    key = random.PRNGKey(0)
    key, k1, k2 = random.split(key, 3)

    # Example inputs for initialization/testing
    x = random.uniform(k1, (batch_size, DoF))

    # Sanity check: explicit vs flax-built MLP
    model_explicit = MLPExplicit(architecture=[DoF, hidden_units, 1])
    params_explicit = model_explicit.init(k2, x)
    y_explicit = model_explicit.apply(params_explicit, x)

    model_flax = MLP(architecture=[DoF, hidden_units, 1])
    params_flax = model_flax.init(k2, x)
    y_flax = model_flax.apply(params_flax, x)

    assert jnp.allclose(y_explicit, y_flax, atol=1e-6), "Outputs do not match!"

    # Prepare random numbers for all chains: shape (N_CHAINS, N_STEPS_PER_CHAIN, DoF+1)
    key, k_rv, k_init = random.split(key, 3)
    rand_vals = random.uniform(k_rv, (N_CHAINS, N_STEPS_PER_CHAIN, DoF + 1))
    x_init = random.uniform(k_init, (N_CHAINS, DoF))

    beta_array = [10**i for i in range(-2, 3)]

    plt.figure(figsize=(8, 6))
    x_plot = jnp.linspace(-5, 5, 200).reshape(-1, 1)

    for beta in beta_array:
        # Build a model with the desired beta scaling
        current_model = MLPExplicit(
            architecture=[DoF, hidden_units, 1],
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.zeros,
            beta=beta,
        )
        key, k_init_model = random.split(key)
        current_params = current_model.init(k_init_model, x)

        # Build a prob function that accepts (x, params)
        current_prob_fn = make_prob_fn_from_model(current_model, current_params)

        # Run sampler and time it
        start_time = time.time()
        samples, _ = sampler(rand_vals, current_prob_fn, current_params, x_init)
        jax.block_until_ready(samples)
        end_time = time.time()

        print(f"beta: {beta}, time: {end_time - start_time:.4f}s")

        samples_np = jnp.array(samples).reshape(-1, DoF)
        current_distribution = jnp.array(
            current_prob_fn(x_plot, current_params)
        ).flatten()
        plt.plot(
            x_plot.flatten(),
            current_distribution
            / jnp.trapezoid(current_distribution, x_plot.flatten()),
            label=f"beta={beta}",
        )
        plt.hist(
            samples_np.flatten(), bins=50, density=True, alpha=0.6, label=f"beta={beta}"
        )

    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Posterior Distributions for different beta values")
    plt.show()


if __name__ == "__main__":
    main()
