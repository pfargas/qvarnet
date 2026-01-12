import jax

device = "cuda"  # Change to 'cuda' to use GPU

jax.config.update("jax_platform_name", device)

print(jax.devices())

from jax import random
import jax.numpy as jnp
import time
from functools import partial
from flax import linen as nn
from jax import numpy as jnp

from typing import Callable
import matplotlib.pyplot as plt


@partial(jax.jit, static_argnames=("prob_fn",))
def mh_chain_with_all_random_nums(random_values, prob_fn, init_pos):
    # Placeholder implementation of mh_chain

    def mh_kernel(carry, random_values):
        position, old_prob = carry
        proposal = position + (2 * random_values[0] - 1)
        proposal_prob = prob_fn(proposal)
        accept_prob = jnp.minimum(1.0, proposal_prob / old_prob)
        accept = random_values[-1] < accept_prob
        new_position = jnp.where(accept, proposal, position)
        new_prob = jnp.where(accept, proposal_prob, old_prob)
        carry = (new_position, new_prob)
        return carry, new_position

    initial_prob = prob_fn(init_pos)
    carry = (init_pos, initial_prob)
    positions, _ = jax.lax.scan(mh_kernel, carry, random_values)
    return positions


sampler = jax.vmap(mh_chain_with_all_random_nums, in_axes=(0, None, 0))


class CustomDense(nn.Module):
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    β: float = 1.0  # parameter for scaling the weights

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            "kernel",
            self.kernel_init,  # Initialization function
            (inputs.shape[-1], self.features),
        )  # shape info.
        y = jnp.dot(inputs, self.β * kernel)
        bias = self.param("bias", self.bias_init, (self.features,))
        y += bias
        return y


class MLP_explicit(nn.Module):
    architecture: list
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.constant(0.5)
    bias_init: Callable = nn.initializers.zeros_init()
    β: float = 1.0  # parameter for scaling the weights

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.architecture) - 1):
            x = CustomDense(
                features=self.architecture[i + 1],
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                β=self.β,
            )(x)
            if i < len(self.architecture) - 2:
                x = self.hidden_activation(x)
        return x


class MLP(nn.Module):
    architecture: list
    hidden_activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.constant(0.5)
    bias_init: Callable = nn.initializers.zeros_init()

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


DoF = 1
hidden_units = 3
batch_size = 10
N_CHAINS = 10
N_STEPS_PER_CHAIN = 500


key1, key2 = random.split(random.key(0), 2)
x = random.uniform(key1, (batch_size, DoF))

model_mlp = MLP_explicit(architecture=[DoF, hidden_units, 1])
params = model_mlp.init(key2, x)
y = model_mlp.apply(params, x)

model_mlp_flax = MLP(architecture=[DoF, hidden_units, 1])
params_explicit = model_mlp_flax.init(key2, x)
y_explicit = model_mlp_flax.apply(params_explicit, x)

assert jnp.allclose(y, y_explicit), "Outputs do not match!"

β_array = [10**i for i in range(-5, 6)]

rand_vals = random.uniform(key1, (N_CHAINS, N_STEPS_PER_CHAIN, DoF + 1))
x_init = random.uniform(key2, (N_CHAINS, DoF))


for β in β_array:
    current_model = MLP_explicit(
        architecture=[DoF, hidden_units, 1],
        kernel_init=nn.initializers.lecun_normal(),
        bias_init=nn.initializers.zeros_init(),
        β=β,
    )
    current_params = current_model.init(key2, x)
    print(current_params)

    def current_prob_fn(x):
        psi = current_model.apply(current_params, x).squeeze()
        return jnp.square(psi)

    start_time = time.time()
    y_current, _ = sampler(rand_vals, current_prob_fn, x_init)
    end_time = time.time()
    print(f"β: {β}, Time taken: {end_time - start_time} seconds")
    print(y_current)
    plt.hist(y_current.flatten(), bins=50, density=True, alpha=0.6, label=f"β={β}")

x = jnp.linspace(-5, 5, 100)
x = x.reshape(-1, 1)
plt.plot(x, current_prob_fn(x), label=f"Posterior (β={β})", color="black")

plt.legend()
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Posterior Distributions")
plt.show()
