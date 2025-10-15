import jax
import jax.numpy as jnp
from jax import random, grad, vmap

# from jax.tree import map
from functools import partial
import optax
from flax.training import train_state
from flax import linen as nn
from jax.scipy.integrate import trapezoid

import numpy as np
import json
import matplotlib.pyplot as plt
import os

try:
    from tqdm import tqdm

    tqdm_available = True
except ImportError:
    tqdm_available = False
    print("tqdm not found, progress bars will not be displayed.")


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


# compute kinetic term with AD (correct)
def local_energy_batch(params, xs, model_apply):
    # xs: (batch, 1) or (batch,)
    xs_flat = xs.squeeze()  # shape (batch,)

    # psi(x) -> scalar
    def psi_fn(x):
        # ensure input has shape (1,) as model expects last-dim features
        return model_apply(params, x.reshape(1, 1)).squeeze()

    # second derivative per sample via AD
    d2psi_fn = jax.vmap(jax.grad(jax.grad(psi_fn)))
    d2psi = d2psi_fn(xs_flat)  # shape (batch,)
    psi_vals = jax.vmap(lambda x: psi_fn(x))(xs_flat)  # shape (batch,)

    # avoid division by zero / small psi
    psi_safe = psi_vals + 1e-12

    kinetic = -0.5 * (d2psi / psi_safe)  # shape (batch,)
    potential = 0.5 * (xs_flat**2)  # shape (batch,)
    return (kinetic + potential).reshape(-1, 1)  # keep your (batch,1) convention


def log_psi(x, params, model_apply):
    psi = model_apply(params, x)
    return jnp.log(jnp.abs(psi) + 1e-8).squeeze()  # Add small constant to avoid log(0)


grad_log_psi = jax.grad(
    lambda params, x, model_apply: log_psi(x, params, model_apply), argnums=0
)  # CAREFUL WITH THE DERIVATIVES: WE WANT THE GRADIENT WRT THE PARAMETERS, so we turn the order of the p and x

# now grad_log_psi is a function that takes (params, x, model_apply) and returns the gradient of log_psi wrt params


def energy_fn(params, batch, model_apply):
    local_energy_per_point = local_energy_batch(params, batch, model_apply)
    E = jnp.mean(local_energy_per_point)
    return E


def energy_fn_trapezoidal(params, batch, model_apply):
    psi = jax.vmap(lambda x: model_apply(params, x))(batch)
    psi_squared = jnp.abs(psi) ** 2
    local_energy_per_point = local_energy_batch(params, batch, model_apply)

    energy_integrand = psi_squared * local_energy_per_point
    norm = trapezoid(psi_squared.squeeze(), batch.squeeze())
    integral = trapezoid(energy_integrand.squeeze(), batch.squeeze())
    return integral / norm


def loss_and_grads(params, batch, model_apply):
    local_energy_per_point = local_energy_batch(params, batch, model_apply)
    E = energy_fn(params, batch, model_apply)
    E_trap = energy_fn_trapezoidal(
        params, jnp.linspace(-5, 5, 1000).reshape(-1, 1), model_apply
    )
    E_centered = local_energy_per_point - E
    log_psi_grads = jax.vmap(lambda x: grad_log_psi(params, x, model_apply))(batch)
    grad_E = jax.tree_util.tree_map(
        lambda g: 2 * jnp.mean(E_centered[:, None] * g), log_psi_grads
    )

    grad_E_trapezoidal = jax.grad(energy_fn_trapezoidal, argnums=0)(
        params, batch, model_apply
    )

    jax.debug.print(
        "Energy sampled: {E}, Energy trapezoidal: {E_trap}", E=E, E_trap=E_trap
    )

    # compute the modulus of the gradient for logging
    # grad_E_mod = jnp.sqrt(
    #     sum([jnp.sum(jnp.abs(g) ** 2) for g in jax.tree_util.tree_leaves(grad_E)])
    # )
    # jax.debug.print("Energy: {E}, |grad_E|: {grad_E_mod}", E=E, grad_E_mod=grad_E_mod)
    # jax.debug.print(
    #     "Energy (trapezoidal): {E_trap}",
    #     E_trap=E_trap,
    # )
    return E_trap, grad_E_trapezoidal


@jax.jit
def train_step(state, batch):
    E, grads = loss_and_grads(state.params, batch, state.apply_fn)
    new_state = state.apply_gradients(grads=grads)
    return new_state, E


@partial(jax.jit, static_argnums=(1,))
def mh_kernel(rng_key, prob_fn, prob_params, position, prob, PBC=None):
    key1, key2 = random.split(rng_key)
    proposal = position + random.normal(key1, shape=position.shape) * 0.5
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
            subkey, prob_fn, prob_params, position, prob, PBC=None
        )
        return key, new_position, new_prob

    init_prob = prob_fn(init_position, prob_params)
    init_val = (rng_key, init_position, init_prob)
    _, positions, _ = jax.lax.fori_loop(0, n_steps, body_fn, init_val)
    return positions


def nan_callback(x):
    if jnp.isnan(x).any():
        return True
    return False


def train(
    n_steps, init_params, shape, model_apply, optimizer, PBC=10, n_steps_sampler=500
):

    state = train_state.TrainState.create(
        apply_fn=model_apply, params=init_params, tx=optimizer
    )

    def prob_fn(x, params):
        # Ensure x has a batch dimension, run the MLP, and return non-negative per-input values.
        x = jnp.atleast_1d(x).reshape(-1, 1)  # (batch, 1)
        forward = model_apply(params, x).flatten()  # (batch,)
        out = jnp.square(forward)  # non-negative density probability
        return jnp.squeeze(out)  # scalar for scalar input, (batch,) for batch

    energy_history = []
    # batch = jnp.linspace(-5,5,1000).reshape(-1,1)
    sampler = jax.vmap(mh_chain, in_axes=(0, None, None, None, None, 0), out_axes=0)
    n_chains = shape[0]
    rng_keys = random.split(random.PRNGKey(872643), n_chains)
    init_position = jax.random.normal(random.PRNGKey(0), (n_chains,)) * (PBC / 8)
    wf_hist = []
    best_energy = jnp.inf
    best_params = None
    x = jnp.linspace(-PBC / 2, PBC / 2, 1000).reshape(-1, 1)
    debugSampling = True

    os.makedirs("results", exist_ok=True)
    if debugSampling:
        # remove all images in results
        for f in os.listdir("results"):
            if f.endswith(".png"):
                os.remove(os.path.join("results", f))

    for step in tqdm(range(n_steps)) if tqdm_available else range(n_steps):
        rng_keys = random.split(random.PRNGKey(step), n_chains)
        batch = sampler(
            rng_keys, n_steps_sampler, PBC, prob_fn, state.params, init_position
        )
        batch = batch.reshape(-1, 1)  # Flatten the chains into a single batch

        state, energy = train_step(state, batch)
        energy_history.append(energy)
        wf_hist.append(state.params)
        if energy < best_energy:
            best_energy = energy
            best_params = state.params
        if nan_callback(energy):
            print("NaN detected in energy, stopping training.")
            break

        if step % 1000 == 0 and debugSampling:
            plt.clf()
            plt.title(f"Batch size: {batch.shape[0]}")
            plt.hist(batch, bins=int(np.ceil(np.sqrt(batch.size))), density=True)
            plt.plot(
                x,
                prob_fn(x, state.params)
                / trapezoid(prob_fn(x, state.params), x.flatten()),
                color="red",
            )
            plt.savefig(f"results/wavefunction_step_{step}.png")
            plt.clf()

        if step % 100 == 0 and not tqdm_available:
            print(f"Step {step}, Energy: {energy}")
            # print("Current parameters:")
            # print(state.params)
            print("==============================")

    return state.params, energy_history, wf_hist, best_params, best_energy


if __name__ == "__main__":
    model = MLP(architecture=[1, 5, 1])

    class StupidModel(nn.Module):
        alpha: jnp.ndarray

        @nn.compact
        def __call__(self, x):
            return jnp.exp(-self.alpha * x**2)

    # model = StupidModel(alpha=jnp.array(1.0))

    rng = jax.random.PRNGKey(0)
    input_shape = (5_000, 1)  # Batch size of 5000, input dimension
    params = model.init(rng, jnp.ones(input_shape) * 0.1)  # Initialize parameters
    PBC = 10
    params_fin, energy, wf_hist, best_params, best_energy = train(
        3,
        params,
        input_shape,
        model.apply,
        optax.adam(1e0),
        PBC=PBC,
        n_steps_sampler=1_000,
    )
    import matplotlib.pyplot as plt

    print(f"last energy: {energy[-1]}, before: {energy[-2]}")
    plt.plot(energy)
    plt.show()

    # Reconstruct wavefunction
    x = jnp.linspace(-PBC / 2, PBC / 2, 1000).reshape(-1, 1)
    psi_approx = model.apply(params_fin, x)
    print(type(psi_approx))
    print(psi_approx.shape)
    norm = jnp.sqrt(trapezoid((psi_approx**2).squeeze(), x.squeeze()))
    print(f"Norm: {norm}")
    psi_approx = psi_approx / norm
    print(
        f"Norm after normalization: {jnp.sqrt(trapezoid((psi_approx**2).squeeze(), x.squeeze()))}"
    )
    plt.plot(x, psi_approx**2)
    plt.plot(x, jnp.pi ** (-0.5) * jnp.exp(-(x**2)), linestyle="dashed")
    plt.show()
