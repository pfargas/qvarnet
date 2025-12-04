import jax
import jax.numpy as jnp
from jax import random

from flax.training import train_state
from jax.scipy.integrate import trapezoid
from .callback import nan_callback

import numpy as np
import matplotlib.pyplot as plt
import os
from .sampler import mh_chain

import signal

try:
    from tqdm import tqdm

    tqdm_available = True
except ImportError:
    tqdm_available = False
    print("tqdm not found, progress bars will not be displayed.")


stop_requested = False


def signal_handler(signum, frame):
    global stop_requested
    stop_requested = True
    print("Signal received, will stop after current training step.")


signal.signal(signal.SIGINT, signal_handler)


def laplace(func, x):
    """Compute the Laplace operator of the model output with respect to inputs."""
    grad_fn = jax.grad(func)
    d2_dx2 = 0
    for i in range(x.shape[1]):
        d2_dx2 += jax.vmap(jax.grad(lambda xi: grad_fn(xi)[i]))(x)[:, i]
    return d2_dx2


# compute kinetic term with AD (correct)
def local_energy_batch(params, xs, model_apply):
    # xs: (batch, 1) or (batch,)
    # psi(x) -> scalar
    def psi_fn(x):
        # ensure input has shape (1,) as model expects last-dim features
        x = jnp.atleast_1d(x).reshape(1, -1)  # (1, DoF)
        return model_apply(params, x).squeeze()

    # second derivative per sample via AD
    d2psi = laplace(psi_fn, xs)

    psi_vals = jax.vmap(lambda x: psi_fn(x))(xs)  # shape (batch,)

    # avoid division by zero / small psi
    psi_safe = psi_vals + 1e-12

    kinetic = -0.5 * (d2psi / psi_safe)  # shape (batch,)
    # potential = 0.5 * (xs_flat**2)  # shape (batch,)
    potential = 0.5 * jnp.sum(xs**2, axis=1)
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
    return E, local_energy_per_point


def loss_and_grads(params, batch, model_apply):
    E, local_energy_per_point = energy_fn(params, batch, model_apply)
    centered_local_energy = jax.lax.stop_gradient(local_energy_per_point - E)

    oi = lambda p: 2 * jnp.mean(
        jax.lax.stop_gradient(local_energy_per_point - E)
        * log_psi(batch, p, model_apply).reshape(-1, 1)
    )

    def grad_log_psi_fn(x, params):
        pass

    # o_i = jax.vmap(lambda x: O_i_fn(x, params))(batch)  # (batch, n_params)

    grad_E = 2 * jnp.mean(centered_local_energy * o_i, axis=0)
    return E, grad_E


@jax.jit
def train_step(state, batch):
    E, grads = loss_and_grads(state.params, batch, state.apply_fn)
    new_state = state.apply_gradients(grads=grads)
    return new_state, E


def train(
    n_steps,
    init_params,
    shape,
    model_apply,
    optimizer,
    sampler_params,
    PBC=10,
    n_steps_sampler=500,
):

    state = train_state.TrainState.create(
        apply_fn=model_apply, params=init_params, tx=optimizer
    )

    def prob_fn(x, params):
        forward = model_apply(params, x).flatten()  # (batch,)
        out = jnp.square(forward)  # non-negative density probability
        return jnp.squeeze(out)  # scalar for scalar input, (batch,) for batch

    energy_history = []
    sampler = jax.vmap(
        mh_chain, in_axes=(0, None, None, None, None, 0, None), out_axes=0
    )
    n_chains = shape[0]
    DoF = shape[1] if len(shape) > 1 else 1
    rng_keys = random.split(random.PRNGKey(872643), n_chains)
    init_position = jnp.zeros(shape)  # start all chains at 0
    print(f"Initial positions shape: {init_position.shape}\n====================\n\n")
    wf_hist = []
    best_energy = jnp.inf
    best_params = None
    step_size = sampler_params.get("step_size", 1.0)

    for step in tqdm(range(n_steps)) if tqdm_available else range(n_steps):
        if stop_requested:
            break

        rng_keys = random.split(random.PRNGKey(step), n_chains)
        # with jax.profiler.TraceAnnotation("Sampling"):
        batch = sampler(
            rng_keys,
            n_steps_sampler,
            PBC,
            prob_fn,
            state.params,
            init_position,
            step_size,
        )
        # combine n_chains and n_steps_sampler into one big batch
        batch = batch.reshape(-1, DoF)  # (n_chains * n_steps_sampler, DoF)

        state, energy = train_step(state, batch)
        energy_history.append(energy)
        # wf_hist.append(state.params)
        # if energy < best_energy:
        #     best_energy = energy
        #     best_params = state.params
        # init_position = batch.reshape(
        #     n_chains, n_steps_sampler, DoF
        # )  # warm start next sampling

    return state.params, energy_history, wf_hist, best_params, best_energy
