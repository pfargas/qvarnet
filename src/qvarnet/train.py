import jax
import jax.numpy as jnp
from jax import random

from flax.training import train_state
from jax.scipy.integrate import trapezoid
from .callback import nan_callback, update_best_params

import numpy as np
import matplotlib.pyplot as plt
import os
from .sampler import mh_chain

import signal

from functools import partial

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


# @partial(jax.jit, static_argnames=["func"])
def laplace(func, x):
    """Compute the Laplace operator of the model output with respect to inputs."""
    grad_fn = jax.grad(func)
    d2_dx2 = 0
    for i in range(x.shape[1]):
        d2_dx2 += jax.vmap(jax.grad(lambda xi: grad_fn(xi)[i]))(x)[:, i]
    return d2_dx2


# @jax.jit
def V(x):
    """Harmonic oscillator potential."""
    return 0.5 * jnp.sum(x**2, axis=1)  # sum over dimensions


# @partial(jax.jit, static_argnames=["model_apply"])
def local_energy_batch(params, xs, model_apply):
    def psi_fn(x):
        # ensure input has shape (1,) as model expects last-dim features
        x = jnp.atleast_1d(x).reshape(1, -1)  # (1, DoF)
        return model_apply(params, x).squeeze()

    d2psi = laplace(psi_fn, xs)

    psi_vals = jax.vmap(lambda x: psi_fn(x))(xs)  # shape (batch,)

    psi_safe = psi_vals + 1e-12

    kinetic = -0.5 * (d2psi / psi_safe)  # shape (batch,)
    potential = V(xs)  # shape (batch,)
    return (kinetic + potential).reshape(-1, 1)  # keep your (batch,1) convention


# @partial(jax.jit, static_argnames=["model_apply"])
def log_psi(x, params, model_apply):
    psi = model_apply(params, x)
    return jnp.log(jnp.abs(psi) + 1e-8).squeeze()  # Add small constant to avoid log(0)


def grad_log_psi(params, x, model_apply):
    """Compute the gradient of log(psi) with respect to parameters."""
    return jax.grad(lambda p: log_psi(x, p, model_apply), argnums=0)(params)


# @partial(jax.jit, static_argnames=["model_apply"])
def energy_fn(params, batch, model_apply):
    local_energy_per_point = local_energy_batch(params, batch, model_apply)
    E = jnp.mean(local_energy_per_point)
    return E, local_energy_per_point


# @partial(jax.jit, static_argnames=["model_apply"])
def loss_and_grads(params, batch, model_apply):
    E, local_energy_per_point = energy_fn(params, batch, model_apply)
    loss = lambda p: 2 * jnp.mean(
        jax.lax.stop_gradient(local_energy_per_point - E)
        * log_psi(batch, p, model_apply).reshape(-1, 1)
    )
    grad_E = jax.grad(loss)(params)
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
    rng_seed=0,
):
    r"""Main function to optimize the wavefunction parameters using Variational Monte Carlo.

    The optimizer approach is based on gradient descent:

    .. math::

        \theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)

    Where $\mathcal{L}$ is the loss function defined as:

    .. math::

        \Psi

    .. math::

        \mathcal{L}(\theta) = 2
        E_{x \sim |\psi_\theta(x)|^2}\Big[
            (E_{\rm loc}(x) - \langle E \rangle)
            \log |\psi_\theta(x)|
        \Big]



    Args:
        n_steps: Number of training steps."""

    state = train_state.TrainState.create(
        apply_fn=model_apply, params=init_params, tx=optimizer
    )

    def prob_fn(x, params):
        forward = model_apply(params, x).flatten()  # (batch,)
        out = jnp.square(forward)  # non-negative density probability
        return jnp.squeeze(out)  # scalar for scalar input, (batch,) for batch

    sampler = jax.vmap(
        mh_chain,
        in_axes=(
            0,
            None,
            None,
            None,
            0,
            None,
        ),  # random_values, PBC, prob_fn, prob_params, init_position, step_size
        out_axes=0,
    )

    n_chains = shape[0]
    DoF = shape[1] if len(shape) > 1 else 1

    key = random.key(rng_seed)
    energy_history = jnp.zeros(n_steps)
    init_position = jnp.zeros(shape)  # start all chains at 0
    wf_hist = []
    best_energy = jnp.inf
    best_params = init_params
    step_size = sampler_params.get("step_size", 1.0)

    for step in tqdm(range(n_steps)) if tqdm_available else range(n_steps):
        if stop_requested:
            break

        key, subkey = random.split(key)
        rand_nums = random.uniform(subkey, (n_chains, n_steps_sampler, DoF + 1))
        # --------------------------------------------
        # ---            SAMPLING STEP             ---
        # --------------------------------------------
        batch = sampler(
            rand_nums,
            PBC,
            prob_fn,
            state.params,
            init_position,
            step_size,
        )
        # combine n_chains and n_steps_sampler into one big batch
        batch = batch.reshape(-1, DoF)  # (n_chains * n_steps_sampler, DoF)

        # --------------------------------------------
        # ---          TRAINING STEP              ---
        # --------------------------------------------
        state, energy = train_step(state, batch)

        energy_history = energy_history.at[step].set(energy)
        # if one wants to monitor the device in which energy is stored
        # print(energy.device)
        # wf_hist.append(state.params)
        if energy < best_energy:
            best_energy = energy
            best_params = state.params

        # this is slower than the above approach
        # best_energy, best_params = update_best_params(
        #     energy, best_energy, state.params, best_params
        # )

    return state.params, energy_history, wf_hist, best_params, best_energy
