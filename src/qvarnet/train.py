import jax
import jax.numpy as jnp
from jax import random

from .vmc_state import VMCState
from .callbacks import *
from .samplers import mh_chain

from .hamiltonian import V, kinetic_term

import signal

from functools import partial

from .utils import load_doc, save_checkpoint, load_checkpoint

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


@partial(jax.jit, static_argnames=["model_apply"])
def local_energy_batch(params, xs, model_apply):
    # TODO: look into efficient laplacian implementations
    kinetic = kinetic_term(params, xs, model_apply)
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
    sigma_e = jnp.std(local_energy_per_point)
    return E, local_energy_per_point, sigma_e


# @partial(jax.jit, static_argnames=["model_apply"])
def loss_and_grads(params, batch, model_apply, score_factor=2.0):
    E, local_energy_per_point, sigma_e = energy_fn(params, batch, model_apply)
    loss = lambda p: 2 * jnp.mean(
        jax.lax.stop_gradient(local_energy_per_point - E)
        * log_psi(batch, p, model_apply).reshape(-1, 1)
    )
    grad_E = jax.grad(loss)(params)
    score = E + score_factor * sigma_e
    # Could i return the loss to monitor it?
    return E, sigma_e, grad_E, score


@jax.jit
def train_step(state, batch):
    E, sigma_e, grads, score = loss_and_grads(state.params, batch, state.apply_fn)
    new_state = state.apply_gradients(grads=grads)
    return new_state.replace(energy=E, std=sigma_e, score=score)


@load_doc("train.txt")
def train(
    n_epochs,
    shape,
    model,
    optimizer,
    sampler_params,
    rng_seed=0,
    hamiltonian_params=None,
    checkpoint_path="./",
    save_checkpoints=False,
):
    """Train a VMC model using Metropolis-Hastings sampling.
    Docs loaded from _docs/train.txt
    """
    key = random.key(rng_seed)

    sampler = jax.vmap(
        mh_chain,
        in_axes=(
            0,  # random_values (n_chains, n_steps, DoF + 1)
            None,  # PBC
            None,  # prob_fn
            None,  # prob_params
            0,  # init_position (n_chains, DoF)
            None,  # step_size
        ),
        out_axes=0,
    )

    params = model.init(key, jnp.ones(shape) * 0.1)  # Initialize parameters

    state = VMCState.create(apply_fn=model.apply, params=params, tx=optimizer)

    state = load_checkpoint(state, path=checkpoint_path, filename="checkpoint.msgpack")

    init_steps = state.n_step if hasattr(state, "n_step") else 0

    def prob_fn(x, params):
        forward = model.apply(params, x).flatten()  # (batch,)
        out = jnp.square(forward)  # non-negative density probability
        return jnp.squeeze(out)  # scalar for scalar input, (batch,) for batch

    n_chains = shape[0]
    DoF = shape[1] if len(shape) > 1 else 1

    energy_history = jnp.zeros(n_epochs)
    init_position = jnp.zeros(shape)  # start all chains at 0
    best_state = state

    step_size = sampler_params.get("step_size", 1.0)
    n_steps_sampler = sampler_params.get("chain_length", 500)

    burn_in_steps = sampler_params.get("thermalization_steps", 50)
    thinning_factor = sampler_params.get("thinning_factor", 5)

    PBC = sampler_params.get("PBC", 40.0)

    # --------------------------------------------
    # ---          TRAINING LOOP              ---
    # --------------------------------------------
    progress_bar = tqdm(range(init_steps, n_epochs), disable=not tqdm_available)
    for step in progress_bar:
        if stop_requested:
            break

        # --------------------------------------------
        # ---            SAMPLING STEP             ---
        # --------------------------------------------
        key, subkey = random.split(key)
        rand_nums = random.uniform(subkey, (n_chains, n_steps_sampler, DoF + 1))
        batch = sampler(
            rand_nums,
            PBC,
            prob_fn,
            state.params,
            init_position,
            step_size,
        )
        # combine n_chains and n_steps_sampler into one big batch
        batch = batch[:, burn_in_steps:, :]  # burn-in
        batch = batch[:, ::thinning_factor, :]  # Thinning to reduce correlations
        batch = batch.reshape(-1, DoF)  # (n_chains * n_steps_sampler, DoF)

        # check_size_batch(batch, step, n_chains, n_steps_sampler)

        # --------------------------------------------
        # ---          TRAINING STEP              ---
        # --------------------------------------------
        state = train_step(state, batch)

        energy_history = energy_history.at[step].set(state.energy)
        if save_checkpoints:
            save_checkpoint(state, path=checkpoint_path, filename="checkpoint.msgpack")

        if tqdm_available:
            # You can pass keyword arguments directly
            progress_bar.set_postfix(
                E=f"{state.energy:.2f}", best_score=f"{best_state.score:.3f}"
            )

        if state.score < best_state.score:
            best_state = state

    return energy_history, best_state


def check_size_batch(batch, step, n_chains, n_steps_sampler):
    with open("batch_debug.txt", "a") as f:
        f.write(f"# STEP {step}\n")
        f.write("BATCH INFO\n")
        f.write(f"Batch shape: {batch.shape}\n")
        f.write(f"Batch data type: {batch.dtype}\n")
        f.write(f"Batch size in bytes: {batch.nbytes} bytes\n")

        num = batch[0]
        f.write("EXPLORING SINGLE SAMPLE\n")
        size_in_bytes = num.nbytes
        f.write(f"Sample size: {num.shape}, Size in bytes: {size_in_bytes} bytes\n")
        f.write(f"Sample data: {num}\n")
        # check the attributes of the single num
        f.write(f"Type: {type(num)}\n")
        for mini_num in num:
            f.write(f"  Mini value: {mini_num}, Type: {type(mini_num)}\n")

        f.write(
            f"computed size as the product of shape dimensions and itemsize: {n_chains * n_steps_sampler * size_in_bytes} bytes\n"
        )
        f.write("\n")
