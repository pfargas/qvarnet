import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree

from .vmc_state import VMCState
from .callbacks import *
from .samplers import mh_chain
from .probability import build_prob_fn
from .sampling_step import sample_and_process
from .config.training_setup import parse_sampler_params, parse_training_params

import signal

from functools import partial

from .utils import (
    load_doc,
    save_checkpoint,
    load_checkpoint,
    numerical_parameter_gradients,
)

try:
    from tqdm import tqdm

    tqdm_available = True
except ImportError:
    tqdm_available = False
    print("tqdm not found, progress bars will not be displayed.")

from .qgt import (
    compute_natural_gradient,
    DEFAULT_QGT_CONFIG,
)

stop_requested = False


def signal_handler(signum, frame):
    global stop_requested
    stop_requested = True
    print("Signal received, will stop after current training step.")


signal.signal(signal.SIGINT, signal_handler)


def compute_local_energy(hamiltonian, params, samples, model_apply, is_log_model):
    local_energy = hamiltonian.local_energy(
        params, samples, model_apply, is_log_model=is_log_model
    )
    return local_energy.reshape(-1, 1)


@partial(jax.jit, static_argnames=["model_apply"])
def log_psi(x, params, model_apply):
    psi = model_apply(params, x)
    return jnp.log(jnp.abs(psi)).squeeze()  # + 1e-8).squeeze()


@partial(jax.jit, static_argnames=["model_apply", "is_log_model"])
def energy_fn(hamiltonian, params, batch, model_apply, is_log_model):
    local_energy_per_point = compute_local_energy(
        hamiltonian, params, batch, model_apply, is_log_model=is_log_model
    )
    E = jnp.mean(local_energy_per_point)
    sigma_e = jnp.std(local_energy_per_point)
    return E, local_energy_per_point, sigma_e


@partial(jax.jit, static_argnames=["model_apply", "is_log_model"])
def energy_and_grads(hamiltonian, params, batch, model_apply, is_log_model):
    E, E_loc, sigma_e = energy_fn(
        hamiltonian, params, batch, model_apply, is_log_model=is_log_model
    )
    if not is_log_model:
        loss = lambda p: 2 * jnp.mean(
            jax.lax.stop_gradient(E_loc - E)
            * log_psi(batch, p, model_apply).reshape(-1, 1)
        )
    else:
        loss = lambda p: 2 * jnp.mean(
            jax.lax.stop_gradient(E_loc - E) * model_apply(p, batch).reshape(-1, 1)
        )
    grad_E = jax.grad(loss)(params)
    return E, sigma_e, grad_E


@partial(jax.jit, static_argnames=["is_log_model", "use_qgt", "qgt_config"])
def train_step(
    state,
    samples,
    hamiltonian,
    is_log_model=False,
    use_qgt=False,
    qgt_config=DEFAULT_QGT_CONFIG.to_dict(),
):
    E, sigma_e, grads = energy_and_grads(
        hamiltonian, state.params, samples, state.apply_fn, is_log_model=is_log_model
    )
    if not use_qgt:
        new_state = state.apply_gradients(grads=grads)
    else:
        natural_grad_flat, unravel_fn = compute_natural_gradient(
            state.params, samples, state.apply_fn, grads, qgt_config
        )
        learning_rate = qgt_config.get("learning_rate", 1e-3)
        new_params_flat = (
            ravel_pytree(state.params)[0] - learning_rate * natural_grad_flat
        )
        new_params = unravel_fn(new_params_flat)
        new_state = state.replace(params=new_params)
    return new_state, E, sigma_e


@jax.jit
def update_step_size(
    step_size,
    acceptance_rate,
    min_step,
    max_step,
    target_acc=0.5,
    adaptation_rate=0.1,
):
    factor = 1.0 + adaptation_rate * (jnp.mean(acceptance_rate) - target_acc)
    new_step_size = jnp.clip(step_size * factor, min_step, max_step)
    return new_step_size


@load_doc("train.txt")
def train(
    n_epochs,
    shape,
    model,
    optimizer,
    sampler_params,
    hamiltonian,
    rng_seed=0,
    checkpoint_path="./",
    save_checkpoints=False,
    init_positions="normal",
    warm_walkers=False,
    min_step=1e-5,
    max_step=5.0,
    is_update_step_size=False,
    is_log_model=False,
):
    """Train a VMC model using Metropolis-Hastings sampling.
    Docs loaded from _docs/train.txt
    """
    key = random.PRNGKey(rng_seed)

    params = model.init(key, jnp.ones(shape))
    state = VMCState.create(apply_fn=model.apply, params=params, tx=optimizer)
    state = load_checkpoint(state, path=checkpoint_path, filename="checkpoint.msgpack")

    init_steps = state.n_step if hasattr(state, "n_step") else 0

    # Build probability function based on model type
    prob_fn = build_prob_fn(model.apply, is_log_model=is_log_model)

    state_history = []

    # Parse configuration into typed dataclasses
    sampling_config = parse_sampler_params(sampler_params, is_log_prob=is_log_model)

    if init_positions == "normal":
        current_positions = jax.random.normal(key, shape) * 0.5
    elif init_positions == "zeros":
        current_positions = jnp.zeros(shape)
    else:
        raise ValueError(f"Unknown init_positions: {init_positions}")

    # Extract sampling parameters
    step_size = sampling_config.step_size
    n_steps_sampler = sampling_config.chain_length
    burn_in_steps = sampling_config.thermalization_steps
    thinning_factor = sampling_config.thinning_factor
    PBC = sampling_config.PBC

    # Get chain structure
    n_chains, DoF = shape

    @partial(
        jax.jit,
        static_argnames=[
            "n_chains",
            "DoF",
            "n_steps",
            "burn_in",
            "thinning",
            "PBC",
            "warm_walkers",
            "is_update_step_size",
            "is_log_model",
        ],
    )
    def full_update(
        state,
        key,
        current_pos,
        prob_fn,
        step_size,
        n_chains,
        DoF,
        n_steps,
        burn_in,
        thinning,
        PBC,
        hamiltonian,
        min_step,
        max_step,
        warm_walkers=False,
        is_update_step_size=False,
        is_log_model=False,
    ):
        key, subkey = jax.random.split(key)

        # Sample from MCMC
        batch, new_pos, acceptance_rate = sample_and_process(
            key=subkey,
            prob_fn=prob_fn,
            prob_params=state.params,
            init_positions=current_pos,
            step_size=step_size,
            n_chains=n_chains,
            DoF=DoF,
            n_steps=n_steps,
            burn_in=burn_in,
            thinning=thinning,
            PBC=PBC,
            is_log_prob=is_log_model,
        )

        # Update walker positions if requested
        if not warm_walkers:
            new_pos = current_pos  # Reset to initial positions

        if is_update_step_size:
            step_size = update_step_size(
                step_size, acceptance_rate, min_step=min_step, max_step=max_step
            )

        new_state, E, sigma_e = train_step(
            state, batch, hamiltonian, is_log_model=is_log_model
        )

        return (
            new_state,
            key,
            new_pos,
            E,
            sigma_e,
            acceptance_rate,
            step_size,
        )

    progress_bar = tqdm(range(init_steps, n_epochs), disable=not tqdm_available)

    for step in progress_bar:
        if stop_requested:
            break

        (
            new_state,
            key,
            current_positions,
            E,
            sigma_e,
            acceptance_rate,
            step_size,
        ) = full_update(
            state=state,
            key=key,
            current_pos=current_positions,
            prob_fn=prob_fn,
            step_size=step_size,
            n_chains=n_chains,
            DoF=DoF,
            n_steps=n_steps_sampler,
            burn_in=burn_in_steps,
            thinning=thinning_factor,
            PBC=PBC,
            hamiltonian=hamiltonian,
            min_step=min_step,
            max_step=max_step,
            warm_walkers=warm_walkers,
            is_update_step_size=is_update_step_size,
            is_log_model=is_log_model,
        )

        state_history.append(
            state.replace(
                energy=E,
                std=sigma_e,
                acceptance_rate=acceptance_rate,
                step_size=step_size,
            )
        )
        state = new_state

        if nan_callback(E):
            print(f"NaN detected in energy at step {step}. Stopping training.")
            break

        if tqdm_available and step % 10 == 0:
            progress_bar.set_postfix(
                E=f"{E:.2f}",
                sigma_E=f"{sigma_e:.2f}",
            )

        if save_checkpoints and step % 50 == 0:
            save_checkpoint(
                new_state, path=checkpoint_path, filename="checkpoint.msgpack"
            )
    return state_history
