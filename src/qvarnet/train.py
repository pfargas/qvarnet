import jax
import jax.numpy as jnp
from jax import random

from .vmc_state import VMCState
from .callbacks import *
from .samplers import mh_chain
from .probability import build_prob_fn
from .sampling_step import sample_and_process
from .training_step import compute_step
from .config.training_setup import parse_sampler_params, parse_training_params

import signal

from functools import partial

from .utils import (
    load_doc,
    save_checkpoint,
    load_checkpoint,
    numerical_parameter_gradients,
    from_jacobi_to_lab,
)

try:
    from tqdm import tqdm

    tqdm_available = True
except ImportError:
    tqdm_available = False
    print("tqdm not found, progress bars will not be displayed.")

# QGT functions are now imported in training_step.py

stop_requested = False


def _cm_relative(x, n_particles, n_dim):
    """Subtract center-of-mass from particle coordinates.

    x:      (..., n_particles * n_dim)
    r:      (..., n_particles, n_dim)
    cm:     (..., 1, n_dim)
    return: (..., n_particles * n_dim)  — same shape as input, CM-subtracted
    """
    shape = x.shape[:-1]
    r = x.reshape(*shape, n_particles, n_dim)  # (..., n_particles, n_dim)
    cm = r.mean(axis=-2, keepdims=True)        # (..., 1, n_dim)
    return (r - cm).reshape(*shape, n_particles * n_dim)


def signal_handler(signum, frame):
    global stop_requested
    stop_requested = True
    print("Signal received, will stop after current training step.")


signal.signal(signal.SIGINT, signal_handler)

# Note: Energy computation and training step functions have been moved to
# training_step.py for better modularity. See compute_step() there.


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
    use_cm_coords=False,
    n_particles=None,
    n_dim=None,
    use_jacobi=False,
    n_dim_jacobi=1,
):
    """Train a VMC model using Metropolis-Hastings sampling.
    Docs loaded from _docs/train.txt
    """
    key = random.PRNGKey(rng_seed)

    if use_jacobi:
        # shape is (n_chains, N) — N Jacobi relative coords.
        # Model is psi(x_1,...,x_{N+1}) and must be init'd with N+1 lab coords.
        # n_particles_physical = N + 1 = shape[-1] // n_dim_jacobi + 1
        N = shape[-1] // n_dim_jacobi
        assert n_dim_jacobi == 1, "Jacobi path currently only implemented for n_dim_jacobi=1"
        n_particles_physical = N + 1
        init_shape = (*shape[:-1], n_particles_physical * n_dim_jacobi)
        params = model.init(key, jnp.ones(init_shape))
    else:
        params = model.init(key, jnp.ones(shape))

    if use_cm_coords:
        assert n_particles is not None and n_dim is not None, \
            "n_particles and n_dim are required when use_cm_coords=True"
        _base_apply = model.apply
        def effective_apply(params, x):
            return _base_apply(params, _cm_relative(x, n_particles, n_dim))
    elif use_jacobi:
        # Jacobi path: sampler works in N-dimensional Jacobi relative coord space
        # (shape = (n_chains, N)), but the model is psi(x_1,...,x_{N+1}) and
        # receives N+1 reconstructed lab coords.
        # effective_apply wraps the pad-and-reconstruct step transparently so that
        # kinetic energy, prob_fn, and the loss function all see the correct model.
        _base_apply = model.apply
        def effective_apply(params, x):
            # x: (..., N) Jacobi relative coords
            zeros = jnp.zeros((*x.shape[:-1], 1))
            u_tilde = jnp.concatenate([x, zeros], axis=-1)                         # (..., N+1)
            x_lab = from_jacobi_to_lab(u_tilde, n_particles_physical, n_dim_jacobi) # (..., N+1)
            return _base_apply(params, x_lab)
    else:
        effective_apply = model.apply

    state = VMCState.create(apply_fn=effective_apply, params=params, tx=optimizer)
    state = load_checkpoint(state, path=checkpoint_path, filename="checkpoint.msgpack")

    init_steps = state.n_step if hasattr(state, "n_step") else 0

    # Build probability function based on model type
    prob_fn = build_prob_fn(effective_apply, is_log_model=is_log_model)

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

    # shape = (n_chains, DoF)  — n_chains walkers, each with DoF = n_particles * n_dim degrees of freedom
    n_chains, DoF = shape

    # CM
    cm_mean = [] # center of mass for each epoch, mean of all the chains
    cm_std = []

    @partial(
        jax.jit,
        static_argnames=[
            "prob_fn",
            "hamiltonian",
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

        # new_pos: (n_chains, DoF)
        # cm: average position across DoF — NOTE: this mixes particles and dims,
        #     only meaningful for 1D (n_dim=1) where DoF = n_particles
        cm = jnp.sum(new_pos, axis=1) / new_pos.shape[-1]  # (n_chains,)
        cm_mean = jnp.mean(cm, axis=0)  # scalar
        cm_std = jnp.std(cm, axis=0)    # scalar

        # Update walker positions if requested
        if not warm_walkers:
            new_pos = current_pos  # Reset to initial positions

        if is_update_step_size:
            step_size = update_step_size(
                step_size, acceptance_rate, min_step=min_step, max_step=max_step
            )

        new_state, E, sigma_e, grads = compute_step(
            state=state,
            batch=batch,
            hamiltonian=hamiltonian,
            is_log_model=is_log_model,
            use_qgt=False,  # TODO: make configurable
            qgt_config=None,
        )

        return (
            new_state,
            key,
            new_pos,
            E,
            sigma_e,
            acceptance_rate,
            step_size,
            grads,
            cm_mean,
            cm_std
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
            grads,
            cm_mean_single,
            cm_std_single
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
                grads=grads,
            )
        )
        state = new_state

        cm_mean.append(cm_mean_single)
        cm_std.append(cm_std_single)

        if nan_callback(E):
            print(f"NaN detected in energy at step {step}. Stopping training.")
            with open(f"{checkpoint_path}/nan_checkpoint.msgpack", "wb") as f:
                save_checkpoint(state, path=checkpoint_path, filename="nan_checkpoint.msgpack")
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
    return state_history, cm_mean, cm_std
