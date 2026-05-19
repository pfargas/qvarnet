import collections
import signal
from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from .vmc_state import VMCState
from .callbacks import nan_callback
from .probability import build_prob_fn
from .sampling_step import sample_and_process
from .training_step import compute_step
from .config.training_setup import parse_sampler_params, TrainingConfig, SamplingConfig
from .config.coord_mode import CoordMode, LabCoords
from .utils.coord_transforms import build_effective_apply, init_shape_for_model
from .utils import load_doc, save_checkpoint, load_checkpoint
from .cusp import make_cusp_configs, make_cusp_pair_indices

try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    tqdm_available = False
    print("tqdm not found, progress bars will not be displayed.")


@jax.jit
def _update_step_size(step_size, acceptance_rate, min_step, max_step, target_acc, adaptation_rate):
    factor = 1.0 + adaptation_rate * (jnp.mean(acceptance_rate) - target_acc)
    return jnp.clip(step_size * factor, min_step, max_step)


@load_doc("train.txt")
def train(
    shape,
    model,
    optimizer,
    hamiltonian,
    training_config: TrainingConfig,
    sampler_params,
    coord_mode: CoordMode = None,
    qgt_config=None,
):
    """Train a VMC model using Metropolis-Hastings sampling."""
    if coord_mode is None:
        coord_mode = LabCoords()

    # Inject coord_mode into the hamiltonian so potential_energy always
    # receives lab coordinates regardless of which sampler space is used.
    hamiltonian = hamiltonian.replace(coord_mode=coord_mode)

    key = random.PRNGKey(training_config.rng_seed)

    init_shape = init_shape_for_model(shape, coord_mode)
    params = model.init(key, jnp.ones(init_shape))

    effective_apply = build_effective_apply(model.apply, coord_mode)
    state = VMCState.create(apply_fn=effective_apply, params=params, tx=optimizer)
    state = load_checkpoint(state, path=training_config.checkpoint_path, filename="checkpoint.msgpack")

    prob_fn = build_prob_fn(effective_apply, is_log_model=training_config.is_log_model)
    sampling_config = parse_sampler_params(sampler_params, is_log_prob=training_config.is_log_model)

    if training_config.init_positions == "normal":
        current_positions = jax.random.normal(key, shape) * 0.5
    elif training_config.init_positions == "zeros":
        current_positions = jnp.zeros(shape)
    else:
        raise ValueError(f"Unknown init_positions: {training_config.init_positions!r}")

    step_size = sampling_config.step_size
    state_history = []

    cusp_configs = None
    cusp_pair_i = None
    cusp_pair_j = None
    if training_config.use_cusp_condition:
        n_particles = shape[-1]
        L = getattr(hamiltonian, "L", sampling_config.PBC)
        cusp_configs = make_cusp_configs(
            n_particles=n_particles,
            L=L,
            epsilon=training_config.cusp_epsilon,
            n_configs_per_pair=training_config.cusp_n_configs_per_pair,
            rng_seed=training_config.cusp_rng_seed,
        )
        cusp_pair_i, cusp_pair_j = make_cusp_pair_indices(
            n_particles=n_particles,
            n_configs_per_pair=training_config.cusp_n_configs_per_pair,
        )

    @partial(
        jax.jit,
        static_argnames=["prob_fn", "hamiltonian", "sampling_config", "training_config"],
    )
    def full_update(state, key, current_pos, prob_fn, step_size, hamiltonian, sampling_config, training_config):
        key, subkey = jax.random.split(key)
        n_chains, DoF = current_pos.shape

        batch, new_pos, acceptance_rate = sample_and_process(
            key=subkey,
            prob_fn=prob_fn,
            prob_params=state.params,
            init_positions=current_pos,
            step_size=step_size,
            n_chains=n_chains,
            DoF=DoF,
            n_steps=sampling_config.chain_length,
            burn_in=sampling_config.thermalization_steps,
            thinning=sampling_config.thinning_factor,
            PBC=sampling_config.PBC,
            is_log_prob=sampling_config.is_log_prob,
        )

        # Centre-of-mass diagnostic (meaningful for 1D, n_dim=1)
        cm = jnp.sum(new_pos, axis=1) / new_pos.shape[-1]
        cm_mean_val = jnp.mean(cm)
        cm_std_val = jnp.std(cm)

        if not training_config.warm_walkers:
            new_pos = current_pos

        if training_config.is_update_step_size:
            step_size = _update_step_size(
                step_size,
                acceptance_rate,
                min_step=training_config.min_step,
                max_step=training_config.max_step,
                target_acc=training_config.target_acceptance,
                adaptation_rate=training_config.adaptation_rate,
            )

        new_state, E, sigma_e, grads = compute_step(
            state=state,
            batch=batch,
            hamiltonian=hamiltonian,
            is_log_model=training_config.is_log_model,
            use_qgt=training_config.use_qgt,
            qgt_config=qgt_config,
            use_cusp_condition=training_config.use_cusp_condition,
            cusp_configs=cusp_configs,
            cusp_alpha=training_config.cusp_alpha,
            cusp_pair_i=cusp_pair_i,
            cusp_pair_j=cusp_pair_j,
            cusp_epsilon=training_config.cusp_epsilon,
            cusp_n=training_config.cusp_n,
            cusp_C_n=training_config.cusp_C_n,
        )

        return new_state, key, new_pos, E, sigma_e, acceptance_rate, step_size, grads, cm_mean_val, cm_std_val

    # Signal handler — no global state, restores original handler on exit
    _stop = [False]

    def _handle_sigint(signum, frame):
        _stop[0] = True
        print("\nSignal received, stopping after current step.")

    original_handler = signal.signal(signal.SIGINT, _handle_sigint)

    init_steps = int(state.step)
    progress_bar = tqdm(range(init_steps, training_config.n_epochs), disable=not tqdm_available)

    try:
        for step in progress_bar:
            if _stop[0]:
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
                cm_std_single,
            ) = full_update(
                state=state,
                key=key,
                current_pos=current_positions,
                prob_fn=prob_fn,
                step_size=step_size,
                hamiltonian=hamiltonian,
                sampling_config=sampling_config,
                training_config=training_config,
            )

            state_history.append(
                state.replace(
                    energy=E,
                    std=sigma_e,
                    acceptance_rate=acceptance_rate,
                    step_size=step_size,
                    grads=grads,
                    cm_mean=float(cm_mean_single),
                    cm_std=float(cm_std_single),
                )
            )
            state = new_state

            if nan_callback(E):
                print(f"NaN detected at step {step}. Stopping.")
                save_checkpoint(state, path=training_config.checkpoint_path, filename="nan_checkpoint.msgpack")
                break

            if tqdm_available and step % 10 == 0:
                progress_bar.set_postfix(E=f"{E:.4f}", sigma_E=f"{sigma_e:.4f}")

            if training_config.save_checkpoints and step % 50 == 0:
                save_checkpoint(new_state, path=training_config.checkpoint_path, filename="checkpoint.msgpack")

    finally:
        signal.signal(signal.SIGINT, original_handler)

    # TrainResult unpacks as (history, cm_mean, cm_std) for backward compatibility
    TrainResult = collections.namedtuple("TrainResult", ["history", "cm_mean", "cm_std"])
    return TrainResult(
        history=state_history,
        cm_mean=[s.cm_mean for s in state_history],
        cm_std=[s.cm_std for s in state_history],
    )
