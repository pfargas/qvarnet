import signal
from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from .callbacks import CheckpointCallback, NaNCallback, ProgressCallback
from .config.coord_mode import CoordMode, LabCoords
from .config.training_setup import TrainingConfig, parse_sampler_params
from .cusp import make_cusp_configs, make_cusp_pair_indices
from .losses import CuspLoss
from .probability import build_prob_fn
from .qgt import DEFAULT_QGT_CONFIG, QGTConfig
from .sampling_step import sample_and_process
from .training_step import compute_step
from .utils import load_checkpoint, load_doc, save_run_config
from .vmc_state import VMCState

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


class TrainResult:
    """Result of a VMC training run.

    Attributes:
        history: list of VMCState, one per epoch (includes full params).
        cm_mean: list of float — per-epoch mean centre-of-mass.
        cm_std:  list of float — per-epoch std of centre-of-mass.

    Methods:
        best(n, metric): return the N states with the lowest value of metric.
    """

    def __init__(self, history, cm_mean, cm_std):
        self.history = history
        self.cm_mean = cm_mean
        self.cm_std = cm_std

    def best(self, n: int = 1, metric: str = "energy"):
        """Return the N VMCState objects with the lowest value of metric.

        Args:
            n: number of states to return.
            metric: "energy" or "std".

        Returns:
            List of VMCState sorted ascending by metric (best first).
        """
        key_fns = {
            "energy": lambda s: float(s.energy),
            "std":    lambda s: float(s.std),
        }
        if metric not in key_fns:
            raise ValueError(f"metric must be one of {list(key_fns)}, got {metric!r}")
        return sorted(self.history, key=key_fns[metric])[:n]

    def __iter__(self):
        # Backward compat: allows  history, cm_mean, cm_std = result
        return iter((self.history, self.cm_mean, self.cm_std))

    def __repr__(self):
        n = len(self.history)
        if n:
            last_e = self.history[-1].energy
            return f"TrainResult(n_steps={n}, last_energy={float(last_e):.6f})"
        return "TrainResult(n_steps=0)"


@load_doc("train.txt")
def train(
    shape,
    model,
    optimizer,
    hamiltonian,
    training_config: TrainingConfig,
    sampler_params,
    coord_mode: CoordMode = None,
    model_name: str = None,
    model_args: dict = None,
    qgt_config=None,
    auxiliary_losses: tuple = (),
    callbacks: list = None,
):
    """Train a VMC model using Metropolis-Hastings sampling."""
    if coord_mode is None:
        coord_mode = LabCoords()
    if qgt_config is None:
        qgt_config = DEFAULT_QGT_CONFIG
    elif isinstance(qgt_config, dict):
        qgt_config = QGTConfig(**qgt_config)

    hamiltonian = hamiltonian.replace(coord_mode=coord_mode)

    assert len(shape) == 2, f"shape must be (n_chains, dof), got {shape}"
    n_chains, dof = shape
    assert n_chains > 0 and dof > 0, f"shape dimensions must be positive, got {shape}"

    key = random.PRNGKey(training_config.rng_seed)

    init_shape = coord_mode.model_input_shape(shape)
    params = model.init(key, jnp.ones(init_shape))

    effective_apply = coord_mode.wrap_model_apply(model.apply)
    state = VMCState.create(apply_fn=effective_apply, params=params, tx=optimizer)
    state = load_checkpoint(state, path=training_config.checkpoint_path, filename="checkpoint.msgpack")

    if model_name is not None and model_args is not None:
        save_run_config(
            path=training_config.checkpoint_path,
            model_name=model_name,
            model_args=model_args,
            sample_shape=shape,
            coord_mode=coord_mode,
            training_config=training_config,
        )

    prob_fn = build_prob_fn(effective_apply)
    sampling_config = parse_sampler_params(sampler_params)

    if training_config.init_positions == "normal":
        current_positions = jax.random.normal(key, shape) * 0.5
    elif training_config.init_positions == "zeros":
        current_positions = jnp.zeros(shape)
    else:
        raise ValueError(f"Unknown init_positions: {training_config.init_positions!r}")

    assert current_positions.shape == (n_chains, dof), (
        f"init_positions shape mismatch: expected {(n_chains, dof)}, "
        f"got {current_positions.shape}"
    )

    step_size = sampling_config.step_size

    # Build auxiliary losses: cusp (if configured) + user-provided
    _cusp = training_config.cusp
    _aux_losses = []
    if _cusp is not None:
        n_particles = shape[-1]
        L = getattr(hamiltonian, "L", sampling_config.PBC)
        cusp_configs = make_cusp_configs(
            n_particles=n_particles,
            L=L,
            epsilon=_cusp.epsilon,
            n_configs_per_pair=_cusp.n_configs_per_pair,
            rng_seed=_cusp.rng_seed,
        )
        cusp_pair_i, cusp_pair_j = make_cusp_pair_indices(
            n_particles=n_particles,
            n_configs_per_pair=_cusp.n_configs_per_pair,
        )
        _aux_losses.append(CuspLoss(
            cusp_configs, cusp_pair_i, cusp_pair_j,
            alpha=_cusp.alpha,
            epsilon=_cusp.epsilon,
            n=_cusp.n,
            C_n=_cusp.C_n,
        ))
    _aux_losses.extend(auxiliary_losses)
    _auxiliary_losses = tuple(_aux_losses)

    @partial(
        jax.jit,
        static_argnames=["prob_fn", "hamiltonian", "sampling_config", "training_config"],
    )
    def full_update(state, key, current_pos, prob_fn, step_size, hamiltonian, sampling_config, training_config):
        key, subkey, lap_key = jax.random.split(key, 3)
        n_chains, dof = current_pos.shape

        batch, new_pos, acceptance_rate = sample_and_process(
            key=subkey,
            prob_fn=prob_fn,
            prob_params=state.params,
            init_positions=current_pos,
            step_size=step_size,
            n_chains=n_chains,
            dof=dof,
            n_steps=sampling_config.chain_length,
            burn_in=sampling_config.thermalization_steps,
            thinning=sampling_config.thinning_factor,
            PBC=sampling_config.PBC,
        )

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
            use_qgt=training_config.use_qgt,
            qgt_config=qgt_config,
            auxiliary_losses=_auxiliary_losses,
            key=lap_key,
        )

        return new_state, key, new_pos, E, sigma_e, acceptance_rate, step_size, grads, cm_mean_val, cm_std_val

    # Build callback list: always NaN guard, then user-supplied, then built-ins from config
    _callbacks = list(callbacks or [])
    _callbacks.insert(0, NaNCallback(training_config.checkpoint_path))
    if training_config.save_checkpoints:
        _callbacks.append(CheckpointCallback(training_config.checkpoint_path))

    state_history = []

    # Signal handler — no global state, restores original handler on exit
    _stop = [False]

    def _handle_sigint(signum, frame):
        _stop[0] = True
        print("\nSignal received, stopping after current step.")

    original_handler = signal.signal(signal.SIGINT, _handle_sigint)

    init_steps = int(state.step)
    progress_bar = tqdm(range(init_steps, training_config.n_epochs), disable=not tqdm_available)
    if tqdm_available:
        _callbacks.append(ProgressCallback(progress_bar))

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

            metrics = {
                "energy": float(E),
                "std": float(sigma_e),
                "acceptance_rate": acceptance_rate,
                "step_size": float(step_size),
                "cm_mean": float(cm_mean_single),
                "cm_std": float(cm_std_single),
            }

            if any(cb.on_step_end(step, state, metrics) for cb in _callbacks):
                break

    finally:
        signal.signal(signal.SIGINT, original_handler)
        for cb in _callbacks:
            cb.on_train_end(state, state_history)

    return TrainResult(
        history=state_history,
        cm_mean=[s.cm_mean for s in state_history],
        cm_std=[s.cm_std for s in state_history],
    )
