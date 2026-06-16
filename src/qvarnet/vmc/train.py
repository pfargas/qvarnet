import signal
import time
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.flatten_util import ravel_pytree

from ..callbacks import (
    CheckpointCallback,
    NaNCallback,
    ProgressCallback,
    RunOutputCallback,
    SnapshotCallback,
)
from ..config.coord_mode import CoordMode, LabCoords
from ..config.training_setup import SamplingConfig, TrainingConfig, parse_sampler_params
from ..geometry.qgt import DEFAULT_QGT_CONFIG, QGTConfig
from ..losses import CuspLoss, make_cusp_configs, make_cusp_pair_indices
from ..samplers import geometric_betas, sample_and_process, sample_parallel_tempering
from ..utils import load_checkpoint, load_doc, save_run_config
from .metrics_history import MetricsHistory
from .probability import build_prob_fn
from .train_result import TrainResult
from .training_step import compute_step
from .vmc_state import VMCState

try:
    from tqdm import tqdm

    tqdm_available = True
except ImportError:
    tqdm_available = False
    print("tqdm not found, progress bars will not be displayed.")


def _is_periodic_ansatz(model) -> bool:
    """Best-effort detection of a PeriodicBoundary transform on the model.

    Checks ``LogWavefunction.transform`` and ``BoundaryModel.boundary``. Used only for
    the PBC sanity warning in ``train()`` — false negatives are harmless (no warning).
    """
    from ..boundaries import PeriodicBoundary

    for attr in ("transform", "boundary"):
        if isinstance(getattr(model, attr, None), PeriodicBoundary):
            return True
    return False


@jax.jit
def _update_step_size(
    step_size, acceptance_rate, min_step, max_step, target_acc, adaptation_rate
):
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
    model_name: str = None,
    model_args: dict = None,
    qgt_config=None,
    auxiliary_losses: tuple = (),
    callbacks: list = None,
    select="std",
    k_best: int = 3,
):
    """Train a VMC model using Metropolis-Hastings sampling.

    Parameter retrieval (roadmap step 8): unless a ``SnapshotCallback`` is passed explicitly,
    the ``k_best`` best-by-``select`` parameter sets are retained and exposed on the returned
    ``TrainResult`` as ``best_params()`` / ``best_k_params(n)``; the final-epoch params are
    always available as ``result.final_params``. ``select`` is a string ("std" default,
    "energy", "e_plus_sigma", or any metrics key) or a callable ``(metrics_dict) -> float``
    (lower = better). Set ``k_best=0`` to keep nothing.
    """
    if coord_mode is None:
        coord_mode = LabCoords()
    if qgt_config is None:
        qgt_config = DEFAULT_QGT_CONFIG
    elif isinstance(qgt_config, dict):
        qgt_config = QGTConfig(**qgt_config)

    hamiltonian = hamiltonian.replace(coord_mode=coord_mode)

    # Stochastic reconfiguration is applied as a preconditioner inside compute_step,
    # so the SR step size η must live in the optimizer. Override the passed optimizer
    # with SGD(η=qgt learning rate): apply_gradients(natural_grad) then does
    # θ ← θ - η·S⁻¹∇E, advances state.step, and supports optax LR schedules (pass a
    # schedule as qgt_config.learning_rate). A passed Adam optimizer is ignored here.
    if training_config.use_qgt:
        optimizer = optax.sgd(qgt_config.learning_rate)

    assert len(shape) == 2, f"shape must be (n_chains, dof), got {shape}"
    n_chains, dof = shape
    assert n_chains > 0 and dof > 0, f"shape dimensions must be positive, got {shape}"

    key = random.PRNGKey(training_config.rng_seed)

    init_shape = coord_mode.model_input_shape(shape)
    params = model.init(key, jnp.ones(init_shape))

    effective_apply = coord_mode.wrap_model_apply(model.apply)
    state = VMCState.create(apply_fn=effective_apply, params=params, tx=optimizer)
    state = load_checkpoint(
        state, path=training_config.checkpoint_path, filename="checkpoint.msgpack"
    )

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
    if isinstance(sampler_params, SamplingConfig):
        sampling_config = sampler_params
    else:
        sampling_config = parse_sampler_params(sampler_params)

    # Resolve the parallel-tempering ladder once (a concrete tuple captured by full_update).
    _pt_betas = None
    if sampling_config.sampler == "pt":
        _pt_betas = sampling_config.pt_betas or geometric_betas(
            sampling_config.pt_n_replicas, sampling_config.pt_beta_min
        )

    # PBC sanity check: the periodic-ansatz toggle (model transform) and the PBC-sampler
    # toggle (sampling_config.box_L) are independent by design, but a mismatch is a likely
    # user error — warn, don't forbid. A periodic ansatz with an unwrapped sampler is still
    # correct for the energy (|ψ|² is periodic) but yields unwrapped coords for observables;
    # a wrapped sampler with a non-periodic ansatz biases the energy at the box face.
    ansatz_periodic = _is_periodic_ansatz(model)
    sampler_periodic = bool(sampling_config.box_L)
    if ansatz_periodic and not sampler_periodic:
        warnings.warn(
            "Model uses a PeriodicBoundary ansatz but the sampler is not wrapped "
            "(sampling_config.box_L is None): walkers diffuse on the covering space. "
            "Energy is unbiased, but sampled positions are unwrapped — fold them (or set "
            "box_L) before computing position-binned observables.",
            stacklevel=2,
        )
    elif sampler_periodic and not ansatz_periodic:
        warnings.warn(
            "PBC sampler is enabled (box_L set) but the ansatz is not periodic: log|ψ| is "
            "discontinuous across the box face, so the energy is biased there. Use a "
            "PeriodicBoundary transform (and periodic Jastrow) for a valid PBC state.",
            stacklevel=2,
        )

    if training_config.init_positions == "normal":
        current_positions = jax.random.normal(key, shape) * 0.5
    elif training_config.init_positions == "zeros":
        current_positions = jnp.zeros(shape)
    elif training_config.init_positions == "uniform":
        # Uniform over the periodic box [0, L)^dof — the correct prior for a homogeneous
        # (untrapped) gas. Starting from a "normal" speck in a large box (L ~ (N/x)^{1/3}
        # can be hundreds of scattering lengths) leaves the walkers unequilibrated for
        # thousands of MH steps; uniform init removes that burn-in entirely.
        if not sampling_config.box_L:
            raise ValueError("init_positions='uniform' requires sampling_config.box_L (a PBC box)")
        current_positions = jax.random.uniform(key, shape) * sampling_config.box_L
    else:
        raise ValueError(f"Unknown init_positions: {training_config.init_positions!r}")

    assert current_positions.shape == (
        n_chains,
        dof,
    ), f"init_positions shape mismatch: expected {(n_chains, dof)}, got {current_positions.shape}"

    step_size = sampling_config.step_size

    # Build auxiliary losses: cusp (if configured) + user-provided
    _cusp = training_config.cusp
    _aux_losses = []
    if _cusp is not None:
        n_particles = shape[-1]
        L = _cusp.L if _cusp.L is not None else getattr(hamiltonian, "L", None)
        if L is None:
            raise ValueError(
                "CuspConfig requires a box size L, but neither CuspConfig.L nor "
                "hamiltonian.L is set. Pass cusp=CuspConfig(L=...) to TrainingConfig."
            )
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
        _aux_losses.append(
            CuspLoss(
                cusp_configs,
                cusp_pair_i,
                cusp_pair_j,
                alpha=_cusp.alpha,
                epsilon=_cusp.epsilon,
                n=_cusp.n,
                C_n=_cusp.C_n,
            )
        )
    _aux_losses.extend(auxiliary_losses)
    _auxiliary_losses = tuple(_aux_losses)

    @partial(
        jax.jit,
        static_argnames=[
            "prob_fn",
            "hamiltonian",
            "sampling_config",
            "training_config",
        ],
    )
    def full_update(
        state,
        key,
        current_pos,
        prob_fn,
        step_size,
        hamiltonian,
        sampling_config,
        training_config,
    ):
        key, subkey, lap_key = jax.random.split(key, 3)
        n_chains, dof = current_pos.shape

        if sampling_config.sampler == "pt":
            batch, new_pos, acceptance_rate = sample_parallel_tempering(
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
                betas=_pt_betas,
                swap_every=sampling_config.swap_every,
                box_L=sampling_config.box_L or 0.0,
                scale_steps=sampling_config.pt_scale_steps,
            )
        else:
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
                block_size=sampling_config.block_size,
                box_L=sampling_config.box_L or 0.0,
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

        new_state, E, sigma_e, E_chain, grads = compute_step(
            state=state,
            batch=batch,
            hamiltonian=hamiltonian,
            n_chains=n_chains,
            use_qgt=training_config.use_qgt,
            qgt_config=qgt_config,
            auxiliary_losses=_auxiliary_losses,
            key=lap_key,
        )

        # Naive MC error of the mean; upgraded to σ_E·sqrt(τ_int/M) once IAT lands (step 4).
        error_of_mean = sigma_e / jnp.sqrt(batch.shape[0])

        return (
            new_state,
            key,
            new_pos,
            E,
            sigma_e,
            E_chain,
            error_of_mean,
            acceptance_rate,
            step_size,
            grads,
            cm_mean_val,
            cm_std_val,
        )

    # Build callback list: always NaN guard, then user-supplied, then built-ins from config
    _callbacks = list(callbacks or [])
    _callbacks.insert(0, NaNCallback(training_config.checkpoint_path))
    # Parameter retrieval (step 8): reuse a user-supplied SnapshotCallback if present, else
    # auto-add a best_k policy so result.best_params()/best_k_params() work out of the box.
    snapshot_cb = next((cb for cb in _callbacks if isinstance(cb, SnapshotCallback)), None)
    if snapshot_cb is None and k_best > 0:
        snapshot_cb = SnapshotCallback(policy="best_k", k=k_best, metric=select)
        _callbacks.append(snapshot_cb)
    if training_config.save_checkpoints:
        _callbacks.append(CheckpointCallback(training_config.checkpoint_path))
    if not any(isinstance(cb, RunOutputCallback) for cb in _callbacks):
        if training_config.checkpoint_path == "./":
            warnings.warn(
                "checkpoint_path is './' (the default) — run outputs will be saved in the "
                "current working directory. Set TrainingConfig.checkpoint_path to a named "
                "run directory, or pass a RunOutputCallback with an explicit path.",
                UserWarning,
                stacklevel=2,
            )
        _callbacks.append(RunOutputCallback(n=1, path=training_config.checkpoint_path))

    metrics_history = MetricsHistory()

    # Signal handler — no global state, restores original handler on exit
    _stop = [False]

    def _handle_sigint(signum, frame):
        _stop[0] = True
        print("\nSignal received, stopping after current step.")

    original_handler = signal.signal(signal.SIGINT, _handle_sigint)

    init_steps = int(state.step)
    progress_bar = tqdm(
        range(init_steps, training_config.n_epochs), disable=not tqdm_available
    )
    if tqdm_available:
        _callbacks.append(ProgressCallback(progress_bar))

    try:
        for step in progress_bar:
            if _stop[0]:
                break

            t0 = time.perf_counter()
            (
                new_state,
                key,
                current_positions,
                E,
                sigma_e,
                E_chain,
                error_of_mean,
                acceptance_rate,
                step_size,  # kept on-device: threaded back into the next full_update
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

            # Optimisation diagnostics (§4), computed *outside* the jitted full_update so its
            # graph — and the energy trace — stay bit-identical. Logging only.
            grad_norm = optax.tree.norm(grads)
            old_flat = ravel_pytree(state.params)[0]  # state is still pre-update here
            new_flat = ravel_pytree(new_state.params)[0]
            theta_ratio = jnp.linalg.norm(new_flat - old_flat) / (
                jnp.linalg.norm(old_flat) + 1e-8
            )

            state = new_state

            # One host sync per epoch (instead of ~6 float() calls) — also the §10.2
            # multi-GPU constraint. step_size stays on-device for threading; step_size_v
            # is its host copy for logging.
            (
                E_v,
                sigma_e_v,
                error_of_mean_v,
                E_chain_v,
                acceptance_rate_v,
                cm_mean_v,
                cm_std_v,
                step_size_v,
                grad_norm_v,
                theta_ratio_v,
            ) = jax.device_get(
                (
                    E,
                    sigma_e,
                    error_of_mean,
                    E_chain,
                    acceptance_rate,
                    cm_mean_single,
                    cm_std_single,
                    step_size,
                    grad_norm,
                    theta_ratio,
                )
            )
            dt = time.perf_counter() - t0

            metrics = {
                "step": step,
                "energy": float(E_v),
                "std": float(sigma_e_v),
                "error_of_mean": float(error_of_mean_v),
                "E_chain": E_chain_v,
                "acceptance_rate": acceptance_rate_v,
                "step_size": float(step_size_v),
                "grad_norm": float(grad_norm_v),
                "theta_ratio": float(theta_ratio_v),
                "cm_mean": float(cm_mean_v),
                "cm_std": float(cm_std_v),
                "wall_time": dt,
            }
            metrics_history.append(metrics)
            # Param retrieval: the SnapshotCallback (best_k by `select`) keeps the k best params
            # off-device; exposed post-run via result.best_params()/best_k_params(). No per-epoch
            # device copy in this hot loop — only the snapshot policy and one final device_get.
            if any(cb.on_step_end(step, state, metrics) for cb in _callbacks):
                break

    finally:
        signal.signal(signal.SIGINT, original_handler)
        for cb in _callbacks:
            cb.on_train_end(state, metrics_history)

    # One host copy of the final params (no per-epoch cost); best-k come from the snapshot policy.
    final_params = jax.device_get(state.params)
    snapshots = snapshot_cb.snapshots if snapshot_cb is not None else []
    return TrainResult(history=metrics_history, final_params=final_params, snapshots=snapshots)
