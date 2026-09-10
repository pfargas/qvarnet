"""Host-side assembly of a VMC run: resolve arguments, build the initial state.

Nothing here is on the hot path -- it runs once, before the first epoch. Order is
load-bearing in one respect: the PRNG key is used (not split) by ``model.init``
and again by the walker initialiser, and is only ever advanced inside the jitted
update. Do not reorder those uses.
"""

import warnings

import jax
import jax.numpy as jnp
import optax
from jax import random

from qvarnet.ansatz.probability import build_prob_fn
from qvarnet.callbacks import (
    CheckpointCallback,
    NaNCallback,
    RunOutputCallback,
    SnapshotCallback,
)
from qvarnet.core.coords import LabCoords
from qvarnet.core.serialization import load_checkpoint
from qvarnet.optim.losses import CuspLoss, make_cusp_configs, make_cusp_pair_indices
from qvarnet.optim.qgt import DEFAULT_QGT_CONFIG, QGTConfig
from qvarnet.physics.boundaries import PeriodicBoundary
from qvarnet.sampling import Metropolis
from qvarnet.sampling.config import SamplingConfig
from qvarnet.vmc.config import ChainInitAndWarmupConfig
from qvarnet.vmc.context import TrainContext
from qvarnet.vmc.run_io import save_run_config
from qvarnet.vmc.state import VMCState


def _is_periodic_ansatz(model) -> bool:
    """Best-effort detection of a PeriodicBoundary transform on the model.

    Only drives the PBC mismatch warning, so a false negative costs nothing.
    """
    for attr in ("transform", "boundary"):
        if isinstance(getattr(model, attr, None), PeriodicBoundary):
            return True
    return False


def _warn_on_pbc_mismatch(model, sampling_config):
    """The periodic-ansatz and PBC-sampler toggles are independent by design.

    A mismatch is legal but almost always a mistake, so warn rather than forbid.
    """
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


def _as_device_scalar(value, dtype=float):
    """A *strongly* typed device scalar.

    The jitted update returns step_size and state.step as concrete arrays, so
    handing it a Python scalar (or a weakly-typed one -- ``jnp.asarray(0.5)`` is
    weak) makes epoch 2 a different signature and retraces the entire update.
    Canonicalising respects jax_enable_x64. See tests/test_no_retrace.py.
    """
    return jnp.asarray(value, dtype=jax.dtypes.canonicalize_dtype(dtype))


def _init_walkers(key, shape, initial_chain_config, sampling_config):
    """Starting walker positions: a named strategy, or an explicit array."""
    params = initial_chain_config.init_position_params or {"mean": 0.0, "std": 0.5}
    spec = initial_chain_config.init_positions

    if not isinstance(spec, str):
        # An explicit (n_chains, dof) array — e.g. result.final_positions of an
        # earlier run, so a rerun resumes from equilibrated walkers.
        positions = jnp.asarray(spec)
        if positions.shape != shape:
            raise ValueError(
                f"init_positions array shape {positions.shape} does not match "
                f"the sampler shape {shape}"
            )
        return positions

    if spec == "normal":
        return random.normal(key, shape) * params.get("std", 0.5) + params.get("mean", 0.0)
    if spec == "zeros":
        return jnp.zeros(shape)
    if spec == "uniform":
        # The right prior for a homogeneous (untrapped) gas: a "normal" speck in a
        # large box leaves walkers unequilibrated for thousands of MH steps.
        if not sampling_config.box_L:
            raise ValueError("init_positions='uniform' requires sampling_config.box_L (a PBC box)")
        return random.uniform(key, shape) * sampling_config.box_L
    raise ValueError(f"Unknown init_positions: {spec!r}")


def _build_auxiliary_losses(training_config, hamiltonian, shape, extra):
    """The cusp loss (when configured) followed by any user-supplied losses."""
    cusp = training_config.cusp
    losses = []
    if cusp is not None:
        n_particles = shape[-1]
        L = cusp.L if cusp.L is not None else getattr(hamiltonian, "L", None)
        if L is None:
            raise ValueError(
                "CuspConfig requires a box size L, but neither CuspConfig.L nor "
                "hamiltonian.L is set. Pass cusp=CuspConfig(L=...) to TrainingConfig."
            )
        configs = make_cusp_configs(
            n_particles=n_particles,
            L=L,
            epsilon=cusp.epsilon,
            n_configs_per_pair=cusp.n_configs_per_pair,
            rng_seed=cusp.rng_seed,
        )
        pair_i, pair_j = make_cusp_pair_indices(
            n_particles=n_particles, n_configs_per_pair=cusp.n_configs_per_pair
        )
        losses.append(
            CuspLoss(
                configs,
                pair_i,
                pair_j,
                alpha=cusp.alpha,
                epsilon=cusp.epsilon,
                n=cusp.n,
                C_n=cusp.C_n,
            )
        )
    losses.extend(extra)
    return tuple(losses)


def _build_callbacks(training_config, user_callbacks, select, k_best):
    """NaN guard first, then the user's, then whatever the config implies.

    Returns ``(callbacks, snapshot_cb)``. Order matters: the loop short-circuits on
    the first callback that asks to stop.
    """
    callbacks = list(user_callbacks or [])
    callbacks.insert(0, NaNCallback(training_config.checkpoint_path))

    # Reuse a user-supplied SnapshotCallback if there is one, else add a best_k
    # policy so result.best_params() works without being asked for.
    snapshot_cb = next((cb for cb in callbacks if isinstance(cb, SnapshotCallback)), None)
    if snapshot_cb is None and k_best > 0:
        snapshot_cb = SnapshotCallback(policy="best_k", k=k_best, metric=select)
        callbacks.append(snapshot_cb)

    if training_config.save_checkpoints:
        callbacks.append(CheckpointCallback(training_config.checkpoint_path))

    if not any(isinstance(cb, RunOutputCallback) for cb in callbacks):
        if training_config.checkpoint_path == "./":
            warnings.warn(
                "checkpoint_path is './' (the default) — run outputs will be saved in the "
                "current working directory. Set TrainingConfig.checkpoint_path to a named "
                "run directory, or pass a RunOutputCallback with an explicit path.",
                UserWarning,
                stacklevel=2,
            )
        callbacks.append(RunOutputCallback(n=1, path=training_config.checkpoint_path))

    return callbacks, snapshot_cb


def build_context(
    *,
    shape,
    model,
    optimizer,
    hamiltonian,
    training_config,
    initial_chain_config=None,
    sampler_params=None,
    coord_mode=None,
    sampler=None,
    model_name=None,
    model_args=None,
    qgt_config=None,
    auxiliary_losses=(),
    callbacks=None,
    select="std",
    k_best=3,
    init_params=None,
) -> TrainContext:
    """Resolve every argument and build the initial TrainContext."""
    coord_mode = coord_mode if coord_mode is not None else LabCoords()
    initial_chain_config = initial_chain_config or ChainInitAndWarmupConfig()
    sampler = sampler if sampler is not None else Metropolis()
    if sampler_params is None:
        sampler_params = {}
    if qgt_config is None:
        qgt_config = DEFAULT_QGT_CONFIG
    elif isinstance(qgt_config, dict):
        qgt_config = QGTConfig(**qgt_config)

    hamiltonian = hamiltonian.replace(coord_mode=coord_mode)

    # SR is a gradient *preconditioner*: compute_step hands the natural gradient
    # S⁻¹∇E to the optimizer, so the passed optimizer stays the update rule —
    # optax.sgd(η) gives classic SR, optax.adam gives SR-preconditioned Adam.
    # See docs/adr/ for the trust-region caveat under adaptive optimizers.
    if training_config.use_qgt and qgt_config.grad_clip_norm is not None:
        optimizer = optax.chain(optax.clip_by_global_norm(qgt_config.grad_clip_norm), optimizer)

    assert len(shape) == 2, f"shape must be (n_chains, dof), got {shape}"
    n_chains, dof = shape
    assert n_chains > 0 and dof > 0, f"shape dimensions must be positive, got {shape}"

    key = random.PRNGKey(training_config.rng_seed)

    params = model.init(key, jnp.ones(coord_mode.model_input_shape(shape)))
    effective_apply = coord_mode.wrap_model_apply(model.apply)
    state = VMCState.create(apply_fn=effective_apply, params=params, tx=optimizer)
    state = load_checkpoint(
        state, path=training_config.checkpoint_path, filename="checkpoint.msgpack"
    )
    # Warm start from supplied parameters (e.g. an earlier run's best snapshot):
    # new params, but a fresh optimizer state and step=0 — a separate training that
    # merely begins from a good point. A real checkpoint above still wins
    # (resume beats warm-start).
    if init_params is not None:
        state = state.replace(params=init_params)
    # Same reason as step_size: TrainState.create() sets a Python int.
    state = state.replace(step=_as_device_scalar(state.step, int))

    if model_name is not None and model_args is not None:
        save_run_config(
            path=training_config.checkpoint_path,
            model_name=model_name,
            model_args=model_args,
            sample_shape=shape,
            coord_mode=coord_mode,
            training_config=training_config,
        )

    sampling_config = (
        sampler_params
        if isinstance(sampler_params, SamplingConfig)
        else SamplingConfig(**sampler_params)
    )
    _warn_on_pbc_mismatch(model, sampling_config)

    positions = _init_walkers(key, shape, initial_chain_config, sampling_config)
    assert positions.shape == (n_chains, dof), (
        f"init_positions shape mismatch: expected {(n_chains, dof)}, got {positions.shape}"
    )

    cb_list, snapshot_cb = _build_callbacks(training_config, callbacks, select, k_best)

    return TrainContext(
        shape=tuple(shape),
        model=model,
        hamiltonian=hamiltonian,
        coord_mode=coord_mode,
        sampler=sampler,
        prob_fn=build_prob_fn(effective_apply),
        sampling_config=sampling_config,
        training_config=training_config,
        initial_chain_config=initial_chain_config,
        qgt_config=qgt_config,
        auxiliary_losses=_build_auxiliary_losses(
            training_config, hamiltonian, shape, auxiliary_losses
        ),
        state=state,
        key=key,
        positions=positions,
        step_size=_as_device_scalar(sampling_config.step_size),
        callbacks=cb_list,
        snapshot_cb=snapshot_cb,
    )
