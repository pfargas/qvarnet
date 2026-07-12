"""Named training recipes — validated config bundles for the common workflows.

The configuration surface of :func:`qvarnet.train` (TrainingConfig, SamplingConfig,
ChainInitAndWarmupConfig, QGTConfig, optimizer, ...) exists so every knob has one
owner; these recipes exist so you never have to remember how the knobs combine.
Each returns a dict of ``train()`` keyword arguments to splat::

    from qvarnet.recipes import adam_train, sr_train

    r1 = train(shape=shape, model=model, hamiltonian=ham,
               **adam_train(n_epochs=20_000, learning_rate=1e-2,
                            checkpoint_path="./runs/adam"))
    r2 = train(shape=shape, model=model, hamiltonian=ham,
               **sr_train(n_epochs=1_000, prev_result=r1,
                          checkpoint_path="./runs/sr"))

Both recipes accept ``prev_result`` (a :class:`TrainResult`) to warm-restart: the best
retained parameters, the final walker positions and the adapted MH step size all carry
over, so the rerun resumes sampling where the previous run left off instead of
re-thermalising at the default step size (the near-frozen-chain trap).

The SR recipe encodes the numerically-validated stack: ``solver="auto"`` (minSR when
P > M), Fisher trust region, gradient-norm safety net, block-adaptive warmup. It does
NOT choose your ansatz — for singular interactions (Calogero-Sutherland etc.) SR from
scratch additionally needs a cusp-exact Jastrow init (λ_init = L), or the heavy-tailed
local-energy spikes poison every gradient before the optimizer can act.
"""

import optax

from .config.training_setup import ChainInitAndWarmupConfig, TrainingConfig
from .geometry.qgt import QGTConfig

_DEFAULT_SAMPLER = {
    "step_size": 0.5,
    "chain_length": 21,
    "thermalization_steps": 20,
    "thinning_factor": 1,
}


def _chain_init(prev_result, warmup_steps):
    """Fresh init with block-adaptive warmup, or resume from a previous run."""
    if prev_result is not None and getattr(prev_result, "final_positions", None) is not None:
        # Walkers are already equilibrated — short adaptive warmup re-tunes the step
        # to the (possibly changed) parameters without re-thermalising.
        return ChainInitAndWarmupConfig(
            init_positions=prev_result.final_positions,
            warmup_steps=min(warmup_steps, 100),
            warmup_adapt_step_size=True,
            warmup_n_blocks=5,
        )
    return ChainInitAndWarmupConfig(
        init_positions="normal",
        init_position_params={"mean": 0.0, "std": 0.5},
        warmup_steps=warmup_steps,
        warmup_adapt_step_size=True,
        warmup_n_blocks=10,
    )


def _base_kwargs(prev_result, sampler_params, warmup_steps):
    sampler = dict(_DEFAULT_SAMPLER, **(sampler_params or {}))
    init_params = None
    if prev_result is not None:
        init_params = prev_result.best_params()
        if getattr(prev_result, "final_step_size", None):
            sampler.setdefault("step_size_from_prev", True)
            sampler["step_size"] = prev_result.final_step_size
    return {
        "sampler_params": {k: v for k, v in sampler.items() if k != "step_size_from_prev"},
        "initial_chain_config": _chain_init(prev_result, warmup_steps),
        "init_params": init_params,
    }


def adam_train(
    *,
    n_epochs: int,
    learning_rate: float = 1e-2,
    checkpoint_path: str = "./",
    prev_result=None,
    seed: int = 0,
    sampler_params: dict = None,
    warmup_steps: int = 300,
):
    """Adam training (from scratch, or continued from ``prev_result``).

    The robust default: Adam's per-parameter normalisation bounds every update by
    ~learning_rate regardless of gradient magnitude, so it survives the heavy-tailed
    local-energy spikes that kill plain SGD. Use it for the exploratory phase; switch
    to :func:`sr_train` for the convergence phase.
    """
    kwargs = _base_kwargs(prev_result, sampler_params, warmup_steps)
    kwargs["optimizer"] = optax.adam(learning_rate)
    kwargs["training_config"] = TrainingConfig(
        n_epochs=n_epochs,
        rng_seed=seed,
        warm_walkers=True,
        is_update_step_size=True,
        checkpoint_path=checkpoint_path,
    )
    return kwargs


def sr_train(
    *,
    n_epochs: int,
    learning_rate: float = 1e-3,
    checkpoint_path: str = "./",
    prev_result=None,
    max_state_change: float = 0.1,
    regularization: float = 1e-2,
    grad_clip_norm: float = None,
    solver: str = "auto",
    seed: int = 0,
    sampler_params: dict = None,
    warmup_steps: int = 300,
):
    """Stochastic reconfiguration (natural gradient), stabilised.

    θ ← θ − η·S⁻¹∇E with the validated guard stack: "auto" solver (minSR in the
    P > M regime — same regularised step, solved full-rank in sample space) and a
    Fisher-metric trust region. SR is a *preconditioner*; the update rule is the
    optimizer this recipe sets: optax.sgd(learning_rate), i.e. classic SR. train()
    honours whatever optimizer it receives, so overriding kwargs["optimizer"] (e.g.
    with Adam) gives SR-preconditioned Adam — then keep qgt_config.learning_rate in
    sync or set trust_region explicitly (see QGTConfig).

    ``grad_clip_norm`` defaults to OFF: a Euclidean clip re-throttles the natural
    gradient (whose Euclidean norm is legitimately huge along the model's flat
    directions) and SR stops descending — the 2026-07-11 guard-binding probe measured
    clip 10 binding 100% of epochs with the energy flat, vs Adam-matching descent
    with a 3× cleaner tail once removed. The trust region alone carries the spike
    protection (0 failed solves, 0 NaNs across all probe stages).

    ``max_state_change`` is the physical trust-region knob: the maximum wavefunction
    change per step in the Fisher metric, √(ΔθᵀSΔθ) ≤ max_state_change, whatever the
    estimator claims — the same thing at any η (QGTConfig derives the direction cap).
    0.1 is the validated default: big enough to move, small enough to ride out
    cusp-residual spike epochs. 0.3 descended ~3× faster on CS N=30 (2026-07-11
    probe 2: tail 726.55±0.24 vs exact 726 in 2000 finetune epochs, trust binding
    only 31% of early epochs, zero failed solves) — worth trying when spikes are mild.
    """
    kwargs = _base_kwargs(prev_result, sampler_params, warmup_steps)
    kwargs["optimizer"] = optax.sgd(learning_rate)  # the SR update rule: θ ← θ − η·δ
    kwargs["training_config"] = TrainingConfig(
        n_epochs=n_epochs,
        rng_seed=seed,
        warm_walkers=True,
        is_update_step_size=True,
        use_qgt=True,
        checkpoint_path=checkpoint_path,
    )
    kwargs["qgt_config"] = QGTConfig(
        solver=solver,
        learning_rate=learning_rate,
        regularization=regularization,
        max_state_change=max_state_change,
        grad_clip_norm=grad_clip_norm,
    )
    return kwargs
