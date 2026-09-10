"""Named training recipes: validated bundles of the configuration knobs.

The configuration surface exists so every knob has one owner; these exist so you
need not remember how the knobs combine. Each returns a dict of ``train()`` keyword
arguments to splat::

    r1 = train(shape=shape, model=model, hamiltonian=ham,
               **adam_train(n_epochs=20_000, learning_rate=1e-2))
    r2 = train(shape=shape, model=model, hamiltonian=ham,
               **sr_train(n_epochs=1_000, prev_result=r1))

Both accept ``prev_result`` to warm-restart: best parameters, final walker positions
and the adapted step size all carry over, so the rerun resumes where the last left
off instead of re-thermalising at the default step.

``sr_train`` encodes the numerically validated SR stack -- see
docs/explainers/stochastic-reconfiguration.md.
"""

import optax

from qvarnet.optim.qgt import QGTConfig
from qvarnet.vmc.config import ChainInitAndWarmupConfig, TrainingConfig

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
    """Keyword arguments for a stochastic-reconfiguration run.

    Encodes the validated SR stack: solver="auto" (minSR when P > M), the Fisher
    trust region, a gradient-norm safety net and block-adaptive warmup. It does not
    choose your ansatz -- for singular interactions SR from scratch also needs a
    cusp-exact Jastrow init, or heavy-tailed local-energy spikes poison the gradients
    before the optimizer can act. See docs/explainers/stochastic-reconfiguration.md.

    Args:
        n_epochs: training epochs.
        learning_rate: SGD step; SR uses optax.sgd(learning_rate) as the update rule.
        prev_result: a TrainResult to warm-restart from (parameters, walkers, step).
        **overrides: any TrainingConfig / QGTConfig / sampler field.
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
