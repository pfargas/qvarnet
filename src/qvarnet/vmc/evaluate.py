"""Frozen-parameter evaluation: pure MC measurement of a trained wavefunction.

Training histories average over a *moving* distribution — walkers sampled while the
parameters were still being optimised. Paper numbers come from here instead: freeze the
parameters, sample |ψ|² with no gradients, and report a block-averaged energy whose
error bar honestly accounts for chain autocorrelation.

Notebook flow (params straight from the TrainResult in memory):

    result = train(...)
    ev = evaluate_result(result, model=model, hamiltonian=ham, shape=(1024, dof),
                         sampling_config=cfg, sample_factor=2.0)   # 2x training samples
    print(ev)

CLI / artifacts flow (params reloaded from a run dir's best_params.msgpack):

    params = artifacts.load_params(f"{run_dir}/best_params.msgpack")["params"][0]
    ev = evaluate(model, params, hamiltonian, shape=(1024, dof),
                  sampling_config=cfg, n_epochs=400)

Error bars use plain fixed-count block averaging over the per-epoch energy series
(deliberately textbook — auditable in ten lines): split the series into ``n_blocks``
contiguous blocks; the scatter of block means estimates the true error including
autocorrelation, provided blocks are longer than the correlation time. Compare
``error`` to ``error_naive`` — a large ratio means strong autocorrelation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from ..config.coord_mode import LabCoords
from ..samplers import sample_and_process
from .probability import build_prob_fn


@dataclass(frozen=True)
class EvalResult:
    """Outcome of one frozen-parameter measurement run."""

    energy: float          # mean of the per-epoch energy means
    error: float           # block-averaged error of the mean (use this one)
    error_naive: float     # σ_E/√(total samples) — ignores autocorrelation
    sigma: float           # per-sample spread sqrt(Var(E_loc)), tail mean
    acceptance: float
    n_epochs: int
    n_samples: int         # total kept samples = n_epochs · n_chains · n_eff
    n_blocks: int
    energies: np.ndarray = field(repr=False)   # the per-epoch series that was blocked

    def __str__(self):
        corr = (self.error / self.error_naive) if self.error_naive > 0 else float("nan")
        return (f"E = {self.energy:.6f} ± {self.error:.2e}   (naive ± {self.error_naive:.2e}, "
                f"ratio {corr:.1f})   σ_E = {self.sigma:.4f}   acc = {self.acceptance:.3f}   "
                f"[{self.n_samples} samples / {self.n_epochs} epochs / {self.n_blocks} blocks]")


def block_error(series, n_blocks: int = 20) -> float:
    """Error of the mean of a (possibly autocorrelated) series via block averaging.

    Splits the series into ``n_blocks`` contiguous blocks (trimming the remainder) and
    returns std(block means)/√n_blocks. Valid when each block is ≫ the autocorrelation
    time; for an uncorrelated series it reproduces the naive std/√n estimate.
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 2 * n_blocks:
        n_blocks = max(2, len(series) // 2)
    per = len(series) // n_blocks
    blocks = series[: per * n_blocks].reshape(n_blocks, per).mean(axis=1)
    return float(blocks.std(ddof=1) / np.sqrt(n_blocks))


def evaluate(
    model,
    params,
    hamiltonian,
    shape,
    sampling_config,
    *,
    n_epochs: int,
    coord_mode=None,
    rng_seed: int = 0,
    step_size: float | None = None,
    init_positions: str = "normal",
    burn_in_epochs: int | None = None,
    n_blocks: int = 20,
    progress=None,
) -> EvalResult:
    """Measure ⟨E⟩ of ``model`` at fixed ``params`` — no training, honest error bars.

    shape:           (n_chains, dof), like train().
    n_epochs:        sampling epochs *kept* for the estimate (after burn-in).
    step_size:       MH step; pass the trained run's final step size for the right
                     acceptance (``evaluate_result`` does this automatically).
    burn_in_epochs:  discarded equilibration epochs (default: 10% of n_epochs, ≥ 1).
    n_blocks:        blocks for the block-averaged error bar.
    progress:        optional callable(epoch_index) for external progress reporting.
    """
    if sampling_config.sampler == "pt":
        raise NotImplementedError("evaluate() supports the plain MH sampler only (for now)")

    coord_mode = coord_mode or LabCoords()
    hamiltonian = hamiltonian.replace(coord_mode=coord_mode)
    n_chains, dof = shape
    effective_apply = coord_mode.wrap_model_apply(model.apply)
    prob_fn = build_prob_fn(effective_apply)

    key = random.PRNGKey(rng_seed)
    if init_positions == "uniform":
        if not sampling_config.box_L:
            raise ValueError("init_positions='uniform' requires sampling_config.box_L")
        positions = jax.random.uniform(key, shape) * sampling_config.box_L
    elif init_positions == "zeros":
        positions = jnp.zeros(shape)
    else:
        positions = jax.random.normal(key, shape) * 0.5

    step = step_size if step_size is not None else sampling_config.step_size
    burn_in = burn_in_epochs if burn_in_epochs is not None else max(1, n_epochs // 10)
    n_eff = max(1, (sampling_config.chain_length - sampling_config.thermalization_steps)
                // sampling_config.thinning_factor)

    @partial(jax.jit, static_argnames=["prob_fn", "hamiltonian", "sampling_config"])
    def eval_step(params, key, current_pos, prob_fn, hamiltonian, sampling_config, step_size):
        key, subkey, lap_key = jax.random.split(key, 3)
        batch, new_pos, acceptance = sample_and_process(
            key=subkey,
            prob_fn=prob_fn,
            prob_params=params,
            init_positions=current_pos,
            step_size=step_size,
            n_chains=n_chains,
            dof=dof,
            n_steps=sampling_config.chain_length,
            burn_in=sampling_config.thermalization_steps,
            thinning=sampling_config.thinning_factor,
            proposal=sampling_config.proposal,
            box_L=sampling_config.box_L or 0.0,
        )
        e_loc = hamiltonian.local_energy(params, batch, effective_apply, key=lap_key)
        return key, new_pos, jnp.mean(e_loc), jnp.std(e_loc), jnp.mean(acceptance)

    e_means, e_stds, accs = [], [], []
    for epoch in range(burn_in + n_epochs):
        key, positions, e_mean, e_std, acc = eval_step(
            params, key, positions, prob_fn, hamiltonian, sampling_config, step)
        if epoch >= burn_in:
            e_means.append(float(e_mean))
            e_stds.append(float(e_std))
            accs.append(float(acc))
        if progress is not None:
            progress(epoch)

    e_means = np.asarray(e_means)
    n_samples = n_epochs * n_chains * n_eff
    sigma = float(np.mean(e_stds))
    return EvalResult(
        energy=float(e_means.mean()),
        error=block_error(e_means, n_blocks=n_blocks),
        error_naive=sigma / math.sqrt(n_samples),
        sigma=sigma,
        acceptance=float(np.mean(accs)),
        n_epochs=n_epochs,
        n_samples=n_samples,
        n_blocks=min(n_blocks, max(2, len(e_means) // 2)),
        energies=e_means,
    )


def evaluate_result(
    result,
    *,
    model,
    hamiltonian,
    shape,
    sampling_config,
    sample_factor: float = 1.0,
    snapshot_index: int = 0,
    **kwargs,
) -> EvalResult:
    """Evaluate a :class:`TrainResult`'s best retained model (the notebook flow).

    Uses snapshot ``snapshot_index`` of ``result.best_k()`` (0 = best by the training
    ``select`` metric; falls back to the final params if no snapshots were kept), a
    sampling budget of ``sample_factor`` × the training epochs, and — unless overridden —
    the last training step size, so the MH acceptance carries over.
    """
    ranked = result.best_k()
    params = ranked[snapshot_index]["params"] if ranked else result.final_params
    n_epochs = max(1, math.ceil(sample_factor * len(result.history)))
    if "step_size" not in kwargs and len(result.history):
        kwargs["step_size"] = float(result.history.get("step_size")[-1])
    return evaluate(model, params, hamiltonian, shape, sampling_config,
                    n_epochs=n_epochs, **kwargs)
