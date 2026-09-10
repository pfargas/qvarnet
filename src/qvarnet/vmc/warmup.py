"""Equilibrate the walkers before epoch 0.

Two modes. Plain warmup runs one long chain and keeps the last positions.
Block-adaptive warmup additionally retunes the proposal step between blocks, which
matters for a warm-started run: its converged |psi|^2 typically wants a step an
order of magnitude smaller than the config default, and without retuning the first
epochs sample at a few percent acceptance with near-frozen chains.

Both go through ``ctx.sampler``, so a constrained sampler is honoured here too.
"""

import jax
import jax.numpy as jnp

from .setup import _as_device_scalar


def _draw_positions(ctx, key, step_size, n_steps):
    """Run ``n_steps`` and keep only the final walker positions and acceptance."""
    _, positions, acceptance = ctx.sampler.draw(
        key=key,
        prob_fn=ctx.prob_fn,
        prob_params=ctx.state.params,
        init_positions=ctx.positions,
        step_size=step_size,
        n_chains=ctx.n_chains,
        dof=ctx.dof,
        n_steps=n_steps,
        burn_in=n_steps - 1,
        thinning=1,
        box_L=ctx.box_L,
    )
    return positions, acceptance


def warm_up(ctx) -> None:
    """Equilibrate ``ctx.positions`` in place; may also update ``ctx.step_size``."""
    chain_cfg = ctx.initial_chain_config
    if not (chain_cfg.warmup_steps > 0 and chain_cfg.warmup_starting_positions):
        return

    if not chain_cfg.warmup_adapt_step_size:
        ctx.positions, _ = _draw_positions(
            ctx, ctx.key, chain_cfg.warmup_step_size, chain_cfg.warmup_steps
        )
        return

    train_cfg = ctx.training_config
    n_blocks = min(chain_cfg.warmup_n_blocks, chain_cfg.warmup_steps)
    block_len = chain_cfg.warmup_steps // n_blocks
    warmup_step = chain_cfg.warmup_step_size

    for block in range(n_blocks):
        # fold_in, not split: the run's key must not be advanced here.
        block_key = jax.random.fold_in(ctx.key, block)
        ctx.positions, acceptance = _draw_positions(ctx, block_key, warmup_step, block_len)
        # Proportional retune, clipped per block so one bad block cannot run away.
        factor = float(jnp.mean(acceptance)) / train_cfg.target_acceptance
        warmup_step = float(
            jnp.clip(
                warmup_step * jnp.clip(factor, 0.2, 5.0),
                train_cfg.min_step,
                train_cfg.max_step,
            )
        )

    # Hand the adapted step to the training sampler, which keeps adapting from there.
    # Strongly typed, for the same no-retrace reason as in setup.
    if train_cfg.is_update_step_size:
        ctx.step_size = _as_device_scalar(warmup_step)
