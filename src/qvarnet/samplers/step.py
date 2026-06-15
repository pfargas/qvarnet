"""MCMC sampling step for Variational Monte Carlo."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import random


def create_sampler_fn(mh_chain: Callable, box_L: float = 0.0) -> Callable:
    """Vectorize a single-chain MH kernel over n_chains via vmap.

    Returns a function that expects random_values of shape
    (n_chains, n_steps, dof+1) and init_positions of shape (n_chains, dof).
    ``box_L`` (the same for every chain) is bound before vmapping so it is not a
    mapped axis.
    """
    chain = partial(mh_chain, box_L=box_L)
    return jax.vmap(
        chain,
        in_axes=(0, None, None, 0, None),
        out_axes=0,
    )


def _gen_rand(key, n_chains, n_steps, dof, uniform):
    """Generate (n_chains, n_steps, dof+1) random numbers."""
    if uniform:
        return random.uniform(key, (n_chains, n_steps, dof + 1))
    u_key, n_key = random.split(key)
    rand_n = random.normal(n_key, (n_chains, n_steps, dof))
    rand_u = random.uniform(u_key, (n_chains, n_steps, 1))
    return jnp.concatenate([rand_n, rand_u], axis=-1)


def _sample_blocked(
    key,
    prob_fn,
    prob_params,
    init_positions,
    step_size,
    n_chains,
    dof,
    n_steps,
    block_size,
    uniform,
    box_L=0.0,
):
    """Run MH chains in blocks of block_size steps.

    Peak random-number memory: O(n_chains * block_size * dof) instead of
    O(n_chains * n_steps * dof).  Positions are still accumulated into a
    (n_chains, n_steps, dof) buffer so that burn_in / thinning can be
    applied identically to the non-blocked path.
    """
    from .kernel import mh_kernel_log

    n_blocks = n_steps // block_size  # static: both are static_argnames

    # Initial log-probs — computed once, threaded through the carry.
    init_log_probs = jax.vmap(prob_fn, in_axes=(0, None))(init_positions, prob_params)

    buf = jnp.zeros((n_chains, n_steps, dof))
    # carry: (positions, log_probs, accept_counts, position_buffer)
    init_carry = (
        init_positions,
        init_log_probs,
        jnp.zeros(n_chains, dtype=jnp.int32),
        buf,
    )

    def run_block(block_idx, carry):
        positions, log_probs, counts, buf = carry

        # Fresh random numbers for this block only — O(B * block_size * dof)
        block_key = jax.random.fold_in(key, block_idx)
        rand_nums = _gen_rand(block_key, n_chains, block_size, dof, uniform)

        # Continue each chain for block_size steps from its current state.
        def run_one_chain(pos, log_prob, count, rand):
            carry0 = (pos, log_prob, count)

            def body(inner_carry, step_rand):
                p, lp, cnt = inner_carry
                new_p, new_lp, accepted = mh_kernel_log(
                    step_rand, prob_fn, prob_params, p, lp, step_size, uniform, box_L
                )
                return (new_p, new_lp, cnt + accepted), new_p

            (new_pos, new_lp, new_cnt), block_pos = jax.lax.scan(body, carry0, rand)
            return new_pos, new_lp, new_cnt, block_pos

        new_pos, new_log_probs, new_counts, block_positions = jax.vmap(
            run_one_chain, in_axes=(0, 0, 0, 0)
        )(positions, log_probs, counts, rand_nums)

        # Write block into the pre-allocated buffer at the correct offset.
        # block_idx is a traced int, so dynamic_update_slice handles it.
        buf = jax.lax.dynamic_update_slice(buf, block_positions, [0, block_idx * block_size, 0])
        return (new_pos, new_log_probs, new_counts, buf)

    _, _, final_counts, all_positions = jax.lax.fori_loop(0, n_blocks, run_block, init_carry)

    acceptance_rates = final_counts.astype(jnp.float32) / n_steps
    return all_positions, acceptance_rates


@partial(
    jax.jit,
    static_argnames=[
        "prob_fn",
        "n_chains",
        "dof",
        "n_steps",
        "burn_in",
        "thinning",
        "uniform",
        "block_size",
    ],
)
def sample_and_process(
    key: jax.Array,
    prob_fn: Callable,
    prob_params,
    init_positions: jnp.ndarray,
    step_size: float,
    n_chains: int,
    dof: int,
    n_steps: int,
    burn_in: int,
    thinning: int,
    uniform: bool = False,
    block_size: int = 0,
    box_L: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate one batch of samples from MCMC and process them.

    Runs log-space Metropolis-Hastings chains in parallel (vmapped), discards
    burn-in, applies thinning, and returns a flattened batch ready for VMC.

    Args:
        key:        JAX random key.
        prob_fn:    Log-probability ``(x, params) -> log P(x)``.
        prob_params: Parameters for prob_fn.
        init_positions: Starting positions, shape ``(n_chains, dof)``.
        step_size:  MH proposal step size.
        n_chains:   Number of parallel MCMC chains.
        dof:        Degrees of freedom per sample.
        n_steps:    Total MH steps per chain.
        burn_in:    Initial samples to discard.
        thinning:   Keep every thinning-th sample after burn-in.
        uniform:    Use uniform proposals instead of Gaussian.
        block_size: If > 0, generate random numbers in blocks of this many
                    steps to cap peak memory at O(n_chains * block_size * dof).
                    Must divide n_steps exactly.  Default 0 = no blocking.
        box_L:      Periodic box side length. > 0 enables the PBC sampler (proposals
                    wrapped into [0, L)); 0 (default) leaves walkers on the covering
                    space. Independent of whether the ansatz is periodic.

    Returns:
        samples:          ``(n_chains * n_effective, dof)``
        last_positions:   ``(n_chains, dof)``
        acceptance_rates: ``(n_chains,)``
    """
    from .kernel import mh_chain as mh_chain_fn

    if block_size > 0:
        if n_steps % block_size != 0:
            raise ValueError(f"block_size ({block_size}) must divide n_steps ({n_steps}) exactly.")
        raw_batch, acceptance_rates = _sample_blocked(
            key,
            prob_fn,
            prob_params,
            init_positions,
            step_size,
            n_chains,
            dof,
            n_steps,
            block_size,
            uniform,
            box_L,
        )
    else:
        rand_nums = _gen_rand(key, n_chains, n_steps, dof, uniform)
        sampler_fn = create_sampler_fn(mh_chain_fn, box_L)
        raw_batch, acceptance_rates = sampler_fn(
            rand_nums,
            prob_fn,
            prob_params,
            init_positions,
            step_size,
        )

    processed = raw_batch[:, burn_in::thinning, :]  # (n_chains, n_effective, dof)
    last_positions = raw_batch[:, -1, :]  # (n_chains, dof)
    batch_flat = processed.reshape(-1, dof)  # (n_chains * n_effective, dof)

    return batch_flat, last_positions, acceptance_rates
