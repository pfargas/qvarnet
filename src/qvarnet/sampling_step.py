"""MCMC sampling step for Variational Monte Carlo."""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import random


def create_sampler_fn(
    mh_chain: Callable,
) -> Callable:
    """
    Create a vectorized sampler function over multiple MCMC chains.

    Wraps a single-chain MH kernel with :func:`jax.vmap` to parallelise
    sampling across all chains simultaneously.

    Args:
        mh_chain: Single-chain MH kernel with signature
            ``(random_values, PBC, prob_fn, prob_params, init_position,
            step_size) -> (positions, acceptance_rate)``.

    Returns:
        sampler_fn: Vectorised function that samples all chains in parallel.
            Expects ``random_values`` of shape ``(n_chains, n_steps, dof+1)``.
    """
    sampler_fn = jax.vmap(
        mh_chain,
        in_axes=(
            0,     # random_values: vectorize over chains
            None,  # PBC: same for all chains
            None,  # prob_fn: same function for all chains
            None,  # prob_params: same parameters for all chains
            0,     # init_position: different position per chain
            None,  # step_size: same for all chains
        ),
        out_axes=0,
    )
    return sampler_fn


@partial(
    jax.jit,
    static_argnames=[
        "prob_fn",
        "n_chains",
        "dof",
        "n_steps",
        "burn_in",
        "thinning",
        "PBC",
        "uniform",
    ],
)
def sample_and_process(
    key: jax.random.PRNGKey,
    prob_fn: Callable,
    prob_params,
    init_positions: jnp.ndarray,
    step_size: float,
    n_chains: int,
    dof: int,
    n_steps: int,
    burn_in: int,
    thinning: int,
    PBC: float,
    uniform: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate one batch of samples from MCMC and process them.

    Runs log-space Metropolis-Hastings chains in parallel, discards burn-in,
    applies thinning, and returns a flattened batch.

    Args:
        key: JAX random key.
        prob_fn: Log-probability function ``(x, params) -> log P(x)``.
        prob_params: Parameters for prob_fn.
        init_positions: Starting positions, shape ``(n_chains, dof)``.
        step_size: MH proposal step size.
        n_chains: Number of parallel MCMC chains.
        dof: Degrees of freedom per sample (n_particles * n_dim).
        n_steps: Total steps per chain.
        burn_in: Initial samples to discard (thermalization).
        thinning: Keep every thinning-th sample.
        PBC: Periodic boundary condition size (0 for none).

    Returns:
        samples: shape ``(n_chains * n_effective, dof)``.
        last_positions: shape ``(n_chains, dof)``.
        acceptance_rates: shape ``(n_chains,)``.
    """
    from .samplers import mh_chain as mh_chain_fn

    # Shape: (n_chains, n_steps, dof+1) — last slot is the accept/reject draw
    if uniform:
        rand_nums = random.uniform(key, (n_chains, n_steps, dof + 1))
    else:
        uniform_key, normal_key = random.split(key)
        rand_nums_normal = random.normal(normal_key, (n_chains, n_steps, dof))
        rand_nums_uniform = random.uniform(uniform_key, (n_chains, n_steps, 1))
        rand_nums = jnp.concatenate([rand_nums_normal, rand_nums_uniform], axis=-1)

    sampler_fn = create_sampler_fn(mh_chain_fn)

    # raw_batch: (n_chains, n_steps, dof)
    raw_batch, acceptance_rates = sampler_fn(
        rand_nums,
        PBC,
        prob_fn,
        prob_params,
        init_positions,
        step_size,
    )

    processed_batch = raw_batch[:, burn_in::thinning, :]  # (n_chains, n_effective, dof)
    last_positions = raw_batch[:, -1, :]                   # (n_chains, dof)
    batch_flat = processed_batch.reshape(-1, dof)          # (n_chains * n_effective, dof)

    return batch_flat, last_positions, acceptance_rates
