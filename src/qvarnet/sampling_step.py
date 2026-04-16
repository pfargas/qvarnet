"""MCMC sampling step for Variational Monte Carlo."""

from functools import partial
from typing import Callable, Tuple

from matplotlib.pylab import uniform
import jax
import jax.numpy as jnp
from jax import random


def create_sampler_fn(
    mh_chain: Callable,
) -> Callable:
    """
    Create a vectorized sampler function over multiple MCMC chains.

    This function wraps the single-chain Metropolis-Hastings kernel with
    jax.vmap to parallelize sampling across multiple chains.

    Args:
        mh_chain: Single-chain MH kernel function with signature:
                  (random_values, prob_fn, prob_params, init_position,
                   step_size, PBC, is_log_prob) -> (positions, acceptance_rate)

    Returns:
        sampler_fn: Vectorized function that samples from all chains in parallel.
                   Expects random_values of shape (n_chains, n_steps, DoF+1)
    """
    sampler_fn = jax.vmap(
        mh_chain,
        in_axes=(
            0,      # random_values: vectorize over chains (axis 0)
            None,   # PBC: same for all chains
            None,   # prob_fn: same function for all chains
            None,   # prob_params: same parameters for all chains
            0,      # init_position: different position per chain
            None,   # step_size: same for all chains
            None,   # is_log_prob: same for all chains
        ),
        out_axes=0,  # Output: result for each chain (axis 0)
    )
    return sampler_fn


@partial(
    jax.jit,
    static_argnames=[
        "prob_fn",
        "n_chains",
        "DoF",
        "n_steps",
        "burn_in",
        "thinning",
        "PBC",
        "is_log_prob",
    ],
)
def sample_and_process(
    key: jax.random.PRNGKey,
    prob_fn: Callable,
    prob_params,
    init_positions: jnp.ndarray,
    step_size: float,
    n_chains: int,
    DoF: int,
    n_steps: int,
    burn_in: int,
    thinning: int,
    PBC: float,
    is_log_prob: bool,
    uniform: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate one batch of samples from MCMC and process them.

    This function:
    1. Generates random numbers for all chains
    2. Runs Metropolis-Hastings chains in parallel
    3. Discards burn-in samples
    4. Applies thinning to reduce autocorrelation
    5. Returns flattened batch and diagnostics

    Args:
        key: JAX random key for reproducibility
        prob_fn: Probability density function (x, params) -> ℝ
        prob_params: Parameters for prob_fn (typically neural network weights)
        init_positions: Starting positions for all chains, shape (n_chains, DoF)
        step_size: Step size for MH proposal distribution
        n_chains: Number of parallel MCMC chains
        DoF: Degrees of freedom per sample
        n_steps: Total number of steps per chain
        burn_in: Number of initial samples to discard (thermalization)
        thinning: Keep every thinning-th sample (reduce autocorrelation)
        PBC: Periodic boundary condition size (0 for no PBC)
        is_log_prob: If True, prob_fn outputs log(P). If False, outputs P.

    Returns:
        samples: Processed batch, shape (n_chains * n_samples_effective, DoF)
                where n_samples_effective = (n_steps - burn_in) // thinning
        last_positions: Final positions of all chains, shape (n_chains, DoF)
        acceptance_rates: Acceptance rate per chain, shape (n_chains,)

    Notes:
        - All operations are JIT-compiled for speed
        - The vmap over chains is handled internally by create_sampler_fn
        - Samples are in log-space if is_log_prob=True (not exp-transformed)
    """
    from .samplers import mh_chain as mh_chain_fn

    # Generate random numbers for all chains
    # Shape: (n_chains, n_steps, DoF+1)
    # The extra dimension is used for accept/reject decision in MH kernel
    if uniform:
        rand_nums = random.uniform(key, (n_chains, n_steps, DoF + 1))
    else:
        rand_nums_normal = random.normal(key, (n_chains, n_steps, DoF))
        rand_nums_uniform = random.uniform(key, (n_chains, n_steps, 1))
        rand_nums = jnp.concatenate([rand_nums_normal, rand_nums_uniform], axis=-1)


    # Create vectorized sampler
    sampler_fn = create_sampler_fn(mh_chain_fn)

    # Run all chains in parallel
    # raw_batch shape: (n_chains, n_steps, DoF)
    # acceptance_rates shape: (n_chains,)
    # Note: argument order must match mh_chain signature
    raw_batch, acceptance_rates = sampler_fn(
        rand_nums,      # random_values
        PBC,            # PBC
        prob_fn,        # prob_fn
        prob_params,    # prob_params
        init_positions, # init_position
        step_size,      # step_size
        is_log_prob,    # is_log_prob
    )

    # Post-processing: thermalization and thinning
    # Drop first `burn_in` samples, then take every `thinning`-th sample
    processed_batch = raw_batch[:, burn_in::thinning, :]
    # Shape: (n_chains, (n_steps - burn_in) // thinning, DoF)

    # Get final positions (last sample from each chain)
    last_positions = raw_batch[:, -1, :]  # Shape: (n_chains, DoF)

    # Flatten batch: combine all chains and all samples
    batch_flat = processed_batch.reshape(-1, DoF)
    # Shape: (n_chains * n_samples_effective, DoF)

    return batch_flat, last_positions, acceptance_rates
