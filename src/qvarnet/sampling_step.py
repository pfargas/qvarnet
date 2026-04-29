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

    Wraps a single-chain MH kernel with :func:`jax.vmap` to parallelise
    sampling across all chains simultaneously.

    Args:
        mh_chain: Single-chain MH kernel with signature
            ``(random_values, PBC, prob_fn, prob_params, init_position,
            step_size, is_log_prob) -> (positions, acceptance_rate)``.

    Returns:
        sampler_fn: Vectorised function that samples all chains in parallel.
            Expects ``random_values`` of shape ``(n_chains, n_steps, DoF+1)``.
    """
    sampler_fn = jax.vmap(
        mh_chain,
        in_axes=(
            0,  # random_values: vectorize over chains (axis 0)
            None,  # PBC: same for all chains
            None,  # prob_fn: same function for all chains
            None,  # prob_params: same parameters for all chains
            0,  # init_position: different position per chain
            None,  # step_size: same for all chains
            None,  # is_log_prob: same for all chains
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
        samples: Flattened batch of shape ``(n_chains * n_effective, DoF)``, where ``n_effective = (n_steps - burn_in) // thinning``.
        last_positions: Final walker positions, shape ``(n_chains, DoF)``.
        acceptance_rates: Per-chain acceptance rate, shape ``(n_chains,)``.

    Note:
        All operations are JIT-compiled. The vmap over chains is handled inside
        :func:`create_sampler_fn`. Samples remain in log-space when
        ``is_log_prob=True``.
    """
    from .samplers import mh_chain as mh_chain_fn

    # Generate random numbers for all chains
    # Shape: (n_chains, n_steps, DoF+1)
    # The extra dimension is used for accept/reject decision in MH kernel
    if uniform:
        rand_nums = random.uniform(key, (n_chains, n_steps, DoF + 1))
    else:
        uniform_key, normal_key = random.split(key)
        rand_nums_normal = random.normal(normal_key, (n_chains, n_steps, DoF))
        rand_nums_uniform = random.uniform(uniform_key, (n_chains, n_steps, 1))
        rand_nums = jnp.concatenate([rand_nums_normal, rand_nums_uniform], axis=-1)

    # Create vectorized sampler
    sampler_fn = create_sampler_fn(mh_chain_fn)

    # Run all chains in parallel
    # raw_batch shape: (n_chains, n_steps, DoF)
    # acceptance_rates shape: (n_chains,)
    # Note: argument order must match mh_chain signature
    raw_batch, acceptance_rates = sampler_fn(
        rand_nums,  # random_values
        PBC,  # PBC
        prob_fn,  # prob_fn
        prob_params,  # prob_params
        init_positions,  # init_position
        step_size,  # step_size
        is_log_prob,  # is_log_prob
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


def batched_sample_and_process(
    key: jax.random.PRNGKey,
    prob_fn: Callable,
    prob_params,
    DoF: int,
    n_samples: int,
    step_size: float,
    n_steps: int,
    burn_in: int,
    thinning: int,
    PBC: float,
    is_log_prob: bool,
    chunk_size: int = 5_000,
) -> jnp.ndarray:
    """
    Memory-safe wrapper around :func:`sample_and_process` that splits sampling
    into chunks of ``chunk_size`` chains.

    The random-number matrix inside :func:`sample_and_process` has shape
    ``(n_chains, n_steps, DoF+1)``.  For large ``n_samples`` this can exceed
    available memory; running smaller independent chunks keeps peak allocation
    bounded at ``(chunk_size, n_steps, DoF+1)`` per call.

    Because ``n_chains`` is a ``static_argname`` in the underlying JIT, every
    full chunk reuses the same compiled kernel.  A remainder chunk
    (``n_samples % chunk_size != 0``) is compiled once separately and cached.

    Args:
        key: JAX random key.
        prob_fn: Probability function ``(x, params) -> R``.
        prob_params: Model parameters passed to ``prob_fn``.
        DoF: Degrees of freedom per sample.
        n_samples: Total number of samples to collect.
        step_size: MH proposal step size.
        n_steps: Total chain length (including burn-in).
        burn_in: Number of initial steps to discard.
        thinning: Keep every ``thinning``-th post-burn-in step.
        PBC: Periodic boundary size (0 for none).
        is_log_prob: Whether ``prob_fn`` returns log-probability.
        chunk_size: Number of chains per call. Defaults to 5 000.

    Returns:
        samples: Concatenated samples, shape ``(n_samples_eff, DoF)`` where
            ``n_samples_eff = n_samples * (n_steps - burn_in) // thinning``.
    """
    n_full_chunks = n_samples // chunk_size
    remainder     = n_samples  % chunk_size

    def _chunk(key, n_chains):
        samples, _, _ = sample_and_process(
            key=key,
            prob_fn=prob_fn,
            prob_params=prob_params,
            init_positions=jnp.zeros((n_chains, DoF)),
            step_size=step_size,
            n_chains=n_chains,
            DoF=DoF,
            n_steps=n_steps,
            burn_in=burn_in,
            thinning=thinning,
            PBC=PBC,
            is_log_prob=is_log_prob,
        )
        return samples

    all_samples = []
    for _ in range(n_full_chunks):
        key, subkey = random.split(key)
        all_samples.append(_chunk(subkey, chunk_size))

    if remainder > 0:
        key, subkey = random.split(key)
        all_samples.append(_chunk(subkey, remainder))

    return jnp.concatenate(all_samples, axis=0)
