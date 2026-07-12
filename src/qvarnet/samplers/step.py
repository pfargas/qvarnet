"""MCMC sampling step for Variational Monte Carlo."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from .kernel import GaussianMove, Proposal, mh_chain


@partial(
    jax.jit,
    static_argnames=[
        "prob_fn",
        "n_chains",
        "dof",
        "n_steps",
        "burn_in",
        "thinning",
        "proposal",
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
    proposal: Proposal = GaussianMove(),
    box_L: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate one batch of samples from MCMC and process them.

    Runs log-space Metropolis-Hastings chains in parallel (vmapped), discards
    burn-in, applies thinning, and returns a flattened batch ready for VMC.
    PRNG keys are split per chain and per step inside the scan — nothing is
    pre-generated, so peak memory is the position history alone.

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
        proposal:   Proposal family (see ``samplers.kernel``): GaussianMove (default),
                    UniformMove, ParticleSubsetMove, DoFSubsetMove. Jit-static.
        box_L:      Periodic box side length. > 0 enables the PBC sampler (proposals
                    wrapped into [0, L)); 0 (default) leaves walkers on the covering
                    space. Independent of whether the ansatz is periodic.

    Returns:
        samples:          ``(n_chains * n_effective, dof)``
        last_positions:   ``(n_chains, dof)``
        acceptance_rates: ``(n_chains,)``
    """
    chain_keys = random.split(key, n_chains)
    raw_batch, acceptance_rates = jax.vmap(
        lambda k, x0: mh_chain(
            k, prob_fn, prob_params, x0, step_size, n_steps, proposal, box_L
        )
    )(chain_keys, init_positions)

    processed = raw_batch[:, burn_in::thinning, :]  # (n_chains, n_effective, dof)
    last_positions = raw_batch[:, -1, :]  # (n_chains, dof)
    batch_flat = processed.reshape(-1, dof)  # (n_chains * n_effective, dof)

    return batch_flat, last_positions, acceptance_rates
