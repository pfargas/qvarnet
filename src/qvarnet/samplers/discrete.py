"""Single-spin-flip Metropolis sampler for discrete (spin) systems (roadmap step 3).

Mirrors the continuous ``samplers/step.py`` but the proposal flips one random site instead
of a Gaussian move; acceptance is the same log-space rule A = min(1, |ψ(s')/ψ(s)|²) via the
shared ``prob_fn`` (= 2·log|ψ|). Reuses the ``lax.scan`` chain structure of ``kernel.py``.
"""

from functools import partial

import jax
from jax import numpy as jnp
from jax import random


@partial(jax.jit, static_argnames=("prob_fn",))
def spin_flip_kernel(step_rand, prob_fn, prob_params, position, prob):
    """One single-spin-flip MH step.

    Args:
        step_rand: ``(2,)`` — ``[site_draw, accept_draw]`` uniform in [0, 1).
        prob_fn:   ``(s, params) -> 2·log|ψ(s)|``.
        position:  current spin config, shape ``(N,)`` of ±1.
        prob:      current log-prob at ``position``.
    """
    n = position.shape[0]
    site = jnp.clip((step_rand[0] * n).astype(jnp.int32), 0, n - 1)
    proposal = position.at[site].multiply(-1.0)
    proposal_log_prob = prob_fn(proposal, prob_params)
    accept = jnp.log(step_rand[1]) < jnp.minimum(0.0, proposal_log_prob - prob)
    new_position = jnp.where(accept, proposal, position)
    new_log_prob = jnp.where(accept, proposal_log_prob, prob)
    return new_position, new_log_prob, accept


@partial(jax.jit, static_argnames=("prob_fn",))
def spin_flip_chain(random_values, prob_fn, prob_params, init_position):
    """Run one chain of single-spin-flip steps. random_values: ``(n_steps, 2)``."""
    init_prob = prob_fn(init_position, prob_params)

    def body(carry, step_rand):
        position, prob, count = carry
        new_position, new_prob, accepted = spin_flip_kernel(
            step_rand, prob_fn, prob_params, position, prob
        )
        return (new_position, new_prob, count + accepted), new_position

    (_, _, counts), positions = jax.lax.scan(body, (init_position, init_prob, 0), random_values)
    return positions, counts / random_values.shape[0]


@partial(
    jax.jit,
    static_argnames=("prob_fn", "n_chains", "n_spins", "n_steps", "burn_in", "thinning"),
)
def sample_spins(
    key,
    prob_fn,
    prob_params,
    init_positions,
    n_chains,
    n_spins,
    n_steps,
    burn_in,
    thinning,
):
    """Generate one batch of spin configs from parallel single-flip chains.

    Returns ``(batch, last_positions, acceptance_rates)`` with
    ``batch`` of shape ``(n_chains * n_eff, n_spins)`` (chain-contiguous, so a reshape to
    ``(n_chains, n_eff)`` recovers per-chain structure — matches the continuous path).
    """
    rand = random.uniform(key, (n_chains, n_steps, 2))
    sampler = jax.vmap(spin_flip_chain, in_axes=(0, None, None, 0), out_axes=0)
    raw, acceptance = sampler(rand, prob_fn, prob_params, init_positions)  # (n_chains,n_steps,N)
    processed = raw[:, burn_in::thinning, :]  # (n_chains, n_eff, N)
    last_positions = raw[:, -1, :]
    batch = processed.reshape(-1, n_spins)
    return batch, last_positions, acceptance
