"""Metropolis-Hastings proposal families and the single-step kernel.

A Proposal answers one question -- how a new configuration is suggested::

    propose(key, position, step_size) -> (proposal, log_q_correction)

``log_q_correction`` is the Hastings term log q(x|x') - log q(x'|x). It is 0 for
every symmetric family here; it exists so asymmetric proposals (MALA, ...) plug in
without touching the kernel.

Coordinate layout is particle-major: ``position.reshape(n_particles, n_dim)``.

Which family to use, and why subset moves win for N >~ 30:
docs/explainers/samplers.md.
"""

from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as jnp
from jax import random


@dataclass(frozen=True)
class Proposal:
    """Base class for MH proposal families (frozen ⇒ hashable ⇒ jit-static)."""

    def propose(self, key, position, step_size):
        raise NotImplementedError


@dataclass(frozen=True)
class GaussianMove(Proposal):
    """x' = x + step_size · N(0, 1) on every coordinate (the classic default)."""

    def propose(self, key, position, step_size):
        return position + step_size * random.normal(key, position.shape), 0.0


@dataclass(frozen=True)
class UniformMove(Proposal):
    """x' = x + step_size · U(−1, 1) on every coordinate."""

    def propose(self, key, position, step_size):
        noise = random.uniform(key, position.shape, minval=-1.0, maxval=1.0)
        return position + step_size * noise, 0.0


@dataclass(frozen=True)
class ParticleSubsetMove(Proposal):
    """Gaussian-move all coordinates of ``n_move`` uniformly chosen particles.

    ``n_particles`` is inferred from the position size as ``dof // n_dim``.
    """

    n_move: int
    n_dim: int = 1

    def propose(self, key, position, step_size):
        dof = position.shape[-1]
        n_particles = dof // self.n_dim
        if self.n_move > n_particles:
            raise ValueError(
                f"ParticleSubsetMove: n_move ({self.n_move}) exceeds the "
                f"{n_particles} particles implied by dof={dof}, n_dim={self.n_dim}"
            )
        k_pick, k_noise = random.split(key)
        # n_move is static: the branch resolves at trace time. randint is ~2× cheaper
        # than a full permutation per step, and n_move=1 is the recommended setting.
        if self.n_move == 1:
            picked = random.randint(k_pick, (1,), 0, n_particles)
        else:
            picked = random.permutation(k_pick, n_particles)[: self.n_move]
        particle_mask = jnp.zeros(n_particles).at[picked].set(1.0)
        mask = jnp.repeat(particle_mask, self.n_dim)  # particle-major layout
        noise = random.normal(k_noise, position.shape)
        return position + step_size * noise * mask, 0.0


@dataclass(frozen=True)
class DoFSubsetMove(Proposal):
    """Gaussian-move ``k`` uniformly chosen coordinates (particle-agnostic)."""

    k: int

    def propose(self, key, position, step_size):
        dof = position.shape[-1]
        if self.k > dof:
            raise ValueError(f"DoFSubsetMove: k ({self.k}) exceeds dof ({dof})")
        k_pick, k_noise = random.split(key)
        if self.k == 1:  # static branch; see ParticleSubsetMove
            picked = random.randint(k_pick, (1,), 0, dof)
        else:
            picked = random.permutation(k_pick, dof)[: self.k]
        mask = jnp.zeros(dof).at[picked].set(1.0)
        noise = random.normal(k_noise, position.shape)
        return position + step_size * noise * mask, 0.0


@partial(jax.jit, static_argnames=("prob_fn", "proposal"))
def mh_kernel_log(
    key,
    prob_fn,
    prob_params,
    position,
    prob,
    step_size,
    proposal: Proposal = GaussianMove(),
    box_L=0.0,
    beta=1.0,
):
    """One Metropolis-Hastings step in log-probability space.

    Accepts with probability min(1, exp(beta*(log P(x') - log P(x)) + log q_corr)).

    Args:
        key: PRNG key for this step (proposal noise and the accept draw).
        prob_fn: ``(x, params) -> log P(x)``, log-unnormalised.
        prob_params: parameters passed to prob_fn.
        position: current configuration, shape ``(dof,)``.
        prob: current (untempered) log P(x).
        step_size: proposal step scale.
        proposal: proposal family (jit-static frozen dataclass).
        box_L: periodic box side; > 0 wraps proposals into [0, L). Traced, not static.
        beta: inverse temperature multiplying the log-prob ratio (traced).

    Returns:
        ``(new_position, new_log_prob, accept)``.
    """
    k_prop, k_accept = random.split(key)
    proposed, log_q_corr = proposal.propose(k_prop, position, step_size)
    # PBC sampler: fold proposal into [0, L) when box_L > 0 (no-op when box_L == 0).
    wrapped = proposed - box_L * jnp.floor(proposed / jnp.where(box_L > 0, box_L, 1.0))
    proposed = jnp.where(box_L > 0, wrapped, proposed)
    proposed_log_prob = prob_fn(proposed, prob_params)
    accept_log_prob = jnp.minimum(0.0, beta * (proposed_log_prob - prob) + log_q_corr)
    accept = jnp.log(random.uniform(k_accept)) < accept_log_prob
    new_position = jnp.where(accept, proposed, position)
    new_log_prob = jnp.where(accept, proposed_log_prob, prob)
    return new_position, new_log_prob, accept


@partial(jax.jit, static_argnames=("prob_fn", "n_steps", "proposal"))
def mh_chain(
    key,
    prob_fn,
    prob_params,
    init_position,
    step_size,
    n_steps,
    proposal: Proposal = GaussianMove(),
    box_L=0.0,
):
    """Run one Metropolis-Hastings chain for ``n_steps`` steps.

    Per-step keys are split inside the scan, so peak memory is the position history
    alone rather than a pre-generated random buffer.

    Args:
        key: PRNG key for the whole chain.
        prob_fn: ``(x, params) -> log P(x)``.
        prob_params: parameters for prob_fn.
        init_position: initial configuration, shape ``(dof,)``.
        step_size: proposal step scale.
        n_steps: number of MH steps (static).
        proposal: proposal family (jit-static).
        box_L: periodic box side; > 0 wraps proposals into [0, L).

    Returns:
        ``(positions (n_steps, dof), acceptance_rate)``.
    """
    init_prob = prob_fn(init_position, prob_params)

    def body_fn(carry, step_key):
        position, prob, count = carry
        new_position, new_prob, accepted = mh_kernel_log(
            key=step_key,
            prob_fn=prob_fn,
            prob_params=prob_params,
            position=position,
            prob=prob,
            step_size=step_size,
            proposal=proposal,
            box_L=box_L,
        )
        return (new_position, new_prob, count + accepted), (new_position, accepted)

    step_keys = random.split(key, n_steps)
    (_, _, counts), (positions, _) = jax.lax.scan(body_fn, (init_position, init_prob, 0), step_keys)
    return positions, counts / n_steps
