"""Metropolis-Hastings kernel and proposal families.

The kernel is generic over a :class:`Proposal` — a frozen, hashable dataclass passed
as a jit-static argument that owns *how* a new configuration is suggested::

    propose(key, position, step_size) -> (proposal, log_q_correction)

``log_q_correction`` is the Hastings term log q(x|x') − log q(x'|x), added to the
log-acceptance ratio. It is exactly 0 for every symmetric family below; it exists so
asymmetric proposals (MALA, ...) plug into the same kernel without touching it.

Proposal families:

- :class:`GaussianMove` — move every coordinate by ``step_size * N(0,1)``.
- :class:`UniformMove` — move every coordinate by ``step_size * U(-1,1)``.
- :class:`ParticleSubsetMove` — Gaussian-move all ``n_dim`` coordinates of ``n_move``
  randomly chosen particles, the rest untouched.
- :class:`DoFSubsetMove` — Gaussian-move ``k`` randomly chosen coordinates.

Why subset moves: a full-configuration move changes N·d coordinates at once, so its
acceptance decays with N at fixed step (the log-prob change grows like the sum of N
per-particle changes). Moving a few particles keeps acceptance high at a *large* step
for the moved coordinates — better mixing per model evaluation for N ≳ 30.
(A subset move updates fewer coordinates per accepted step, so mixing per *chain step*
is lower; the win is acceptance at large steps. Tune ``step_size`` accordingly.)

Subset selection is uniform over subsets and independent of the current position, and
the coordinate displacement is symmetric — so the total proposal is symmetric and the
Hastings correction is 0.

Coordinate layout: particle-major, ``position.reshape(n_particles, n_dim)`` — the same
convention as the Jacobi transforms, PBC Hamiltonians and fermionic models.
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


_PROPOSALS = {
    "gaussian": GaussianMove,
    "uniform": UniformMove,
    "particle-subset": ParticleSubsetMove,
    "dof-subset": DoFSubsetMove,
}


def resolve_proposal(spec) -> Proposal:
    """Turn a proposal spec into a Proposal instance.

    Accepts a Proposal instance (returned as-is), a name ("gaussian" | "uniform" —
    the parameter-free families), or a ``(name, kwargs)`` pair for the rest, e.g.
    ``("particle-subset", {"n_move": 2, "n_dim": 3})``.
    """
    if isinstance(spec, Proposal):
        return spec
    if isinstance(spec, str):
        try:
            return _PROPOSALS[spec]()
        except KeyError:
            raise ValueError(
                f"Unknown proposal {spec!r}; known: {sorted(_PROPOSALS)}"
            ) from None
        except TypeError:
            raise ValueError(
                f"Proposal {spec!r} needs parameters — pass ({spec!r}, {{...}}) or an instance"
            ) from None
    name, kwargs = spec
    return _PROPOSALS[name](**kwargs)


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
    """Single Metropolis-Hastings step in log-probability space.

    Accepts with probability
    :math:`A = \\min(1, e^{\\beta(\\log P(x') - \\log P(x)) + \\log q_{corr}})`,
    where the Hastings correction ``log_q_corr`` comes from the proposal (0 for the
    symmetric families) and ``beta`` is an inverse temperature (1 = plain MH; the
    parallel-tempering replicas pass their own β and store the *untempered* log P).

    Args:
        key: PRNG key for this step (proposal noise + accept/reject draw).
        prob_fn: Callable ``(x, params) -> log P(x)``, log-unnormalised probability.
        prob_params: Parameters passed to ``prob_fn``.
        position: Current configuration, shape ``(dof,)``.
        prob: Current log-probability :math:`\\log P(x)` (untempered).
        step_size: Proposal step scale.
        proposal: Proposal family (jit-static frozen dataclass).
        box_L: Periodic box side length. ``> 0`` wraps each proposed coordinate into
            ``[0, L)`` (PBC sampler). Symmetric proposals stay symmetric on the torus,
            so detailed balance is unchanged. ``0`` (default) disables wrapping.
            Passed as a traced value, not a static arg.
        beta: Inverse temperature multiplying the log-prob ratio (traced).

    Returns:
        new_position: Accepted or current configuration, shape ``(dof,)``.
        new_log_prob: (Untempered) log-probability at ``new_position``.
        accept: Boolean indicating whether the proposal was accepted.
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
    """Run a single Metropolis-Hastings chain for ``n_steps`` steps.

    Per-step PRNG keys are split inside the scan — no pre-generated random arrays
    (the old ``(n_steps, dof+1)`` layout coupled the proposal family to the RNG
    buffer shape and dominated peak memory for long chains).

    Args:
        key: PRNG key for the whole chain.
        prob_fn: Log-probability function ``(x, params) -> log P(x)``.
        prob_params: Parameters for ``prob_fn``.
        init_position: Initial configuration, shape ``(dof,)``.
        step_size: Proposal step scale.
        n_steps: Number of MH steps (static).
        proposal: Proposal family (jit-static).
        box_L: Periodic box side length; ``> 0`` wraps proposals into ``[0, L)``.

    Returns:
        positions: All sampled positions, shape ``(n_steps, dof)``.
        acceptance_rate: Fraction of accepted proposals over all steps.
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
    (_, _, counts), (positions, _) = jax.lax.scan(
        body_fn, (init_position, init_prob, 0), step_keys
    )
    return positions, counts / n_steps
