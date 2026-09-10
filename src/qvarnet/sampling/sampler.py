"""Samplers: one object owns a whole batch draw.

A ``Sampler`` is a frozen dataclass -- hashable, so it can be a jit-static
argument, exactly like a ``Proposal``. It provides ``step``, ``chain`` and
``draw``; a subclass overrides ``constrain`` and nothing else.

``constrain`` projects a configuration onto the space the sampler is allowed to
move in. It is applied in exactly two places -- when a chain starts, and to each
proposal after the periodic wrap -- so a constrained sampler never evaluates the
ansatz on a configuration outside its domain. For plain Metropolis it is the
identity.

That hook is the whole extension surface. A sampler that keeps 1-D walkers in
the ordered sector is::

    @dataclass(frozen=True)
    class OrderedMetropolis(Metropolis):
        def constrain(self, x):
            return jnp.sort(x)

which is the entire implementation below. Write one in your own project file and
pass it as ``train(..., sampler=MySampler())``; there is nothing to register.
"""

from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as jnp
from jax import random

from qvarnet.sampling.kernel import GaussianMove, Proposal


@dataclass(frozen=True)
class Sampler:
    """Base sampler. Override ``constrain``; inherit the rest."""

    proposal: Proposal = GaussianMove()

    # -- the extension hook ---------------------------------------------------

    def constrain(self, x):
        """Project a configuration onto the allowed space. Identity by default."""
        return x

    # -- provided: you should not need to override these ----------------------

    def step(self, key, prob_fn, prob_params, position, prob, step_size, box_L):
        """One Metropolis-Hastings step in log-probability space.

        Accepts with probability min(1, exp(log P(x') - log P(x) + log q_corr)),
        where log q_corr is the proposal's Hastings term (0 when symmetric).
        """
        k_prop, k_accept = random.split(key)
        proposed, log_q_corr = self.proposal.propose(k_prop, position, step_size)
        # PBC: fold into [0, L) when box_L > 0; a no-op at box_L == 0.
        wrapped = proposed - box_L * jnp.floor(proposed / jnp.where(box_L > 0, box_L, 1.0))
        proposed = jnp.where(box_L > 0, wrapped, proposed)
        proposed = self.constrain(proposed)
        proposed_log_prob = prob_fn(proposed, prob_params)
        accept_log_prob = jnp.minimum(0.0, proposed_log_prob - prob + log_q_corr)
        accept = jnp.log(random.uniform(k_accept)) < accept_log_prob
        return (
            jnp.where(accept, proposed, position),
            jnp.where(accept, proposed_log_prob, prob),
            accept,
        )

    def chain(self, key, prob_fn, prob_params, init_position, step_size, n_steps, box_L):
        """Run one chain for ``n_steps``. Returns (positions, acceptance_rate).

        Per-step keys are split inside the scan, so nothing scales with n_steps
        except the position history itself.
        """
        init_position = self.constrain(init_position)
        init_prob = prob_fn(init_position, prob_params)

        def body_fn(carry, step_key):
            position, prob, count = carry
            new_position, new_prob, accepted = self.step(
                step_key, prob_fn, prob_params, position, prob, step_size, box_L
            )
            return (new_position, new_prob, count + accepted), (new_position, accepted)

        step_keys = random.split(key, n_steps)
        (_, _, counts), (positions, _) = jax.lax.scan(
            body_fn, (init_position, init_prob, 0), step_keys
        )
        return positions, counts / n_steps

    def draw(
        self,
        key,
        prob_fn,
        prob_params,
        init_positions,
        step_size,
        n_chains,
        dof,
        n_steps,
        burn_in,
        thinning,
        box_L=0.0,
    ):
        """One batch from |psi|^2: vmapped chains, burn-in dropped, thinned, flattened.

        Returns ``(batch (n_chains*n_eff, dof), last_positions (n_chains, dof),
        acceptance_rates (n_chains,))``.
        """
        return _draw(
            self,
            key,
            prob_fn,
            prob_params,
            init_positions,
            step_size,
            n_chains,
            dof,
            n_steps,
            burn_in,
            thinning,
            box_L,
        )


@partial(
    jax.jit,
    static_argnames=("sampler", "prob_fn", "n_chains", "dof", "n_steps", "burn_in", "thinning"),
)
def _draw(
    sampler,
    key,
    prob_fn,
    prob_params,
    init_positions,
    step_size,
    n_chains,
    dof,
    n_steps,
    burn_in,
    thinning,
    box_L,
):
    chain_keys = random.split(key, n_chains)
    raw_batch, acceptance_rates = jax.vmap(
        lambda k, x0: sampler.chain(k, prob_fn, prob_params, x0, step_size, n_steps, box_L)
    )(chain_keys, init_positions)

    processed = raw_batch[:, burn_in::thinning, :]  # (n_chains, n_eff, dof)
    last_positions = raw_batch[:, -1, :]  # (n_chains, dof)
    return processed.reshape(-1, dof), last_positions, acceptance_rates


@dataclass(frozen=True)
class Metropolis(Sampler):
    """Standard Metropolis-Hastings with local moves. The default."""


@dataclass(frozen=True)
class OrderedMetropolis(Metropolis):
    """1-D walkers confined to the ordered sector x0 < x1 < ... < x_{N-1}.

    Sorting the proposal is only a valid MH move because |psi|^2 is permutation
    symmetric: the induced kernel on the ordered wedge is the symmetrised kernel,
    which is symmetric, so detailed balance holds and the Hastings term stays 0.
    """

    def constrain(self, x):
        return jnp.sort(x)
