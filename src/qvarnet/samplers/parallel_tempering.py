"""Parallel-tempering (replica-exchange) continuum MCMC — a generic *addition* to the sampler.

Motivation (`comments/MOTT_PHASE_INVESTIGATION.md`): in a deep lattice the local-move MH sampler is
barrier-locked and never reaches the Mott (one-per-well) configuration. Rather than a
Hamiltonian-specific proposal (e.g. lattice hops), this adds the standard *generic* fix that works
for **any** continuum target |ψ|²: run several replicas of the **same local Gaussian MH** at
inverse temperatures β₁=1 > β₂ > … > β_R, sampling |ψ|^{2β}. Hot replicas (small β) see a flattened
landscape and cross barriers freely; periodic adjacent-replica **swaps** carry that mobility down to
the cold β=1 replica, which alone provides the unbiased |ψ|² samples used by VMC.

This is purely an addition: the per-replica update **is**
:func:`samplers.kernel.mh_kernel_log` (shared kernel, called with the replica's β; any
:class:`samplers.kernel.Proposal` family works). No Hamiltonian/lattice knowledge
enters — only ``prob_fn``.

Replica exchange acceptance for an adjacent pair (r, r+1):

    A = min(1, exp[(β_r − β_{r+1}) (logP_{r+1} − logP_r)]),   logP = 2 log|ψ|.

Drop-in: :func:`sample_parallel_tempering` returns ``(batch_flat, last_positions,
acceptance_rates)`` with the same shapes as ``samplers.step.sample_and_process`` (cold replica).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from .kernel import GaussianMove, Proposal, mh_kernel_log


def geometric_betas(n_replicas, beta_min=0.1):
    """Geometric inverse-temperature ladder β = 1 … beta_min (β₁=1 is the physical replica)."""
    return tuple(float(b) for b in jnp.geomspace(1.0, beta_min, n_replicas))


@partial(
    jax.jit,
    static_argnames=("prob_fn", "dof", "n_steps", "swap_every", "scale_steps", "proposal"),
)
def pt_chain(
    key,
    prob_fn,
    prob_params,
    init_position,
    step_size,
    dof,
    n_steps,
    betas,
    swap_every=1,
    box_L=0.0,
    scale_steps=True,
    proposal: Proposal = GaussianMove(),
):
    """One parallel-tempered chain (R replicas). Returns cold-replica positions + acceptance.

    Args:
        key: PRNG key.
        prob_fn: ``(x, params) -> logP(x)`` (= 2 log|ψ|), the *untempered* log-prob.
        prob_params: parameters for ``prob_fn``.
        init_position: ``(dof,)`` start, broadcast to all replicas.
        step_size: base local Gaussian std (replica r uses σ/√β_r if ``scale_steps``).
        dof: degrees of freedom (static).
        betas: 1-D array of inverse temperatures, betas[0] = 1.0 (the physical replica).
        swap_every: attempt a replica swap every this many steps.
        box_L: periodic box length (> 0 wraps proposals); matches the rest of the sampler.
        scale_steps: give hotter replicas larger steps (σ/√β) — generic, no Hamiltonian knowledge.
        proposal: local-move proposal family (jit-static; shared MH kernel).

    Returns:
        positions: ``(n_steps, dof)`` configurations of the **cold** (β=1) replica.
        acceptance_rate: scalar local-move acceptance of the cold replica.
    """
    betas = jnp.asarray(betas)
    n_rep = betas.shape[0]
    step_r = step_size / jnp.sqrt(betas) if scale_steps else jnp.full((n_rep,), step_size)

    pos0 = jnp.broadcast_to(init_position, (n_rep, dof))            # (R, dof)
    logp0 = jax.vmap(prob_fn, in_axes=(0, None))(pos0, prob_params)  # (R,)

    def local_step(rkey, pos, logp, s, beta):
        # Shared MH kernel, tempered: accepts on β·ΔlogP, stores untempered logP.
        return mh_kernel_log(
            rkey, prob_fn, prob_params, pos, logp, s,
            proposal=proposal, box_L=box_L, beta=beta,
        )

    def maybe_swap(swap_key, pos, logp):
        # propose one random adjacent pair (r, r+1) — symmetric proposal
        ki, ku = random.split(swap_key)
        r = random.randint(ki, (), 0, n_rep - 1)
        lp_r = jax.lax.dynamic_index_in_dim(logp, r, keepdims=False)
        lp_r1 = jax.lax.dynamic_index_in_dim(logp, r + 1, keepdims=False)
        b_r = jax.lax.dynamic_index_in_dim(betas, r, keepdims=False)
        b_r1 = jax.lax.dynamic_index_in_dim(betas, r + 1, keepdims=False)
        log_ratio = (b_r - b_r1) * (lp_r1 - lp_r)
        accept = jnp.log(random.uniform(ku)) < jnp.minimum(0.0, log_ratio)

        x_r = jax.lax.dynamic_index_in_dim(pos, r, keepdims=True)
        x_r1 = jax.lax.dynamic_index_in_dim(pos, r + 1, keepdims=True)

        def do_swap(_):
            p = jax.lax.dynamic_update_index_in_dim(pos, x_r1[0], r, 0)
            p = jax.lax.dynamic_update_index_in_dim(p, x_r[0], r + 1, 0)
            lq = jax.lax.dynamic_update_index_in_dim(logp, lp_r1, r, 0)
            lq = jax.lax.dynamic_update_index_in_dim(lq, lp_r, r + 1, 0)
            return p, lq

        return jax.lax.cond(accept, do_swap, lambda _: (pos, logp), operand=None)

    def body(carry, step):
        pos, logp, cold_acc = carry
        idx, step_key = step
        k_local, k_swap = random.split(step_key)
        rkeys = random.split(k_local, n_rep)
        pos, logp, accepted = jax.vmap(local_step, in_axes=(0, 0, 0, 0, 0))(
            rkeys, pos, logp, step_r, betas
        )
        pos, logp = jax.lax.cond(
            idx % swap_every == 0,
            lambda _: maybe_swap(k_swap, pos, logp),
            lambda _: (pos, logp),
            operand=None,
        )
        return (pos, logp, cold_acc + accepted[0]), pos[0]  # record cold replica

    idxs = jnp.arange(n_steps)
    keys = random.split(key, n_steps)
    (_, _, cold_acc), cold_positions = jax.lax.scan(body, (pos0, logp0, 0), (idxs, keys))
    return cold_positions, cold_acc / n_steps


@partial(
    jax.jit,
    static_argnames=(
        "prob_fn", "n_chains", "dof", "n_steps", "burn_in", "thinning",
        "swap_every", "scale_steps", "proposal",
    ),
)
def sample_parallel_tempering(
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
    betas=(1.0, 0.5, 0.25, 0.1),
    swap_every=1,
    box_L=0.0,
    scale_steps=True,
    proposal: Proposal = GaussianMove(),
):
    """Vectorised parallel tempering over ``n_chains``; drop-in for ``sample_and_process``.

    ``betas`` is a static tuple (``betas[0]`` must be 1.0 — the physical replica). Returns
    ``(batch_flat, last_positions, acceptance_rates)`` from the cold replica, shapes
    ``((n_chains*n_eff, dof), (n_chains, dof), (n_chains,))``.
    """
    chain_keys = random.split(key, n_chains)

    def run_chain(ckey, init_pos):
        return pt_chain(
            ckey, prob_fn, prob_params, init_pos, step_size, dof, n_steps,
            betas, swap_every, box_L, scale_steps, proposal,
        )

    raw_batch, acceptance_rates = jax.vmap(run_chain, in_axes=(0, 0))(
        chain_keys, init_positions
    )
    processed = raw_batch[:, burn_in::thinning, :]
    last_positions = raw_batch[:, -1, :]
    batch_flat = processed.reshape(-1, dof)
    return batch_flat, last_positions, acceptance_rates
