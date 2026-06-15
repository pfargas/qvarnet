"""Parallel-tempering sampler guards.

The point of PT is barrier crossing: on a bimodal target, a local-move MH chain started in one
mode stays trapped, while PT (same local proposal + temperature ladder + swaps) reaches both modes.
With a single replica (β=1 only) PT must reduce to a plain local-move chain.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from qvarnet.samplers import geometric_betas, sample_and_process, sample_parallel_tempering

D = 4.0   # half-separation of the two modes
S = 0.4   # mode width


def _bimodal_logprob():
    """logP(x) = 2 log|ψ|, |ψ|² = mix of two Gaussians at ±D (deep barrier between)."""
    def prob_fn(x, _params):
        a = -((x - D) ** 2) / (2 * S**2)
        b = -((x + D) ** 2) / (2 * S**2)
        return jnp.squeeze(logsumexp(jnp.stack([a, b], axis=-1), axis=-1))
    return prob_fn


def _frac_other_mode(samples):
    return float(np.mean(np.asarray(samples).ravel() < 0.0))


def test_pt_crosses_barrier_where_plain_mh_cannot():
    prob_fn = _bimodal_logprob()
    nchains, dof = 256, 1
    init = jnp.full((nchains, dof), D)  # everyone starts in the +D mode

    plain, _, _ = sample_and_process(
        key=jax.random.PRNGKey(0), prob_fn=prob_fn, prob_params={},
        init_positions=init, step_size=0.5, n_chains=nchains, dof=dof,
        n_steps=400, burn_in=100, thinning=2,
    )
    pt, _, _ = sample_parallel_tempering(
        key=jax.random.PRNGKey(0), prob_fn=prob_fn, prob_params={},
        init_positions=init, step_size=0.5, n_chains=nchains, dof=dof,
        n_steps=400, burn_in=100, thinning=2,
        betas=geometric_betas(6, beta_min=0.03), swap_every=1,
    )

    plain_other = _frac_other_mode(plain)
    pt_other = _frac_other_mode(pt)
    assert plain_other < 0.02, f"plain MH should stay trapped, got {plain_other:.3f} in other mode"
    assert pt_other > 0.2, f"PT should reach the other mode, got {pt_other:.3f}"


def test_single_replica_reduces_to_local_mh():
    """β=1 only ⇒ PT is just the local Gaussian MH; on a unimodal Gaussian it samples it."""
    def prob_fn(x, _):
        return jnp.squeeze(-(x**2))  # |ψ|² ∝ exp(-2x²) ⇒ N(0, 1/2)
    pt, _, _ = sample_parallel_tempering(
        key=jax.random.PRNGKey(1), prob_fn=prob_fn, prob_params={},
        init_positions=jnp.zeros((512, 1)), step_size=0.6, n_chains=512, dof=1,
        n_steps=400, burn_in=100, thinning=2, betas=(1.0,),
    )
    x = np.asarray(pt).ravel()
    assert abs(float(np.mean(x))) < 0.1
    assert abs(float(np.std(x)) - np.sqrt(0.5)) < 0.1


def test_pt_output_shapes():
    prob_fn = _bimodal_logprob()
    nchains, dof, n_steps, burn_in, thinning = 8, 1, 50, 10, 2
    batch, last, acc = sample_parallel_tempering(
        key=jax.random.PRNGKey(2), prob_fn=prob_fn, prob_params={},
        init_positions=jnp.zeros((nchains, dof)), step_size=0.5, n_chains=nchains, dof=dof,
        n_steps=n_steps, burn_in=burn_in, thinning=thinning, betas=(1.0, 0.5),
    )
    n_eff = len(range(burn_in, n_steps, thinning))
    assert batch.shape == (nchains * n_eff, dof)
    assert last.shape == (nchains, dof)
    assert acc.shape == (nchains,)
