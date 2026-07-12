"""Proposal-family tests for the refactored MH sampler.

1. Every symmetric family samples the correct target (moments of a known Gaussian).
2. Subset moves touch only the coordinates they claim to move.
3. The refactor's motivation holds: at fixed step size, a full-configuration move's
   acceptance collapses as dof grows while a particle-subset move's stays high.
4. PT still crosses barriers with a subset proposal (shared-kernel regression).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qvarnet.config.training_setup import SamplingConfig, parse_sampler_params
from qvarnet.samplers import (
    DoFSubsetMove,
    GaussianMove,
    ParticleSubsetMove,
    UniformMove,
    resolve_proposal,
    sample_and_process,
)


def _gauss_logprob(x, _params):
    """logP for x ~ N(0, 1/2) per coordinate (|ψ|² ∝ exp(-2·x²/2·...)): logP = -x²."""
    return jnp.sum(-(x**2))


def _run(proposal, dof=4, step_size=0.7, n_steps=800, n_chains=256, seed=0):
    return sample_and_process(
        key=jax.random.PRNGKey(seed),
        prob_fn=_gauss_logprob,
        prob_params={},
        init_positions=jnp.zeros((n_chains, dof)),
        step_size=step_size,
        n_chains=n_chains,
        dof=dof,
        n_steps=n_steps,
        burn_in=200,
        thinning=2,
        proposal=proposal,
    )


@pytest.mark.parametrize(
    "proposal",
    [
        GaussianMove(),
        UniformMove(),
        ParticleSubsetMove(n_move=2, n_dim=1),
        DoFSubsetMove(k=2),
    ],
    ids=["gaussian", "uniform", "particle-subset", "dof-subset"],
)
def test_proposal_samples_correct_target(proposal):
    """Detailed balance check by moments: every family must sample N(0, 1/2)."""
    samples, _, acc = _run(proposal)
    x = np.asarray(samples).ravel()
    assert abs(x.mean()) < 0.05, f"mean off: {x.mean():.3f}"
    assert abs(x.std() - np.sqrt(0.5)) < 0.05, f"std off: {x.std():.3f} vs {np.sqrt(0.5):.3f}"
    assert 0.05 < float(np.mean(np.asarray(acc))) < 1.0


def test_symmetric_corrections_are_zero():
    key = jax.random.PRNGKey(0)
    x = jnp.ones(6)
    for p in (GaussianMove(), UniformMove(), ParticleSubsetMove(2, 1), DoFSubsetMove(3)):
        _, corr = p.propose(key, x, 0.5)
        assert corr == 0.0


def test_subset_moves_touch_only_their_subset():
    key = jax.random.PRNGKey(1)
    x = jnp.zeros(12)

    prop, _ = ParticleSubsetMove(n_move=2, n_dim=3).propose(key, x, 1.0)
    moved_particles = np.asarray(prop).reshape(4, 3)  # particle-major
    n_moved = int(np.sum(np.any(moved_particles != 0.0, axis=1)))
    assert n_moved == 2
    # all-or-nothing per particle: a moved particle moves in every dimension
    for row in moved_particles:
        assert np.all(row != 0.0) or np.all(row == 0.0)

    prop, _ = DoFSubsetMove(k=5).propose(key, x, 1.0)
    assert int(np.sum(np.asarray(prop) != 0.0)) == 5


def test_subset_acceptance_survives_large_dof():
    """The refactor's raison d'être: at fixed step, full-configuration acceptance
    collapses with dof; moving one particle keeps it high."""
    dof = 64
    _, _, acc_full = _run(GaussianMove(), dof=dof, step_size=0.7, n_steps=300)
    _, _, acc_sub = _run(ParticleSubsetMove(n_move=1), dof=dof, step_size=0.7, n_steps=300)
    a_full = float(np.mean(np.asarray(acc_full)))
    a_sub = float(np.mean(np.asarray(acc_sub)))
    assert a_sub > 2 * a_full, f"subset {a_sub:.3f} vs full {a_full:.3f}"
    assert a_sub > 0.5


def test_resolve_proposal_and_config_wiring():
    assert resolve_proposal("gaussian") == GaussianMove()
    assert resolve_proposal("uniform") == UniformMove()
    assert resolve_proposal(("particle-subset", {"n_move": 3, "n_dim": 2})) == ParticleSubsetMove(
        3, 2
    )
    assert resolve_proposal(DoFSubsetMove(4)) == DoFSubsetMove(4)
    with pytest.raises(ValueError, match="Unknown proposal"):
        resolve_proposal("mala")
    with pytest.raises(ValueError, match="needs parameters"):
        resolve_proposal("particle-subset")

    # dict path (train(sampler_params={...})) and default
    cfg = parse_sampler_params(
        {"step_size": 0.5, "chain_length": 21, "thermalization_steps": 20,
         "thinning_factor": 1, "proposal": ("particle-subset", {"n_move": 2})}
    )
    assert cfg.proposal == ParticleSubsetMove(2, 1)
    cfg = SamplingConfig(step_size=0.5, chain_length=21, thermalization_steps=20,
                         thinning_factor=1)
    assert cfg.proposal == GaussianMove()
    assert hash(cfg) is not None  # stays jit-static


def test_train_end_to_end_with_subset_proposal(tmp_path):
    """Wiring regression: proposal flows dict → parse_sampler_params → SamplingConfig
    → full_update → sample_and_process inside train() without a retrace error."""
    import optax
    from conftest import make_ho_model

    from qvarnet.config.training_setup import TrainingConfig
    from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
    from qvarnet.train import train

    result = train(
        shape=(32, 2),
        model=make_ho_model(),
        optimizer=optax.adam(1e-2),
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(
            n_epochs=5, rng_seed=0, checkpoint_path=str(tmp_path), print_summary=False
        ),
        sampler_params={
            "step_size": 0.5,
            "chain_length": 21,
            "thermalization_steps": 20,
            "thinning_factor": 1,
            "proposal": ("particle-subset", {"n_move": 1}),
        },
    )
    e = np.array([float(s.energy) for s in result.history])
    assert np.all(np.isfinite(e))


def test_pt_with_subset_proposal_crosses_barrier():
    """Shared-kernel regression: PT with a non-default proposal still mixes modes."""
    from jax.scipy.special import logsumexp

    from qvarnet.samplers import geometric_betas, sample_parallel_tempering

    D, S = 4.0, 0.4

    def prob_fn(x, _):
        a = -((x - D) ** 2) / (2 * S**2)
        b = -((x + D) ** 2) / (2 * S**2)
        return jnp.squeeze(logsumexp(jnp.stack([a, b], axis=-1), axis=-1))

    pt, _, _ = sample_parallel_tempering(
        key=jax.random.PRNGKey(0), prob_fn=prob_fn, prob_params={},
        init_positions=jnp.full((128, 1), D), step_size=0.5, n_chains=128, dof=1,
        n_steps=400, burn_in=100, thinning=2,
        betas=geometric_betas(6, beta_min=0.03), swap_every=1,
        proposal=DoFSubsetMove(k=1),
    )
    frac_other = float(np.mean(np.asarray(pt).ravel() < 0.0))
    assert frac_other > 0.2, f"PT with subset proposal stayed trapped: {frac_other:.3f}"
