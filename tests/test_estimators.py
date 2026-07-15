"""TrainedWavefunction guards, validated against the analytic harmonic oscillator.

Ground state log|ψ₀| = -½ Σ x² → |ψ₀|² samples are N(0, 1/2), n(x) = N·(1/√π)e^{-x²},
and a single trapped particle has a rank-1 OBDM → n₀ = 1. The Jacobi test checks the
lab-coordinate invariant: cached samples and log_psi agree between coord modes.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from qvarnet.config.coord_mode import JacobiCoords
from qvarnet.estimators import TrainedWavefunction


class GaussianLogPsi(nn.Module):
    """Exact HO ground state log|ψ| = -½ Σ x²  (no parameters)."""

    @nn.compact
    def __call__(self, x):
        return -0.5 * jnp.sum(x**2, axis=-1, keepdims=True)


def make_wf(n_particles, seed=0, **kwargs):
    model = GaussianLogPsi()
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, n_particles)))
    return TrainedWavefunction(model, variables, n_particles, seed=seed, **kwargs)


# ---------- sampling + caching ----------


def test_samples_property_requires_sample():
    wf = make_wf(2)
    try:
        _ = wf.samples
        raise AssertionError("expected RuntimeError before sampling")
    except RuntimeError:
        pass


def test_sample_shapes_and_warm_start():
    wf = make_wf(3)
    s = wf.sample(n_chains=64, n_steps=100, burn_in=50, thinning=2, step_size=0.7)
    assert s.shape == (64 * 25, 3)
    assert s is wf.samples
    assert 0.0 < wf.acceptance_rate < 1.0
    # second call warm-starts from the previous walkers (same chain count)
    first_positions = wf._last_positions
    wf.sample(n_chains=64, n_steps=20, burn_in=0)
    assert wf._last_positions.shape == first_positions.shape


def test_log_psi_baked():
    wf = make_wf(2)
    x = np.array([[0.5, -1.0], [0.0, 0.0]])
    got = np.asarray(wf.log_psi(x))
    np.testing.assert_allclose(got, [-0.5 * (0.25 + 1.0), 0.0], atol=1e-6)


# ---------- density ----------


def test_density_matches_ho_ground_state():
    wf = make_wf(1, seed=1)
    wf.sample(n_chains=256, n_steps=400, burn_in=200, step_size=1.0)
    centers, n = wf.density(bins=80, value_range=(-4, 4))
    width = centers[1] - centers[0]
    assert abs(float(np.sum(n) * width) - 1.0) < 0.05  # ∫ n dx = N = 1
    exact = (1.0 / np.sqrt(np.pi)) * np.exp(-(centers**2))
    i0 = np.argmin(np.abs(centers))
    assert abs(n[i0] - exact[i0]) < 0.07


# ---------- g(r), S(k) ----------


def test_pair_correlation_and_structure_factor():
    wf = make_wf(4, seed=2)
    wf.sample(n_chains=128, n_steps=200, burn_in=100)
    r, g = wf.pair_correlation(bins=40, value_range=(0, 6))
    assert r.shape == (40,) and np.all(g >= 0) and g.sum() > 0
    k, S = wf.structure_factor(k_values=[0.5, 5.0, 30.0])
    assert np.all(S > 0)
    assert abs(S[-1] - 1.0) < 0.15  # S(k) → 1 as k → ∞


def test_structure_factor_needs_k_without_box():
    wf = make_wf(2)
    wf.sample(n_chains=32, n_steps=50, burn_in=10)
    try:
        wf.structure_factor()
        raise AssertionError("expected ValueError without box_L")
    except ValueError:
        pass


# ---------- OBDM ----------


def test_condensate_fraction_single_particle_is_one():
    wf = make_wf(1, seed=4)
    wf.sample(n_chains=128, n_steps=200, burn_in=100, step_size=1.0)
    grid = np.linspace(-4.0, 4.0, 41)
    wf.obdm(grid)
    n0 = wf.condensate_fraction()  # uses the cached ρ₁
    assert n0 > 0.9, f"single-particle condensate fraction {n0:.3f} should be ≈ 1"
    occs, orbs = wf.natural_orbitals()
    assert occs[0] > 0.9 and orbs.shape == (41, 41)


def test_obdm_displacement_zero_is_one():
    wf = make_wf(2, seed=5)
    wf.sample(n_chains=128, n_steps=150, burn_in=75)
    deltas, rho1 = wf.obdm_displacement([0.0, 0.5, 1.0, 2.0])
    assert abs(rho1[0] - 1.0) < 1e-6  # ρ₁(Δ=0) = 1
    assert rho1[-1] < rho1[0]  # decays with separation


# ---------- coord modes ----------


def test_jacobi_samples_are_lab_coordinates():
    """Under JacobiCoords the walker lives in N-1 relative coords, but wf.samples
    must be N lab coords with the CM pinned at the origin."""
    n = 3
    model = GaussianLogPsi()
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, n)))
    wf = TrainedWavefunction(
        model, variables, n, coord_mode=JacobiCoords(n_particles_physical=n), seed=6
    )
    assert wf._sampler_dof == n - 1
    s = wf.sample(n_chains=64, n_steps=150, burn_in=75)
    assert s.shape[1] == n  # lab dof, not sampler dof
    cm = s.mean(axis=1)
    np.testing.assert_allclose(cm, 0.0, atol=1e-5)  # CM removed
    # log_psi consumes lab coords directly in both modes
    assert np.isfinite(np.asarray(wf.log_psi(s[:10]))).all()
    # estimators run on the lab-space samples without extra plumbing
    centers, dens = wf.density(bins=40, value_range=(-4, 4))
    assert dens.sum() > 0
