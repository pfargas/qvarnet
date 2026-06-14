"""Step 5 guards. Observables validated against the analytic non-interacting harmonic
oscillator: ground state |ψ₀(x)|² ∝ exp(-x²) (variance 1/2), so exact samples are N(0, 1/2)
and n(x) = N·(1/√π) exp(-x²). A single trapped particle has a rank-1 OBDM → n₀ = 1."""

import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from qvarnet.observables import (
    blocking_error,
    condensate_fraction,
    density_histogram,
    obdm_displacement,
    obdm_grid,
    pair_correlation,
    structure_factor,
)


class GaussianLogPsi(nn.Module):
    """Exact HO ground state log|ψ| = -½ Σ x²  (no parameters)."""

    @nn.compact
    def __call__(self, x):
        return -0.5 * jnp.sum(x**2, axis=-1, keepdims=True)


def ho_samples(M, N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, np.sqrt(0.5), size=(M, N))  # ~ |ψ₀|²


# ---------- blocking ----------

def test_blocking_error_iid_and_correlated():
    rng = np.random.default_rng(0)
    iid = rng.standard_normal(8192)
    err_iid, _ = blocking_error(iid)
    assert abs(err_iid - 1.0 / np.sqrt(8192)) < 0.3 / np.sqrt(8192) * 3

    phi = 0.8
    eps = rng.standard_normal(8192)
    corr = np.empty(8192)
    corr[0] = eps[0]
    for t in range(1, 8192):
        corr[t] = phi * corr[t - 1] + eps[t]
    err_corr, _ = blocking_error(corr)
    assert err_corr > 2 * err_iid  # autocorrelation inflates the error


# ---------- density n(x) ----------

def test_density_matches_ho_ground_state():
    samples = ho_samples(40000, 1, seed=1)
    centers, n = density_histogram(samples, n_particles=1, bins=80, value_range=(-4, 4))
    width = centers[1] - centers[0]
    integral = float(np.sum(n) * width)
    assert abs(integral - 1.0) < 0.05  # ∫ n dx = N = 1
    exact = (1.0 / np.sqrt(np.pi)) * np.exp(-(centers**2))
    peak = n[np.argmin(np.abs(centers))]
    assert abs(peak - exact[np.argmin(np.abs(centers))]) < 0.05


# ---------- g(r) ----------

def test_pair_correlation_shape():
    samples = ho_samples(5000, 4, seed=2)
    r, g = pair_correlation(samples, n_particles=4, bins=40, value_range=(0, 6))
    assert r.shape == (40,) and g.shape == (40,)
    assert np.all(g >= 0) and g.sum() > 0


# ---------- S(k) ----------

def test_structure_factor_large_k_to_one():
    samples = ho_samples(8000, 4, seed=3)
    k, S = structure_factor(samples, n_particles=4, k_values=[0.5, 5.0, 30.0])
    assert np.all(S > 0)
    assert abs(S[-1] - 1.0) < 0.15  # S(k) → 1 as k → ∞


# ---------- OBDM ----------

def test_condensate_fraction_single_particle_is_one():
    model = GaussianLogPsi()
    variables = model.init(__import__("jax").random.PRNGKey(0), jnp.ones((1, 1)))
    samples = ho_samples(4000, 1, seed=4)
    grid = np.linspace(-4.0, 4.0, 41)
    _, rho = obdm_grid(model, variables, samples, grid, n_particles=1)
    n0 = condensate_fraction(rho, grid)
    assert n0 > 0.9, f"single-particle condensate fraction {n0:.3f} should be ≈ 1"


def test_obdm_displacement_zero_is_one():
    model = GaussianLogPsi()
    variables = model.init(__import__("jax").random.PRNGKey(0), jnp.ones((1, 2)))
    samples = ho_samples(3000, 2, seed=5)
    deltas, rho1 = obdm_displacement(model, variables, samples, [0.0, 0.5, 1.0, 2.0])
    assert abs(rho1[0] - 1.0) < 1e-6  # ρ₁(Δ=0) = 1
    assert rho1[-1] < rho1[0]  # decays with separation
