"""The folx kinetic branch and mass-imbalanced (Particles) kinetic energy.

Core guarantees:
* folx branch ≡ forward_ad branch (both exact) on three representative ansätze:
  plain MLP, CS-style MLP+Jastrow, and the soft-sphere DeepSet with PBC features;
* per-dof mass weights: weighted-AD ≡ folx(√m rescaling), and both reproduce the
  analytic mass-imbalanced harmonic-oscillator ground-state energy E0 = ½ Σ 1/√mᵢ;
* masses=None is exactly the pre-Particles behaviour;
* the Hamiltonian-level dispatch (incl. the CS ×2 override) routes through folx.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qvarnet import NoBoundary, Particles, PeriodicBoundary
from qvarnet.hamiltonian.continuous import (
    CalogeroSutherlandHamiltonian,
    HarmonicOscillatorHamiltonian,
)
from qvarnet.hamiltonian.kinetic import kinetic_log
from qvarnet.hamiltonian.laplacian import (
    laplacian_forward_ad,
    laplacian_full_hessian,
    laplacian_hutchinson,
)
from qvarnet.models import MLP, DeepSet, GaussianEnvelope, LogJastrow, LogWavefunction

BATCH = 8


def _init(model, dof, key=0):
    k1, k2 = jax.random.split(jax.random.PRNGKey(key))
    params = model.init(k1, jnp.zeros((1, dof)))
    samples = jax.random.normal(k2, (BATCH, dof))
    return params, samples


def _mlp():
    dof = 6
    model = MLP(hidden=[8], output_dim=1)
    params, samples = _init(model, dof)
    return model.apply, params, samples


def _cs_jastrow():
    n = 4
    model = LogWavefunction(
        transform=NoBoundary(),
        network=MLP(hidden=[8], output_dim=1),
        envelope=GaussianEnvelope(),
        jastrow=LogJastrow(n_particles=n),
    )
    params, samples = _init(model, n)
    return model.apply, params, samples


def _pbc_deepset():
    n, n_dim, box = 4, 3, 5.0
    model = LogWavefunction(
        n_particles=n,
        n_dim=n_dim,
        transform=PeriodicBoundary(L=box),
        network=DeepSet(phi_hidden=[8], F_hidden=[8], hidden_internal_dim=4),
    )
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    params = model.init(k1, jnp.zeros((1, n * n_dim)))
    samples = jax.random.uniform(k2, (BATCH, n * n_dim)) * box
    return model.apply, params, samples


ANSATZE = {"mlp": _mlp, "cs_jastrow": _cs_jastrow, "pbc_deepset": _pbc_deepset}


# ── folx ≡ forward_ad (equal masses) ─────────────────────────────────────────────────


@pytest.mark.parametrize("ansatz", list(ANSATZE))
def test_folx_matches_forward_ad(ansatz):
    apply, params, samples = ANSATZE[ansatz]()
    t_ad = kinetic_log(params, samples, apply, laplacian_fn=laplacian_forward_ad)
    t_folx = kinetic_log(params, samples, apply, use_folx=True)
    np.testing.assert_allclose(t_folx, t_ad, rtol=2e-4, atol=1e-4)


def test_folx_sparse_matches_dense():
    apply, params, samples = _pbc_deepset()
    dense = kinetic_log(params, samples, apply, use_folx=True)
    sparse = kinetic_log(params, samples, apply, use_folx=True, sparsity_threshold=6)
    np.testing.assert_allclose(sparse, dense, rtol=2e-4, atol=1e-4)


# ── mass imbalance ───────────────────────────────────────────────────────────────────


def test_weighted_ad_matches_folx_unequal_masses():
    apply, params, samples = _mlp()  # dof = 6
    w = Particles(n=6, n_dim=1, masses=(1.0, 1.0, 2.0, 2.0, 5.0, 5.0)).dof_weights()
    t_ad = kinetic_log(params, samples, apply,
                       laplacian_fn=laplacian_forward_ad, dof_weights=w)
    t_folx = kinetic_log(params, samples, apply, use_folx=True, dof_weights=w)
    np.testing.assert_allclose(t_folx, t_ad, rtol=2e-4, atol=1e-4)


def test_weighted_forward_ad_matches_full_hessian():
    apply, params, samples = _mlp()
    w = jnp.array([1.0, 0.5, 0.25, 2.0, 1.0, 0.1])

    def log_psi(x):
        return apply(params, x[None]).squeeze()

    lap_fa = laplacian_forward_ad(log_psi, samples, weights=w)
    lap_fh = laplacian_full_hessian(log_psi, samples, weights=w)
    np.testing.assert_allclose(lap_fa, lap_fh, rtol=2e-4, atol=1e-4)


def test_hutchinson_unit_weights_equal_unweighted():
    apply, params, samples = _mlp()

    def log_psi(x):
        return apply(params, x[None]).squeeze()

    key = jax.random.PRNGKey(3)
    unweighted = laplacian_hutchinson(log_psi, samples, key, n_terms=4)
    unit = laplacian_hutchinson(log_psi, samples, key, n_terms=4,
                                weights=jnp.ones(samples.shape[-1]))
    np.testing.assert_allclose(unit, unweighted, rtol=1e-6)


@pytest.mark.parametrize("use_folx", [False, True])
def test_mass_imbalanced_ho_is_analytic(use_folx):
    """H = -Σ 1/(2mᵢ)∂²ᵢ + ½Σxᵢ²; ψ0 = exp(-½Σ√mᵢ xᵢ²) ⇒ E_loc ≡ ½ Σ 1/√mᵢ."""
    masses = (1.0, 4.0, 9.0, 16.0)
    sqrt_m = jnp.sqrt(jnp.asarray(masses))

    def analytic_apply(params, x_batch):        # exact ground state, no parameters
        return -0.5 * jnp.sum(sqrt_m * x_batch**2, axis=-1)

    w = Particles(n=4, n_dim=1, masses=masses).dof_weights()
    samples = jax.random.normal(jax.random.PRNGKey(7), (BATCH, 4))
    kwargs = dict(use_folx=True) if use_folx else dict(laplacian_fn=laplacian_forward_ad)
    t = kinetic_log(None, samples, analytic_apply, dof_weights=w, **kwargs)
    e_loc = t + 0.5 * jnp.sum(samples**2, axis=-1)
    e_exact = 0.5 * float(jnp.sum(1.0 / sqrt_m))
    np.testing.assert_allclose(e_loc, jnp.full(BATCH, e_exact), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("use_folx", [False, True])
def test_masses_none_equals_unit_masses(use_folx):
    apply, params, samples = _mlp()
    kwargs = dict(use_folx=True) if use_folx else {}
    t_none = kinetic_log(params, samples, apply, dof_weights=None, **kwargs)
    t_unit = kinetic_log(params, samples, apply,
                         dof_weights=jnp.ones(samples.shape[-1]), **kwargs)
    np.testing.assert_allclose(t_unit, t_none, rtol=1e-6)


# ── Particles metadata ───────────────────────────────────────────────────────────────


def test_particles_dof_weights_repeat_over_dimensions():
    p = Particles(n=2, n_dim=3, masses=(1.0, 4.0))
    assert p.dof == 6
    np.testing.assert_allclose(p.dof_weights(), [1, 1, 1, 0.25, 0.25, 0.25])
    assert Particles(n=2, n_dim=3).dof_weights() is None  # equal masses -> fast path


def test_particles_validation():
    with pytest.raises(ValueError, match="masses has 3"):
        Particles(n=2, masses=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="positive"):
        Particles(n=2, masses=(1.0, -2.0))
    with pytest.raises(ValueError, match="n > 0"):
        Particles(n=0)
    assert hash(Particles(n=2, masses=(1.0, 2.0)))  # static-field (jit) requirement


# ── Hamiltonian-level dispatch ───────────────────────────────────────────────────────


def test_harmonic_oscillator_folx_dispatch():
    apply, params, samples = _mlp()
    e_ad = HarmonicOscillatorHamiltonian().local_energy(params, samples, apply)
    e_folx = HarmonicOscillatorHamiltonian(laplacian_method="folx").local_energy(
        params, samples, apply)
    np.testing.assert_allclose(e_folx, e_ad, rtol=2e-4, atol=1e-4)


def test_cs_override_routes_through_folx():
    """The CS ×2 kinetic override must respect laplacian_method (the silent-bypass trap)."""
    apply, params, samples = _cs_jastrow()
    ham_ad = CalogeroSutherlandHamiltonian(L=0.8, epsilon=1e-4)
    ham_folx = CalogeroSutherlandHamiltonian(L=0.8, epsilon=1e-4, laplacian_method="folx")
    t_ad = ham_ad.kinetic_local_energy(params, samples, apply)
    t_folx = ham_folx.kinetic_local_energy(params, samples, apply)
    np.testing.assert_allclose(t_folx, t_ad, rtol=2e-4, atol=1e-4)
    # and the ×2 convention is really applied on the folx path
    base = kinetic_log(params, samples, apply, use_folx=True)
    np.testing.assert_allclose(t_folx, 2 * base, rtol=1e-6)


def test_hamiltonian_masses_flow_into_kinetic():
    apply, params, samples = _mlp()
    particles = Particles(n=6, n_dim=1, masses=(1.0, 1.0, 2.0, 2.0, 5.0, 5.0))
    ham = HarmonicOscillatorHamiltonian(particles=particles)
    direct = kinetic_log(params, samples, apply, laplacian_fn=laplacian_forward_ad,
                         dof_weights=particles.dof_weights())
    np.testing.assert_allclose(
        ham.kinetic_local_energy(params, samples, apply), direct, rtol=1e-6)


def test_unknown_method_still_raises():
    apply, params, samples = _mlp()
    with pytest.raises(ValueError, match="Unknown laplacian_method"):
        HarmonicOscillatorHamiltonian(laplacian_method="nope").kinetic_local_energy(
            params, samples, apply)


def test_missing_folx_gives_helpful_error(monkeypatch):
    apply, params, samples = _mlp()
    for mod in [m for m in sys.modules if m == "folx" or m.startswith("folx.")]:
        monkeypatch.setitem(sys.modules, mod, None)  # import -> ImportError
    with pytest.raises(ImportError, match="requires the folx package"):
        kinetic_log(params, samples, apply, use_folx=True)
