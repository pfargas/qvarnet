"""Periodic boundary condition (PBC) guards.

Covers the four PBC seams and their independent on/off toggles:
  * periodic *ansatz*    — sin/cos encoding (no particle scramble) + periodic Jastrow;
  * periodic *sampler*   — MH proposals wrapped into [0, L);
  * periodic *Hamiltonian* — min-image interactions (LatticeBoseHamiltonian);
  * periodic *observables* — coordinate folding.

The two correctness anchors are:
  (1) the encoding groups each particle's own (sin, cos) features, so a permutation-
      invariant DeepSet on a periodic box stays permutation invariant (regression for
      the old concatenate([sin(all), cos(all)]) scramble);
  (2) per-sample local energy is invariant under a global shift x -> x + L for a periodic
      ansatz + min-image Hamiltonian.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from qvarnet.boundaries import PeriodicBoundary
from qvarnet.config.training_setup import SamplingConfig, TrainingConfig
from qvarnet.hamiltonian.periodic import LatticeBoseHamiltonian
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.deep_set import DeepSet
from qvarnet.models.jastrow import LogJastrow
from qvarnet.models.mlp import MLP
from qvarnet.observables import density_histogram, structure_factor
from qvarnet.samplers import sample_and_process
from qvarnet.train import train
from qvarnet.vmc.probability import build_prob_fn

L = 2.0 * np.pi


def _periodic_mlp():
    return LogWavefunction(
        network=MLP(hidden=[16, 16], output_dim=1, hidden_activation=nn.tanh),
        transform=PeriodicBoundary(L),
    )


def _periodic_deepset(n_particles):
    return LogWavefunction(
        network=DeepSet(phi_hidden=[8], F_hidden=[8]),
        transform=PeriodicBoundary(L),
        n_particles=n_particles,
    )


# ---------- (1) encoding: periodicity + per-particle grouping ----------

def test_encode_is_L_periodic():
    enc = PeriodicBoundary(L)
    x = jnp.array([[0.3, 1.1, -0.7]])
    np.testing.assert_allclose(enc.encode(x), enc.encode(x + L), atol=1e-5)


def test_encode_groups_sin_cos_per_particle():
    """After the (..., N) -> (..., 2N) encode and a (..., N, 2) reshape, row i must be
    [sin(2π x_i/L), cos(2π x_i/L)] — the old concatenate scramble fails this."""
    enc = PeriodicBoundary(L)
    x = jnp.array([0.3, 1.1, -0.7])
    out = enc.encode(x).reshape(3, 2)
    phase = 2.0 * jnp.pi * x / L
    np.testing.assert_allclose(out[:, 0], jnp.sin(phase), atol=1e-6)
    np.testing.assert_allclose(out[:, 1], jnp.cos(phase), atol=1e-6)


def test_periodic_deepset_is_permutation_invariant():
    """The real regression for the scramble bug: a DeepSet on a periodic box must be
    invariant to particle relabelling. The old encoding broke this."""
    n = 4
    model = _periodic_deepset(n)
    x = jnp.array([[0.2, 1.3, 2.7, 4.1]])
    params = model.init(jax.random.PRNGKey(0), x)
    perm = jnp.array([[1.3, 4.1, 0.2, 2.7]])  # a permutation of the same particles
    a = model.apply(params, x)
    b = model.apply(params, perm)
    np.testing.assert_allclose(a, b, atol=1e-5)


# ---------- periodic ansatz: log|ψ| is L-periodic ----------

def test_periodic_ansatz_logpsi_is_periodic():
    model = _periodic_mlp()
    x = jnp.array([[0.4, 2.1, 5.0]])
    params = model.init(jax.random.PRNGKey(1), x)
    np.testing.assert_allclose(model.apply(params, x), model.apply(params, x + L), atol=1e-5)


def test_periodic_jastrow_is_periodic_and_open_is_not():
    n = 3
    x = jnp.array([[0.4, 2.1, 5.0]])
    shift = jnp.array([[L, 0.0, 0.0]])  # move one particle by a full box
    per = LogJastrow(n_particles=n, L=L)
    p = per.init(jax.random.PRNGKey(0), x)
    np.testing.assert_allclose(per.apply(p, x), per.apply(p, x + shift), atol=1e-5)
    # Open-boundary Jastrow (L=None) is NOT periodic — guards against silently using it.
    op = LogJastrow(n_particles=n)
    q = op.init(jax.random.PRNGKey(0), x)
    assert not np.allclose(op.apply(q, x), op.apply(q, x + shift), atol=1e-3)


# ---------- (2) Hamiltonian: shift-by-L invariance of the local energy ----------

def test_local_energy_invariant_under_global_shift_by_L():
    """Per-sample E_loc(x) == E_loc(x + L) for a periodic ansatz + min-image lattice-Bose H,
    with both a periodic lattice potential (V0) and a min-image contact term (g)."""
    n = 3
    model = _periodic_mlp()
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.uniform(0.0, L, size=(64, n)))
    params = model.init(jax.random.PRNGKey(2), x)
    H = LatticeBoseHamiltonian(a=L, V0=0.7, g=0.5, sigma=0.8, boundary=PeriodicBoundary(L))
    key = jax.random.PRNGKey(3)
    e_x = H.local_energy(params, x, model.apply, key=key)
    e_xL = H.local_energy(params, x + L, model.apply, key=key)
    np.testing.assert_allclose(np.asarray(e_x), np.asarray(e_xL), atol=1e-4)


def test_lattice_bose_potential_min_image_used():
    """A pair straddling the boundary must use the short (min-image) distance."""
    H = LatticeBoseHamiltonian(a=L, V0=0.0, g=1.0, sigma=1.0, boundary=PeriodicBoundary(L))
    near = jnp.array([[0.1, L - 0.1]])  # min-image distance 0.2, raw distance L-0.2
    far = jnp.array([[0.1, 0.1 + L / 2]])  # distance L/2 (max separation)
    assert float(H.potential_energy(near)[0]) > float(H.potential_energy(far)[0])


# ---------- (3) sampler: wrap keeps walkers in the box ----------

def test_pbc_sampler_wraps_into_box():
    model = _periodic_mlp()
    x0 = jnp.zeros((8, 3))
    params = model.init(jax.random.PRNGKey(0), x0)
    prob_fn = build_prob_fn(model.apply)
    samples, _, _ = sample_and_process(
        key=jax.random.PRNGKey(4),
        prob_fn=prob_fn,
        prob_params=params,
        init_positions=x0,
        step_size=0.5,
        n_chains=8,
        dof=3,
        n_steps=200,
        burn_in=20,
        thinning=2,
        box_L=L,
    )
    s = np.asarray(samples)
    assert s.min() >= 0.0 and s.max() < L


def test_pbc_sampler_disabled_leaves_covering_space():
    """box_L=0 (default) must NOT wrap — walkers may leave [0, L)."""
    model = _periodic_mlp()
    x0 = jnp.zeros((8, 3))
    params = model.init(jax.random.PRNGKey(0), x0)
    prob_fn = build_prob_fn(model.apply)
    samples, _, _ = sample_and_process(
        key=jax.random.PRNGKey(4),
        prob_fn=prob_fn,
        prob_params=params,
        init_positions=x0,
        step_size=0.8,
        n_chains=8,
        dof=3,
        n_steps=400,
        burn_in=20,
        thinning=2,
    )
    s = np.asarray(samples)
    assert s.min() < 0.0 or s.max() >= L  # free diffusion escapes the cell


# ---------- end-to-end: free particle on a ring → E₀ = 0 ----------

def test_free_particle_on_ring_converges_to_zero(tmp_path):
    result = train(
        shape=(128, 1),
        model=_periodic_mlp(),
        optimizer=optax.adam(1e-2),
        hamiltonian=LatticeBoseHamiltonian(a=L, V0=0.0, g=0.0, boundary=PeriodicBoundary(L)),
        training_config=TrainingConfig(n_epochs=250, rng_seed=0, checkpoint_path=str(tmp_path)),
        sampler_params=SamplingConfig(
            step_size=0.6,
            chain_length=200,
            thermalization_steps=50,
            thinning_factor=2,
            box_L=L,  # PBC sampler on
        ),
    )
    energies = np.array([s.energy for s in result.history])
    tail = float(np.mean(energies[-30:]))
    assert np.isfinite(tail)
    assert abs(tail) < 0.1, f"free-ring tail energy {tail:.4f} not near E0=0"


# ---------- (4) observables: folding & commensurate-k invariance ----------

def test_density_histogram_folds_into_box():
    rng = np.random.default_rng(0)
    # samples spread across several cells (covering space)
    samples = rng.uniform(-2 * L, 3 * L, size=(20000, 2))
    centers, n = density_histogram(samples, n_particles=2, bins=50, L=L)
    assert centers.min() >= 0.0 and centers.max() <= L
    width = centers[1] - centers[0]
    assert abs(float(np.sum(n) * width) - 2.0) < 0.05  # ∫ n dx = N = 2


def test_structure_factor_invariant_on_commensurate_k():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, L, size=(3000, 4))
    k = [2.0 * np.pi * m / L for m in (1, 2, 3)]  # commensurate
    _, S_x = structure_factor(x, n_particles=4, k_values=k)
    _, S_shift = structure_factor(x + L, n_particles=4, k_values=k)  # shift by a box
    np.testing.assert_allclose(S_x, S_shift, atol=1e-6)
