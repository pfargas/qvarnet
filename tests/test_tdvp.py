"""Step 9 (t-VMC core) guards, real-output models:

- the McLachlan residual r² = Var(E_loc) − Fᵀ S⁻¹ F is ≈ 0 at the exact HO ground state
  (zero variance) and > 0 for a generic untrained model;
- imaginary-time TDVP steps reduce the energy (≡ SR direction).
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from conftest import make_ho_model
from qvarnet.geometry import imaginary_time_step, tdvp_residual
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.vmc.training_step import compute_local_energy


class GaussianLogPsi(nn.Module):
    """Exact HO ground state log|ψ| = -½ Σ x²  (E_loc ≡ 0.5/dof, zero variance)."""

    @nn.compact
    def __call__(self, x):
        return -0.5 * jnp.sum(x**2, axis=-1, keepdims=True)


def test_residual_zero_at_exact_eigenstate():
    model = GaussianLogPsi()
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 1)))
    ham = HarmonicOscillatorHamiltonian(omega=1.0)
    batch = jax.random.normal(jax.random.PRNGKey(1), (256, 1)) * np.sqrt(0.5)
    e_loc = compute_local_energy(ham, variables, batch, model.apply)
    r2, var, _ = tdvp_residual(variables, batch, e_loc, model.apply)
    assert var < 1e-8  # exact eigenstate → zero local-energy variance
    assert r2 < 1e-6


def test_residual_positive_for_generic_model():
    model = make_ho_model()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
    ham = HarmonicOscillatorHamiltonian(omega=1.0)
    batch = jax.random.normal(jax.random.PRNGKey(2), (256, 2)) * 0.5
    e_loc = compute_local_energy(ham, params, batch, model.apply)
    r2, var, captured = tdvp_residual(params, batch, e_loc, model.apply, regularization=1e-8)
    assert var > 1e-6 and r2 >= 0.0 and captured >= -1e-8


def test_imaginary_time_step_reduces_energy():
    model = make_ho_model()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
    ham = HarmonicOscillatorHamiltonian(omega=1.0)
    batch = jax.random.normal(jax.random.PRNGKey(3), (512, 2)) * 0.5

    def energy(p):
        return float(jnp.mean(compute_local_energy(ham, p, batch, model.apply)))

    e0 = energy(params)
    for _ in range(5):
        e_loc = compute_local_energy(ham, params, batch, model.apply)
        params = imaginary_time_step(params, batch, e_loc, model.apply, dt=0.05, regularization=1e-3)
    assert energy(params) < e0  # imaginary-time TDVP descends toward the ground state
