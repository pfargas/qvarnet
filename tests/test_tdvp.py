"""Step 9 (t-VMC core) guards, real-output models:

- the McLachlan residual r² = Var(E_loc) − Fᵀ S⁻¹ F is ≈ 0 at the exact HO ground state
  (zero variance) and > 0 for a generic untrained model;
- imaginary-time TDVP steps reduce the energy (≡ SR direction).

The physical guards (residual > 0, energy descent) only hold for |ψ_θ|²-distributed
samples — the SR/TDVP identity ∇⟨E⟩ = 2⟨O(E_loc−Ē)⟩ is an importance-sampling identity,
not an algebraic one — so the batches here are drawn by Metropolis-Hastings, not from an
arbitrary fixed distribution.
"""

import jax
import jax.numpy as jnp
import numpy as np
from conftest import make_ho_model
from flax import linen as nn

from qvarnet.geometry import imaginary_time_step, tdvp_residual
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.samplers.kernel import mh_chain
from qvarnet.vmc.probability import build_prob_fn
from qvarnet.vmc.training_step import compute_local_energy


def sample_psi2(model_apply, params, key, n_chains, dof, n_steps=200, step_size=0.6):
    """Draw a batch of |ψ_θ|²-distributed configurations via n_chains MH chains.

    Returns the final position of each chain (one decorrelated sample per chain), so the
    batch is M = n_chains independent draws — the regime the SR/TDVP identities assume.
    """
    prob_fn = build_prob_fn(model_apply)
    init_key, run_key = jax.random.split(key)
    init_positions = jax.random.normal(init_key, (n_chains, dof))
    chain_keys = jax.random.split(run_key, n_chains)

    def one_chain(ckey, init_pos):
        positions, _ = mh_chain(ckey, prob_fn, params, init_pos, step_size, n_steps)
        return positions[-1]

    return jax.vmap(one_chain)(chain_keys, init_positions)


def test_residual_zero_at_exact_eigenstate():
    class GaussianLogPsi(nn.Module):
        """Exact HO ground state log|ψ| = -½ Σ x²  (E_loc ≡ 0.5/dof, zero variance)."""

        @nn.compact
        def __call__(self, x):
            return -0.5 * jnp.sum(x**2, axis=-1, keepdims=True)

    model = GaussianLogPsi()
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 1)))
    ham = HarmonicOscillatorHamiltonian(omega=1.0)
    batch = jax.random.normal(jax.random.PRNGKey(1), (256, 1)) * np.sqrt(0.5)
    e_loc = compute_local_energy(ham, variables, batch, model.apply)
    r2, var, _ = tdvp_residual(variables, batch, e_loc, model.apply)
    assert var < 1e-8  # exact eigenstate → zero local-energy variance
    assert r2 < 1e-6


def test_residual_positive_for_generic_model():
    dof = 2
    model = make_ho_model()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, dof)))
    ham = HarmonicOscillatorHamiltonian(omega=1.0)
    # M = 512 > P (~337) so the full QGT is SPD and well-conditioned at ε=1e-4.
    batch = sample_psi2(model.apply, params, jax.random.PRNGKey(2), n_chains=512, dof=dof)
    e_loc = compute_local_energy(ham, params, batch, model.apply)
    r2, var, captured = tdvp_residual(params, batch, e_loc, model.apply, regularization=1e-4)
    assert np.isfinite(captured) and np.isfinite(r2)
    assert var > 1e-6 and r2 >= 0.0 and captured >= -1e-8


def test_imaginary_time_step_reduces_energy():
    dof = 2
    model = make_ho_model()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, dof)))
    ham = HarmonicOscillatorHamiltonian(omega=1.0)
    key = jax.random.PRNGKey(3)

    def energy(p, batch):
        return float(jnp.mean(compute_local_energy(ham, p, batch, model.apply)))

    key, sk = jax.random.split(key)
    batch0 = sample_psi2(model.apply, params, sk, n_chains=512, dof=dof)
    e0 = energy(params, batch0)

    # Re-sample from the *current* |ψ_θ|² each step: SR ≡ imaginary-time TDVP descends the
    # variational energy only when the samples track the evolving wavefunction.
    for _ in range(5):
        key, sk = jax.random.split(key)
        batch = sample_psi2(model.apply, params, sk, n_chains=512, dof=dof)
        e_loc = compute_local_energy(ham, params, batch, model.apply)
        params = imaginary_time_step(params, batch, e_loc, model.apply, dt=0.05, regularization=1e-3)

    key, sk = jax.random.split(key)
    batch_f = sample_psi2(model.apply, params, sk, n_chains=512, dof=dof)
    assert energy(params, batch_f) < e0  # imaginary-time TDVP descends toward the ground state
