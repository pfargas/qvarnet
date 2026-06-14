"""Step 3 (discrete mini-VMC) guards, validated against exact diagonalisation:

1. config orderings (jnp vs numpy) agree;
2. the connected-elements local energy is correct: full-sum ⟨E⟩ for random params equals
   ψ†Hψ/ψ†ψ from the dense ED Hamiltonian;
3. deterministic full-sum VMC converges to the exact TFIM ground-state energy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from qvarnet.boundaries import NoBoundary
from qvarnet.hamiltonian.discrete import TFIMHamiltonian
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.mlp import MLP
from qvarnet.utils.exact_diag import all_spin_configs, tfim_dense, tfim_ground_state_energy
from qvarnet.vmc.full_sum import all_spin_configs as jnp_configs
from qvarnet.vmc.full_sum import full_sum_energy, train_full_sum


def _model():
    return LogWavefunction(
        network=MLP(hidden=[32, 32], output_dim=1, hidden_activation=nn.tanh),
        transform=NoBoundary(),
    )


def test_config_orderings_match():
    n = 5
    assert np.allclose(np.asarray(jnp_configs(n)), all_spin_configs(n))


def test_local_energy_matches_dense_operator():
    n, J, h = 4, 1.0, 1.0
    model = _model()
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, n)))
    ham = TFIMHamiltonian(n_spins=n, pbc=True, J=J, h=h)

    # full-sum ⟨E⟩ via connected elements
    e_connected = float(full_sum_energy(model, params, ham))

    # reference: ψ†Hψ/ψ†ψ with the dense ED Hamiltonian, same config ordering
    configs = jnp_configs(n)
    logpsi = np.asarray(model.apply(params, configs).squeeze(-1))
    psi = np.exp(logpsi)
    H = tfim_dense(n, J=J, h=h, pbc=True)
    e_dense = float(psi @ H @ psi / (psi @ psi))

    assert abs(e_connected - e_dense) < 1e-3, f"{e_connected} vs {e_dense}"


def test_full_sum_converges_to_ed():
    n, J, h = 4, 1.0, 1.0
    e0 = tfim_ground_state_energy(n, J=J, h=h, pbc=True)

    model = _model()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, n)))
    ham = TFIMHamiltonian(n_spins=n, pbc=True, J=J, h=h)

    params, energies = train_full_sum(model, params, ham, optax.adam(2e-2), n_steps=600)
    final = energies[-1]

    # full-sum is exact (no MC noise) → variational bound holds tightly; expressivity-limited.
    assert final >= e0 - 1e-3, f"below variational bound: {final} < E0={e0}"
    assert final - e0 < 0.05, f"did not converge to ED E0: final={final:.4f}, E0={e0:.4f}"
