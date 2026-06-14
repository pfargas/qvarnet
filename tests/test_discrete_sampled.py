"""Step 3 follow-up: the single-spin-flip MCMC path.

1. Sampler correctness: for fixed params the MCMC-estimated energy matches the exact
   full-sum energy (the flip kernel samples |ψ|²).
2. Sampled discrete VMC trains to the exact TFIM ground-state energy (via the existing
   compute_step / SR machinery — the §7 factorization).
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from qvarnet.boundaries import NoBoundary
from qvarnet.geometry.qgt import QGTConfig
from qvarnet.hamiltonian.discrete import TFIMHamiltonian
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.mlp import MLP
from qvarnet.samplers.discrete import sample_spins
from qvarnet.utils.exact_diag import tfim_ground_state_energy
from qvarnet.vmc.discrete_train import train_discrete
from qvarnet.vmc.full_sum import full_sum_energy
from qvarnet.vmc.probability import build_prob_fn


def _model():
    return LogWavefunction(
        network=MLP(hidden=[32, 32], output_dim=1, hidden_activation=nn.tanh),
        transform=NoBoundary(),
    )


def test_spin_sampler_matches_full_sum():
    n = 6
    model = _model()
    params = model.init(jax.random.PRNGKey(3), jnp.ones((1, n)))
    ham = TFIMHamiltonian(n_spins=n, pbc=True, J=1.0, h=1.0)

    e_exact = float(full_sum_energy(model, params, ham))

    prob_fn = build_prob_fn(model.apply)
    n_chains = 256
    init = jnp.where(
        jax.random.bernoulli(jax.random.PRNGKey(4), 0.5, (n_chains, n)), 1.0, -1.0
    )
    batch, _, acc = sample_spins(
        jax.random.PRNGKey(5), prob_fn, params, init, n_chains, n, 600, 200, 4
    )
    e_sampled = float(jnp.mean(ham.local_energy(params, batch, model.apply)))

    assert 0.0 < float(jnp.mean(acc)) <= 1.0
    assert abs(e_sampled - e_exact) < 0.1, f"sampled {e_sampled:.4f} vs exact {e_exact:.4f}"


def test_train_discrete_converges_to_ed():
    n = 4
    e0 = tfim_ground_state_energy(n, J=1.0, h=1.0, pbc=True)

    result, _state = train_discrete(
        _model(),
        TFIMHamiltonian(n_spins=n, pbc=True, J=1.0, h=1.0),
        optimizer=None,  # overridden by SGD(qgt lr) on the QGT path
        n_chains=128,
        n_epochs=100,
        chain_length=200,
        burn_in=50,
        thinning=2,
        use_qgt=True,
        qgt_config=QGTConfig(solver="cholesky", learning_rate=0.05, regularization=1e-3),
    )
    energies = np.array([s.energy for s in result.history])
    tail = float(np.mean(energies[-20:]))
    # per-chain energies are present (the diagnostics input)
    assert np.shape(result.history[-1].E_chain) == (128,)
    assert tail < energies[0], "sampled discrete VMC did not reduce the energy"
    assert abs(tail - e0) < 0.1, f"sampled tail {tail:.4f} not near ED E0={e0:.4f}"
