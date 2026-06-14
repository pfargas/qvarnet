"""Part C guards: MetricsHistory exposes per-chain energies, holds no params, and
keeps the documented ``[s.energy for s in result.history]`` access working."""

import numpy as np
import optax
import pytest
from conftest import make_ho_model

from qvarnet.config.training_setup import TrainingConfig
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.train import train
from qvarnet.vmc.metrics_history import MetricsHistory

N_CHAINS = 16
N_EPOCHS = 3


def _run(checkpoint_path):
    return train(
        shape=(N_CHAINS, 2),
        model=make_ho_model(),
        optimizer=optax.adam(1e-2),
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(
            n_epochs=N_EPOCHS, rng_seed=0, checkpoint_path=checkpoint_path
        ),
        sampler_params={
            "step_size": 0.5,
            "chain_length": 100,
            "thermalization_steps": 20,
            "thinning_factor": 2,
        },
    )


def test_history_is_metrics_history_no_params(tmp_path):
    result = _run(str(tmp_path))
    assert isinstance(result.history, MetricsHistory)
    assert len(result.history) == N_EPOCHS
    rec = result.history[-1]
    # lightweight record — must NOT carry params/grads/opt_state
    for forbidden in ("params", "grads", "opt_state", "tx"):
        with pytest.raises(AttributeError):
            getattr(rec, forbidden)


def test_per_chain_energies_shape(tmp_path):
    result = _run(str(tmp_path))
    rec = result.history[-1]
    assert np.shape(rec.E_chain) == (N_CHAINS,)
    assert np.shape(rec.acceptance_rate) == (N_CHAINS,)
    # stacked accessor: (n_epochs, n_chains)
    assert result.history.get("E_chain").shape == (N_EPOCHS, N_CHAINS)
    # per-chain mean of E_loc averages to the scalar energy
    assert float(np.mean(rec.E_chain)) == pytest.approx(float(rec.energy), rel=1e-5)


def test_backward_compat_access(tmp_path):
    result = _run(str(tmp_path))
    energies = [s.energy for s in result.history]  # documented usage
    assert len(energies) == N_EPOCHS
    assert result.history[-1].std >= 0.0
    best = result.best(n=1, metric="energy")[0]
    assert float(best.energy) == pytest.approx(min(energies))


def test_error_of_mean_is_naive_sem(tmp_path):
    result = _run(str(tmp_path))
    rec = result.history[-1]
    # naive SEM = sigma_E / sqrt(M); M = n_chains * n_eff, with
    # n_eff = (chain_length - thermalization)//thinning = (100-20)//2 = 40.
    M = N_CHAINS * 40
    assert float(rec.error_of_mean) == pytest.approx(float(rec.std) / np.sqrt(M), rel=1e-4)
