"""Welch t-test, snapshot policy and the best() metric factories.

The weight-masking and multi-seed tests that used to live here went with their
modules (vmc/masking.py, vmc/multi_seed.py) in the phase-1 deletions.
"""

import numpy as np
import optax
import pytest
from conftest import make_ho_model

from qvarnet import train
from qvarnet.callbacks import SnapshotCallback
from qvarnet.config.training_setup import TrainingConfig
from qvarnet.diagnostics import welch_t_test
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.vmc.train_result import e_plus_sigma_metric, v_score_metric


def _sampler():
    return {"step_size": 0.5, "chain_length": 100, "thermalization_steps": 20, "thinning_factor": 2}


# ---------- Welch ----------


def test_welch_same_vs_different():
    # near-identical means/spread → not significant (deterministic, not a random null draw)
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = np.array([1.2, 1.9, 3.1, 3.8, 5.2, 5.9])
    assert welch_t_test(a, b)["p"] > 0.2
    c = a + 10.0  # large, clear shift in mean
    assert welch_t_test(a, c)["p"] < 1e-3


# ---------- snapshot policy ----------


def test_snapshot_best_k(tmp_path):
    snap = SnapshotCallback(policy="best_k", k=2, metric="energy")
    result = train(
        shape=(32, 2),
        model=make_ho_model(),
        optimizer=optax.adam(1e-2),
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(n_epochs=40, rng_seed=0, checkpoint_path=str(tmp_path)),
        sampler_params=_sampler(),
        callbacks=[snap],
    )
    assert len(snap.snapshots) == 2
    assert snap.best_params() is not None
    energies = [s.energy for s in result.history]
    assert min(s["metric"] for s in snap.snapshots) == pytest.approx(min(energies))


# ---------- best() metric factories ----------


def test_best_metric_factories(tmp_path):
    result = train(
        shape=(32, 2),
        model=make_ho_model(),
        optimizer=optax.adam(1e-2),
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(n_epochs=30, rng_seed=0, checkpoint_path=str(tmp_path)),
        sampler_params=_sampler(),
    )
    top = result.best(n=1, metric=e_plus_sigma_metric(alpha=0.5))[0]
    assert hasattr(top, "energy")
    vtop = result.best(n=1, metric=v_score_metric(n_particles=2))[0]
    assert hasattr(vtop, "energy")
