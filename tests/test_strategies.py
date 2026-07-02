"""Step 6 guards: Welch t-test, weight masking, multi-seed R̂, snapshot policy, best() metrics."""

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import make_ho_model

from qvarnet.callbacks import SnapshotCallback
from qvarnet.config.training_setup import TrainingConfig
from qvarnet.diagnostics import welch_t_test
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.train import train
from qvarnet.vmc.masking import magnitude_mask, mask_updates, random_update_masking
from qvarnet.vmc.multi_seed import multi_seed_run
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


# ---------- masking ----------


def test_mask_updates_freezes():
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    tx = mask_updates(optax.sgd(0.1), {"w": jnp.array([0.0, 0.0, 0.0])})  # freeze all
    state = tx.init(params)
    updates, _ = tx.update({"w": jnp.array([1.0, 1.0, 1.0])}, state, params)
    assert jnp.allclose(updates["w"], 0.0)


def test_magnitude_mask_sparsity():
    params = {"w": jnp.arange(1.0, 101.0)}  # 100 weights
    mask = magnitude_mask(params, sparsity=0.5)
    assert abs(float(jnp.sum(mask["w"])) - 50.0) <= 1.0  # ~half kept


def test_random_update_masking_runs():
    params = {"w": jnp.ones((50,))}
    tx = random_update_masking(optax.sgd(0.1), drop_prob=0.5, seed=0)
    state = tx.init(params)
    updates, state = tx.update({"w": jnp.ones((50,))}, state, params)
    frac_zero = float(jnp.mean(updates["w"] == 0.0))
    assert 0.2 < frac_zero < 0.8  # ~50% of updates dropped
    assert jnp.all(jnp.isfinite(updates["w"]))


# ---------- multi-seed ----------


def test_multi_seed_run_rhat(tmp_path):
    out = multi_seed_run(
        [0, 1, 2],
        shape=(32, 2),
        model=make_ho_model(),
        optimizer=optax.adam(2e-2),
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(n_epochs=80, rng_seed=0, checkpoint_path=str(tmp_path),
                                       # calibrated for cold-restart sampling (engine default is now True)
                                       warm_walkers=False),
        sampler_params=_sampler(),
    )
    assert len(out["results"]) == 3 and len(out["tail_means"]) == 3
    assert np.isfinite(out["rhat"]) and isinstance(out["seed_safe"], bool)
    assert out["rhat"] < 1.5  # HO seeds should broadly agree


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
