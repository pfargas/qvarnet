"""Chunked local energy (vmc/chunking.py): same numbers as unchunked, and the memory-based choice."""

import jax
import numpy as np
import optax
import pytest
from conftest import make_ho_model

from qvarnet import TrainingConfig, train
from qvarnet.physics.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.vmc import chunking
from qvarnet.vmc.chunking import ChunkedHamiltonian

H = HarmonicOscillatorHamiltonian(omega=1.0)
SAMPLER = {"step_size": 0.6, "chain_length": 40, "thermalization_steps": 20, "thinning_factor": 1}  # 64 × 20 = 1280


def test_chunked_local_energy_equals_unchunked():
    model = make_ho_model()
    x = jax.random.normal(jax.random.PRNGKey(0), (1024, 1))
    params = model.init(jax.random.PRNGKey(1), x)
    e1 = H.local_energy(params, x, model.apply).reshape(-1)
    for n in (2, 8):
        en = ChunkedHamiltonian(H, n).local_energy(params, x, model.apply)
        assert en.shape == (1024,)
        np.testing.assert_allclose(en, e1, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        ChunkedHamiltonian(H, 3).local_energy(params, x, model.apply)
    assert ChunkedHamiltonian(H, 2).omega == H.omega  # attributes delegate to the inner Hamiltonian


def _train(tmp_path, max_chunks):
    return train(shape=(64, 1), model=make_ho_model(), optimizer=optax.adam(1e-2), hamiltonian=H,
                 training_config=TrainingConfig(n_epochs=5, rng_seed=0, checkpoint_path=str(tmp_path),
                                                max_chunks=max_chunks, print_summary=False),
                 sampler_params=SAMPLER)


def test_auto_choice_limits(tmp_path, monkeypatch):
    """Plenty of memory → 1 chunk; nothing fits even with 8 chunks → MemoryError."""
    chosen = []
    real = chunking.choose_n_chunks
    monkeypatch.setattr("qvarnet.vmc.driver.choose_n_chunks", lambda *a: chosen.append(real(*a)) or chosen[-1])
    monkeypatch.setattr(chunking, "_device_memory_limit", lambda: 10**15)
    _train(tmp_path / "a", max_chunks=8)
    assert chosen == [1]
    monkeypatch.setattr(chunking, "_device_memory_limit", lambda: 1)
    with pytest.raises(MemoryError):
        _train(tmp_path / "b", max_chunks=8)


def test_chunked_training_matches(tmp_path, monkeypatch):
    """Force 4 chunks through the driver: same energies as the unchunked run."""
    ref = _train(tmp_path / "ref", max_chunks=1)
    monkeypatch.setattr(chunking, "_device_memory_limit", lambda: 10**15)

    def force(ctx, update_fn, max_chunks):
        ctx.hamiltonian = ChunkedHamiltonian(ctx.hamiltonian, 4)
        return 4

    monkeypatch.setattr("qvarnet.vmc.driver.choose_n_chunks", force)
    got = _train(tmp_path / "chunk", max_chunks=8)
    e_ref = np.array([s.energy for s in ref.history])
    e_got = np.array([s.energy for s in got.history])
    np.testing.assert_allclose(e_got, e_ref, rtol=1e-5)
