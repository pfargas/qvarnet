"""Frozen-parameter evaluation (vmc/evaluate.py) + the end-of-run summary.

The strongest check is exactness: for the analytic HO ground state the local energy is
constant, so evaluate() must return E = dof/2 with (near-)zero error regardless of the
sampler's autocorrelation.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from qvarnet import evaluate, evaluate_result
from qvarnet.config.training_setup import SamplingConfig, TrainingConfig
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.models import MLP
from qvarnet.models.exponential import LogAnalyticWavefunction
from qvarnet.vmc.evaluate import block_error

# NOT `from qvarnet import train`: the legacy shim module qvarnet/train.py shadows the
# re-exported function on the package as soon as any test imports the submodule.
from qvarnet.vmc.train import train

DOF = 3
CFG = SamplingConfig(step_size=0.5, chain_length=12, thermalization_steps=8,
                     thinning_factor=1, sampler="mh")


# ── block_error (the error-bar machinery, in isolation) ─────────────────────────────


def test_block_error_iid_matches_naive():
    rng = np.random.default_rng(0)
    series = rng.normal(size=4000)
    naive = series.std(ddof=1) / np.sqrt(len(series))
    assert block_error(series, n_blocks=20) == pytest.approx(naive, rel=0.5)


def test_block_error_detects_autocorrelation():
    """A strongly correlated series must get a *larger* error than the naive estimate."""
    rng = np.random.default_rng(1)
    x, out = 0.0, []
    for _ in range(4000):
        x = 0.99 * x + rng.normal() * np.sqrt(1 - 0.99**2)  # AR(1), tau ~ 100
        out.append(x)
    series = np.asarray(out)
    naive = series.std(ddof=1) / np.sqrt(len(series))
    assert block_error(series, n_blocks=20) > 3 * naive


def test_block_error_short_series_does_not_crash():
    assert block_error(np.arange(5.0), n_blocks=20) > 0


# ── evaluate(): exactness on the analytic ground state ──────────────────────────────


def test_evaluate_exact_ho_ground_state():
    """logψ = -x²/2 (alpha=1/2) is exact for H = -½Δ + ½x²: E_loc ≡ dof/2, error ≈ 0."""
    model = LogAnalyticWavefunction()
    params = {"params": {"alpha": jnp.array(0.5)}}
    ev = evaluate(model, params, HarmonicOscillatorHamiltonian(), shape=(64, DOF),
                  sampling_config=CFG, n_epochs=20, rng_seed=0)
    assert ev.energy == pytest.approx(DOF / 2, abs=1e-4)
    assert ev.error < 1e-4 and ev.sigma < 1e-3   # constant local energy
    assert 0.0 < ev.acceptance < 1.0
    assert ev.n_samples == 20 * 64 * (12 - 8)


def test_evaluate_variational_model_is_above_ground_state():
    """A wrong alpha is still a valid trial state: E > E0, with a finite error bar."""
    model = LogAnalyticWavefunction()
    params = {"params": {"alpha": jnp.array(0.8)}}
    ev = evaluate(model, params, HarmonicOscillatorHamiltonian(), shape=(64, DOF),
                  sampling_config=CFG, n_epochs=40, rng_seed=0)
    assert ev.energy > DOF / 2
    assert ev.error > 0 and ev.sigma > 0
    assert len(ev.energies) == 40


def test_evaluate_rejects_parallel_tempering():
    with pytest.raises(NotImplementedError):
        evaluate(LogAnalyticWavefunction(), {"params": {"alpha": jnp.array(0.5)}},
                 HarmonicOscillatorHamiltonian(), shape=(8, DOF),
                 sampling_config=SamplingConfig(step_size=0.5, chain_length=4,
                                                thermalization_steps=2, thinning_factor=1,
                                                sampler="pt"),
                 n_epochs=2)


# ── the notebook flow: train -> evaluate_result -> summary ──────────────────────────


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    ckpt = tmp_path_factory.mktemp("run")
    model = MLP(hidden=[4], output_dim=1)
    ham = HarmonicOscillatorHamiltonian()
    result = train(
        shape=(32, DOF),
        model=model,
        optimizer=__import__("optax").adam(1e-2),
        hamiltonian=ham,
        training_config=TrainingConfig(n_epochs=10, rng_seed=0, print_summary=False,
                                       checkpoint_path=str(ckpt)),
        sampler_params=CFG,
        select="std",
        k_best=3,
    )
    return result, model, ham


def test_evaluate_result_uses_best_snapshot_and_factor(tiny_run):
    result, model, ham = tiny_run
    ev = evaluate_result(result, model=model, hamiltonian=ham, shape=(32, DOF),
                         sampling_config=CFG, sample_factor=2.0, rng_seed=1)
    assert ev.n_epochs == 2 * len(result.history)
    # a trained-for-10-epochs MLP is a rough trial state, but still variational
    assert ev.energy > DOF / 2 - 3 * ev.error
    assert np.isfinite(ev.error) and ev.error > 0


def test_summary_returns_expected_fields(tiny_run, capsys):
    result, _, _ = tiny_run
    out = result.summary(print_report=True)
    printed = capsys.readouterr().out
    assert out["epochs_ran"] == len(result.history)
    assert out["n_parameters"] and out["n_parameters"] > 0
    assert out["wall_time_s"] > 0
    assert 0.0 < out["acceptance_tail"] < 1.0
    assert out["n_snapshots_kept"] == 3
    assert {"energy", "error_of_mean", "std"} <= set(out["final"])
    assert "best_snapshot" in out
    assert "training summary" in printed and "best epoch" in printed


def test_train_prints_summary_by_default(tmp_path, capsys):
    train(
        shape=(16, DOF),
        model=LogAnalyticWavefunction(),
        optimizer=__import__("optax").adam(1e-3),
        hamiltonian=HarmonicOscillatorHamiltonian(),
        training_config=TrainingConfig(n_epochs=3, rng_seed=0,
                                       checkpoint_path=str(tmp_path)),
        sampler_params=CFG,
    )
    assert "training summary" in capsys.readouterr().out


def test_warm_walkers_defaults_true():
    assert TrainingConfig(n_epochs=1).warm_walkers is True
