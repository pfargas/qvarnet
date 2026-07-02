"""Step 4 guards. The estimators are validated against synthetic AR(1) chains with known
integrated autocorrelation time τ = (1+φ)/(1-φ), plus pass/fail behaviour on drift, then the
three-referee verdict on a real HO run."""

import numpy as np
import optax
import pytest
from flax import linen as nn

from qvarnet.boundaries import NoBoundary
from qvarnet.config.training_setup import TrainingConfig
from qvarnet.diagnostics import (
    StationarityStopper,
    ess,
    geweke_z,
    heidelberger_welch_t,
    iat_geyer,
    is_stationary,
    split_rhat,
    v_score,
)
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.mlp import MLP
from qvarnet.train import train


def ar1(phi, n, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x + drift * np.arange(n)


# ---------- IAT / ESS against the exact AR(1) value ----------


@pytest.mark.parametrize("phi", [0.5, 0.8])
def test_iat_recovers_ar1_tau(phi):
    tau_exact = (1 + phi) / (1 - phi)
    tau = iat_geyer(ar1(phi, 40000, seed=1))
    assert abs(tau - tau_exact) / tau_exact < 0.2, f"phi={phi}: {tau:.2f} vs {tau_exact:.2f}"


def test_ess_consistent_with_iat():
    x = ar1(0.8, 20000, seed=2)
    assert ess(x) == pytest.approx(len(x) / iat_geyer(x), rel=1e-6)


# ---------- stationarity referees: pass on AR(1), fail on AR(1)+drift ----------


def test_referees_pass_on_stationary():
    x = ar1(0.7, 8000, seed=3)
    assert abs(geweke_z(x)) < 3.0
    assert abs(heidelberger_welch_t(x)) < 2.0
    assert is_stationary(x)


def test_referees_fail_on_drift():
    x = ar1(0.7, 8000, seed=3, drift=0.01)  # slow upward trend
    assert not is_stationary(x)
    assert abs(heidelberger_welch_t(x)) > 2.0


# ---------- split-R̂ ----------


def test_split_rhat_identical_vs_shifted():
    rng = np.random.default_rng(4)
    chains = rng.standard_normal((8, 500))
    assert split_rhat(chains) < 1.1
    shifted = chains.copy()
    shifted[:4] += 5.0  # half the chains in a different mode
    assert split_rhat(shifted) > 1.3


# ---------- StationarityStopper ----------


def _drive_stopper(stopper, trace):
    for step, e in enumerate(trace):
        if stopper.on_step_end(step, None, {"energy": float(e)}):
            return step
    return None


def test_stopper_stops_on_stationary():
    rng = np.random.default_rng(5)
    trace = 1.0 + 0.05 * rng.standard_normal(600)  # flat + noise
    stop = StationarityStopper(warmup=100, check_every=20, window=150, patience=2)
    stopped = _drive_stopper(stop, trace)
    assert stopped is not None and stopped >= 100


def test_stopper_does_not_stop_on_drift():
    trace = np.linspace(5.0, 1.0, 600)  # steadily improving → never stationary
    stop = StationarityStopper(warmup=100, check_every=20, window=150, patience=2)
    assert _drive_stopper(stop, trace) is None


# ---------- the verdict on a real HO run ----------


def test_verdict_and_vscore_on_ho(tmp_path):
    model = LogWavefunction(
        network=MLP(hidden=[32, 32], output_dim=1, hidden_activation=nn.tanh),
        transform=NoBoundary(),
    )
    result = train(
        shape=(128, 1),
        model=model,
        optimizer=optax.adam(1e-2),
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(n_epochs=400, rng_seed=0, checkpoint_path=str(tmp_path),
                                       # calibrated for cold-restart sampling (engine default is now True)
                                       warm_walkers=False),
        sampler_params={
            "step_size": 0.6,
            "chain_length": 200,
            "thermalization_steps": 50,
            "thinning_factor": 2,
        },
    )
    v = result.diagnose(print_report=False)
    assert set(v) >= {"stationary", "at_mc_floor", "chains_mixed", "split_rhat", "passed"}
    assert abs(v["tail_energy"] - 0.5) < 0.15  # near exact E0
    assert v["chains_mixed"] is not None and np.shape(result.history[-1].E_chain) == (128,)
    assert v_score(result.history, n_particles=1) > 0.0
