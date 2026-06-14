"""Step 7: the dashboard must build and save a PNG from a real run without error."""

import optax
from conftest import make_ho_model

from qvarnet.config.training_setup import TrainingConfig
from qvarnet.diagnostics import plot_dashboard
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.train import train


def test_dashboard_builds_and_saves(tmp_path):
    result = train(
        shape=(32, 2),
        model=make_ho_model(),
        optimizer=optax.adam(1e-2),
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(n_epochs=30, rng_seed=0, checkpoint_path=str(tmp_path)),
        sampler_params={
            "step_size": 0.5,
            "chain_length": 100,
            "thermalization_steps": 20,
            "thinning_factor": 2,
        },
    )
    out = tmp_path / "dashboard.png"
    fig = plot_dashboard(result, exact_energy=1.0, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert len(fig.axes) >= 6
