"""Known-answer sanity check: a single particle in a 1D harmonic trap (omega=1) has
exact ground-state energy E0 = 0.5. VMC must approach it. (Precursor to the §7
discrete-VMC exact testbed.)"""

import numpy as np
import optax
from conftest import make_ho_model

from qvarnet.config.training_setup import TrainingConfig
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.train import train


def test_single_particle_ho_converges_to_half(tmp_path):
    result = train(
        shape=(128, 1),
        model=make_ho_model(),
        optimizer=optax.adam(1e-2),
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(n_epochs=300, rng_seed=0, checkpoint_path=str(tmp_path)),
        sampler_params={
            "step_size": 0.6,
            "chain_length": 200,
            "thermalization_steps": 50,
            "thinning_factor": 2,
        },
    )
    energies = np.array([s.energy for s in result.history])
    tail = float(np.mean(energies[-30:]))
    assert energies[0] > tail, "energy did not decrease during training"
    assert abs(tail - 0.5) < 0.15, f"tail energy {tail:.4f} not near exact E0=0.5"
