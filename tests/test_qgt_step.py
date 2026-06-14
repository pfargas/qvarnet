"""Part A guard: SR-as-preconditioner must advance state.step and honour optax
LR schedules on the QGT path (regression for the §0 bug where the QGT branch did
state.replace(params=...) and never incremented step / ignored schedules)."""

import optax
from conftest import make_ho_model

from qvarnet.callbacks.base import Callback
from qvarnet.config.training_setup import TrainingConfig
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.train import train


class _StepRecorder(Callback):
    def __init__(self):
        self.steps = []

    def on_step_end(self, step, state, metrics):
        self.steps.append(int(state.step))
        return False


def _run(n_epochs, use_qgt, optimizer, callbacks, checkpoint_path):
    return train(
        shape=(32, 2),
        model=make_ho_model(),
        optimizer=optimizer,
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(
            n_epochs=n_epochs, rng_seed=0, use_qgt=use_qgt, checkpoint_path=checkpoint_path
        ),
        sampler_params={
            "step_size": 0.5,
            "chain_length": 100,
            "thermalization_steps": 20,
            "thinning_factor": 2,
        },
        callbacks=callbacks,
    )


def test_qgt_path_increments_step(tmp_path):
    rec = _StepRecorder()
    _run(2, use_qgt=True, optimizer=optax.sgd(1e-2), callbacks=[rec], checkpoint_path=str(tmp_path))
    # post-update state.step seen by the callback: 1 after epoch 0, 2 after epoch 1.
    assert rec.steps == [1, 2], rec.steps


def test_adam_path_increments_step(tmp_path):
    rec = _StepRecorder()
    _run(
        2, use_qgt=False, optimizer=optax.adam(1e-2), callbacks=[rec], checkpoint_path=str(tmp_path)
    )
    assert rec.steps == [1, 2], rec.steps
