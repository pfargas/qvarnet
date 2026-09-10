"""The per-epoch update must compile once, not once per epoch.

`train()`'s speed rests on one jitted graph reused across every epoch. The usual
way to lose that is to make something that should be static (a frozen config, the
sampler) vary per call, or to pass a traced value where a static one is expected --
which silently retraces and costs orders of magnitude, without failing any other
test.

This counts real tracings, so it catches that directly rather than by timing.
"""

import optax
import pytest
from conftest import make_ho_model

from qvarnet.config.training_setup import SamplingConfig, TrainingConfig
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.samplers import Metropolis, OrderedMetropolis, ParticleSubsetMove
from qvarnet.vmc.loop import run_loop
from qvarnet.vmc.setup import build_context
from qvarnet.vmc.step import make_update_fn
from qvarnet.vmc.warmup import warm_up

N_EPOCHS = 6


def _context(tmp_path, sampler):
    return build_context(
        shape=(16, 3),
        model=make_ho_model(),
        optimizer=optax.adam(1e-2),
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(
            n_epochs=N_EPOCHS,
            rng_seed=0,
            checkpoint_path=str(tmp_path),
            is_update_step_size=True,
            print_summary=False,
        ),
        sampler_params=SamplingConfig(
            step_size=0.5, chain_length=11, thermalization_steps=10, thinning_factor=1
        ),
        sampler=sampler,
    )


@pytest.mark.parametrize(
    "sampler",
    [
        Metropolis(),
        OrderedMetropolis(proposal=ParticleSubsetMove(n_move=1, n_dim=1)),
    ],
    ids=["metropolis", "ordered"],
)
def test_update_compiles_once(tmp_path, sampler):
    ctx = _context(tmp_path, sampler)
    warm_up(ctx)

    update_fn = make_update_fn(ctx)
    run_loop(ctx, update_fn)

    # One cache entry means one graph was built and every later epoch reused it.
    compilations = update_fn._cache_size()
    assert compilations == 1, (
        f"the epoch update compiled {compilations} times over {N_EPOCHS} epochs; "
        "an argument that should keep a stable signature is varying per call "
        "(a Python scalar where the update returns a device array is the usual cause)"
    )


def test_history_length_matches_epochs(tmp_path):
    """A retrace guard is worthless if the loop silently ran the wrong number of epochs."""
    ctx = _context(tmp_path, Metropolis())
    warm_up(ctx)
    result = run_loop(ctx, make_update_fn(ctx))
    assert len(result.history) == N_EPOCHS
