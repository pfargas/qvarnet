"""Bit-identical regression net for the restructure.

Every phase of the restructure must reproduce these traces *exactly*. Statistical
agreement is not enough: this is a numerical code, and an exact match is the only
check that distinguishes a refactor bug from a legitimate change.

The scenarios are chosen to cover all three sampler call sites in ``train()``,
which are precisely what the sampler-protocol and ``train()`` decomposition phases
rewrite:

  ho_adam           plain MH  + adaptive step size + block-adaptive warmup
  rods_ordered      1d-ordered MH + particle-subset proposal + plain warmup
  rods_ordered_warm 1d-ordered MH + block-adaptive warmup (the ordered branch)

Regenerate ONLY when a trace change is deliberate and understood:

    uv run python tests/test_golden_trace.py --update
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import optax
import pytest
from conftest import make_ho_model

from qvarnet.boundaries import NoBoundary
from qvarnet.config.coord_mode import LabCoords
from qvarnet.config.training_setup import ChainInitAndWarmupConfig, TrainingConfig
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.envelopes import GaussianEnvelope
from qvarnet.models.mlp import MLP
from qvarnet.train import train

GOLDEN_DIR = Path(__file__).parent / "data"
N_EPOCHS = 40
TRACKED = ("energy", "std", "error_of_mean", "step_size")


def _rods_model():
    return LogWavefunction(
        network=MLP(hidden=[32, 32], output_dim=1),
        transform=NoBoundary(),
        envelope=GaussianEnvelope(init=0.5),
        jastrow=None,
    )


def _run(scenario: str, tmpdir: str):
    """One fixed-seed training run. Must stay byte-stable across the restructure."""
    if scenario == "ho_adam":
        return train(
            shape=(128, 1),
            model=make_ho_model(),
            optimizer=optax.adam(1e-2),
            hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
            training_config=TrainingConfig(
                n_epochs=N_EPOCHS,
                rng_seed=0,
                checkpoint_path=tmpdir,
                is_update_step_size=True,
                print_summary=False,
            ),
            initial_chain_config=ChainInitAndWarmupConfig(
                warmup_steps=50,
                warmup_step_size=0.5,
                warmup_adapt_step_size=True,
                warmup_n_blocks=5,
            ),
            sampler_params={
                "step_size": 0.6,
                "chain_length": 21,
                "thermalization_steps": 20,
                "thinning_factor": 1,
            },
            coord_mode=LabCoords(),
        )

    if scenario in ("rods_ordered", "rods_ordered_warm"):
        return train(
            shape=(128, 5),
            model=_rods_model(),
            # lr 3e-4, not the notebook's 3e-3: at 3e-3 this ansatz runs away to
            # E ~ -29 within 40 epochs (jastrow=None has no impenetrability node,
            # so the wedge-restricted energy has no lower bound). A golden trace
            # must not sit on a numerical cliff -- see docs/notes/.
            optimizer=optax.adam(3e-4),
            hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
            training_config=TrainingConfig(
                n_epochs=N_EPOCHS,
                rng_seed=0,
                checkpoint_path=tmpdir,
                print_summary=False,
            ),
            initial_chain_config=ChainInitAndWarmupConfig(
                warmup_steps=50,
                warmup_step_size=0.2,
                warmup_adapt_step_size=(scenario == "rods_ordered_warm"),
                warmup_n_blocks=5,
            ),
            sampler_params={
                "sampler": "1d-ordered",
                "step_size": 0.2,
                "chain_length": 21,
                "thermalization_steps": 20,
                "thinning_factor": 1,
                "proposal": ("particle-subset", {"n_move": 1, "n_dim": 1}),
            },
            coord_mode=LabCoords(),
        )

    raise ValueError(f"unknown scenario {scenario!r}")


def trace_of(scenario: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run(scenario, tmpdir)
    h = result.history
    out = {name: np.asarray(h.get(name), dtype=np.float64) for name in TRACKED}
    out["acceptance_rate"] = np.asarray(
        [float(np.mean(np.asarray(r["acceptance_rate"]))) for r in result.history],
        dtype=np.float64,
    )
    out["final_positions"] = np.asarray(result.final_positions, dtype=np.float64)
    out["final_step_size"] = np.asarray([result.final_step_size], dtype=np.float64)
    return out


SCENARIOS = ("ho_adam", "rods_ordered", "rods_ordered_warm")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_trace_is_bit_identical(scenario):
    path = GOLDEN_DIR / f"golden_{scenario}.npz"
    if not path.is_file():
        pytest.skip(f"no golden for {scenario}; generate with --update")

    golden = np.load(path)
    actual = trace_of(scenario)

    assert set(golden.files) == set(actual), (
        f"{scenario}: tracked quantities changed "
        f"(golden {sorted(golden.files)} vs {sorted(actual)})"
    )
    for name in sorted(actual):
        want, got = golden[name], actual[name]
        assert want.shape == got.shape, f"{scenario}/{name}: shape {got.shape} != {want.shape}"
        if np.array_equal(want, got):
            continue
        bad = int(np.sum(want != got))
        first = int(np.argmax((want != got).ravel()))
        raise AssertionError(
            f"{scenario}/{name}: trace diverged from the golden run.\n"
            f"  {bad}/{want.size} values differ; first at flat index {first}: "
            f"{want.ravel()[first]!r} -> {got.ravel()[first]!r}\n"
            f"  max |delta| = {np.max(np.abs(got - want)):.3e}\n"
            f"  A refactor must not change the numerics. If this change IS "
            f"intended, regenerate with `uv run python tests/test_golden_trace.py "
            f"--update` and say why in the commit."
        )


def test_ho_golden_is_physically_sane():
    """Guards the golden itself: a corrupt baseline would silently bless a bug."""
    golden = np.load(GOLDEN_DIR / "golden_ho_adam.npz")
    energy = golden["energy"]
    assert np.all(np.isfinite(energy)), "golden HO trace contains non-finite energies"
    assert energy[-1] < energy[0], "golden HO run did not descend"
    assert 0.4 < float(np.mean(energy[-10:])) < 1.2, "golden HO tail is far from E0 = 0.5"


def test_rods_golden_stays_ordered():
    """The ordered sampler's whole purpose: x0 < x1 < ... < x4 in every chain."""
    for scenario in ("rods_ordered", "rods_ordered_warm"):
        pos = np.load(GOLDEN_DIR / f"golden_{scenario}.npz")["final_positions"]
        assert np.all(np.diff(pos, axis=-1) >= 0), f"{scenario}: final walkers not ordered"


def test_rods_golden_is_bounded():
    """The golden must not be a runaway trace: FP noise near a divergence would make
    bit-identity impossible to hold across honest refactors."""
    for scenario in ("rods_ordered", "rods_ordered_warm"):
        energy = np.load(GOLDEN_DIR / f"golden_{scenario}.npz")["energy"]
        assert np.all(np.isfinite(energy)), f"{scenario}: non-finite energy in golden"
        assert energy.min() > -5.0, (
            f"{scenario}: golden trace is diverging (min E = {energy.min():.3f}). "
            "Pick a stable configuration before freezing it as a baseline."
        )


if __name__ == "__main__":
    import sys

    if "--update" not in sys.argv:
        print(__doc__)
        raise SystemExit(1)

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        trace = trace_of(scenario)
        path = GOLDEN_DIR / f"golden_{scenario}.npz"
        np.savez(path, **trace)
        e = trace["energy"]
        print(
            f"{scenario:20s} E: {e[0]:.6f} -> {e[-1]:.6f}  "
            f"acc {trace['acceptance_rate'][-1]:.3f}  "
            f"step {trace['final_step_size'][0]:.4f}  -> {path.name}"
        )
