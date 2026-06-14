"""Deterministic short HO run — regression baseline for the foundation refactor.

Run before/after the restructure (Parts A/B/C). The energy trace must match within
MC noise; the refactor must not change the physics.

    uv run python scripts/baseline_ho.py            # prints + writes baseline json
    uv run python scripts/baseline_ho.py --check     # compares against saved baseline

E0 for N particles in a 1D harmonic trap (omega=1): N * 0.5.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

# Pin to single-thread CPU so the energy trace is bit-reproducible (GPU atomic
# reductions are non-deterministic and the optimization trajectory is chaotic).
# This is a regression guard, not a perf benchmark — CPU/GPU agree on the physics.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import optax
from flax import linen as nn

from qvarnet.boundaries import NoBoundary
from qvarnet.config.training_setup import TrainingConfig
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.mlp import MLP
from qvarnet.train import train

N_PARTICLES = 4
N_DIM = 1
N_CHAINS = 64
N_EPOCHS = 60
SEED = 0
BASELINE_PATH = Path(__file__).parent / "baseline_ho.json"


def run():
    model = LogWavefunction(
        network=MLP(hidden=[32, 32], output_dim=1, hidden_activation=nn.tanh),
        transform=NoBoundary(),
    )
    result = train(
        shape=(N_CHAINS, N_PARTICLES * N_DIM),
        model=model,
        optimizer=optax.adam(1e-2),
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(
            n_epochs=N_EPOCHS, rng_seed=SEED, checkpoint_path=tempfile.mkdtemp()
        ),
        sampler_params={
            "step_size": 0.5,
            "chain_length": 200,
            "thermalization_steps": 50,
            "thinning_factor": 2,
        },
    )
    energies = [float(s.energy) for s in result.history]
    return energies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare against saved baseline")
    args = ap.parse_args()

    energies = run()
    e0 = N_PARTICLES * 0.5
    print(
        f"epochs={len(energies)}  E_first={energies[0]:.6f}  E_last={energies[-1]:.6f}  "
        f"E0_exact={e0:.3f}"
    )

    if args.check:
        saved = json.loads(BASELINE_PATH.read_text())["energies"]
        diff = max(abs(a - b) for a, b in zip(saved, energies))
        print(f"max|Δenergy| vs baseline = {diff:.3e}")
        assert diff < 1e-4, f"REGRESSION: energy trace drifted by {diff:.3e}"
        print("OK — energy trace matches baseline within 1e-4")
    else:
        BASELINE_PATH.write_text(json.dumps({"energies": energies}, indent=2))
        print(f"baseline written to {BASELINE_PATH}")


if __name__ == "__main__":
    main()
