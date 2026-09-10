"""Wall-time-per-epoch gate for the restructure.

`train()` is fast and must stay fast: the decomposition phase moves code around a
jit boundary, and the one thing that would silently ruin it is an extra host sync
or a lost `donate`/static-arg. This records median seconds per epoch for a fixed
workload and compares against a committed baseline.

    uv run python scripts/bench_epoch.py --update    # record the baseline
    uv run python scripts/bench_epoch.py             # check against it

The baseline is keyed by JAX platform (cpu/gpu), so a GPU baseline is never
compared against a CPU run. Timings exclude the first epochs (JIT compilation).

Defaults to **CPU**: this box has a single 8 GB GPU that research sweeps occupy,
and a benchmark sharing it both perturbs the timing and risks OOM-ing the run.
Pass ``--gpu`` deliberately, on an idle GPU, if you want a device baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

BASELINE = Path(__file__).resolve().parents[1] / "tests" / "data" / "bench_baseline.json"
TOLERANCE = 1.15  # fail if slower than baseline * this
N_EPOCHS = 60
WARMUP_EPOCHS = 10  # discarded: JIT compile + cache warm


def _bench() -> dict:
    import jax
    import optax

    from qvarnet import train
    from qvarnet.physics.boundaries import NoBoundary
    from qvarnet.core.coords import LabCoords
    from qvarnet import TrainingConfig
    from qvarnet.physics.hamiltonian.continuous import HarmonicOscillatorHamiltonian
    from qvarnet.ansatz.compose import LogWavefunction
    from qvarnet.ansatz.envelopes import GaussianEnvelope
    from qvarnet.ansatz.mlp import MLP

    model = LogWavefunction(
        network=MLP(hidden=[64, 64], output_dim=1),
        transform=NoBoundary(),
        envelope=GaussianEnvelope(),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.perf_counter()
        result = train(
            shape=(1024, 4),
            model=model,
            optimizer=optax.adam(1e-3),
            hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
            training_config=TrainingConfig(
                n_epochs=N_EPOCHS,
                rng_seed=0,
                checkpoint_path=tmpdir,
                is_update_step_size=True,
                print_summary=False,
            ),
            sampler_params={
                "step_size": 0.5,
                "chain_length": 51,
                "thermalization_steps": 20,
                "thinning_factor": 1,
            },
            coord_mode=LabCoords(),
        )
        total = time.perf_counter() - t0

    per_epoch = np.asarray(result.history.get("wall_time"), dtype=float)[WARMUP_EPOCHS:]
    return {
        "platform": jax.default_backend(),
        "median_s_per_epoch": float(np.median(per_epoch)),
        "p90_s_per_epoch": float(np.percentile(per_epoch, 90)),
        "total_s": float(total),
        "n_epochs": N_EPOCHS,
        "n_timed": int(per_epoch.size),
    }


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="record this run as the baseline")
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="benchmark on the GPU (only on an idle device -- see the module docstring)",
    )
    args = ap.parse_args()

    if not args.gpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    now = _bench()
    plat = now["platform"]
    print(
        f"[{plat}] median {now['median_s_per_epoch'] * 1e3:.2f} ms/epoch  "
        f"p90 {now['p90_s_per_epoch'] * 1e3:.2f} ms  "
        f"({now['n_timed']} timed epochs, {now['total_s']:.1f}s total)"
    )

    baselines = json.loads(BASELINE.read_text()) if BASELINE.is_file() else {}

    if args.update:
        baselines[plat] = now
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")
        print(f"baseline recorded for {plat} -> {BASELINE}")
        return 0

    if plat not in baselines:
        print(f"no baseline for platform {plat!r}; record one with --update")
        return 0  # informational, not a failure

    ref = baselines[plat]["median_s_per_epoch"]
    got = now["median_s_per_epoch"]
    ratio = got / ref
    verdict = "OK" if ratio <= TOLERANCE else "REGRESSION"
    print(
        f"{verdict}: {got * 1e3:.2f} ms vs baseline {ref * 1e3:.2f} ms "
        f"({ratio:.2f}x, tolerance {TOLERANCE:.2f}x)"
    )
    return 0 if ratio <= TOLERANCE else 1


if __name__ == "__main__":
    raise SystemExit(main())
