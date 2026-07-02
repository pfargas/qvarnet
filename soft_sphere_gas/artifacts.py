"""Per-run output artifacts, written into the run directory that runq hands the target.

runq owns the run-dir naming/layout (``outputs/runs/<label>/`` next to the DB); this module
only knows *what* a soft-sphere run leaves behind:

* ``meta.json``         — identity, units note, full hyper-params, box L, and the paper
                          benchmarks (4πx lower bound, Lee-Yang, Eq.31 upper bound).
* ``history.csv``       — per epoch: engine-unit energy/std/error + paper-unit E/N and σ/N,
                          acceptance, step_size, cm_*, wall_time.
* ``verdict.json``      — the full three-referee ``result.diagnose()`` dict.
* ``best_params.msgpack`` — the best ``snapshot_frac`` of epochs by ``select`` (default σ_E):
                          ``{"steps", "metrics", "select", "params": [pytree, ...]}`` (best
                          first). Reload with :func:`load_params`.

Nothing here references an absolute path or a live JAX device, so the whole ``outputs/``
tree stays copyable between a cluster and a PC (see ``RUNNING.md``).
"""

from __future__ import annotations

import csv
import json
import math
import os

from dilute_gas import lee_yang_energy_per_particle, to_paper_energy
from flax import serialization


def write_artifacts(run_dir: str, result) -> None:
    """Write meta/history/verdict/params for one run into ``run_dir``."""
    os.makedirs(run_dir, exist_ok=True)
    _write_meta(os.path.join(run_dir, "meta.json"), result)
    _write_history(os.path.join(run_dir, "history.csv"), result)
    with open(os.path.join(run_dir, "verdict.json"), "w") as fh:
        json.dump(_json_safe(result.verdict), fh, indent=2)
    _write_params(os.path.join(run_dir, "best_params.msgpack"), result)


def _write_meta(path: str, result) -> None:
    x, N = result.x, result.N
    meta = {
        "run_id": os.path.basename(os.path.dirname(path)),
        "units": "paper (hbar^2/2m=1, a=1, energies in hbar^2/2m a^2); see CONVENTIONS.md",
        "potential": {"label": result.potential.label, "R": result.potential.R,
                      "V0_paper": result.potential.V0_paper},
        "x": x, "N": N, "seed": result.seed, "L": result.L,
        "hyper_params": result.hp.to_dict(),
        "results_paper": {
            "e_per_n": result.e_per_n, "err_per_n": result.err_per_n,
            "sigma_e_per_n": result.sigma_e_per_n, "acceptance": result.acceptance,
            "passed": bool(result.passed),
        },
        "benchmarks_paper": {
            "lower_bound_4pix": 4 * math.pi * x,
            "lee_yang": lee_yang_energy_per_particle(x),
            "upper_bound_eq31": result.upper_bound,
        },
    }
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)


def _write_history(path: str, result) -> None:
    rows = result.history_rows
    if not rows:
        return
    N = result.N
    fields = ["epoch", "energy_engine", "std_engine", "error_of_mean_engine",
              "e_per_n_paper", "sigma_per_n_paper", "acceptance", "step_size",
              "grad_norm", "theta_ratio", "cm_mean", "cm_std", "wall_time"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            e_eng, s_eng = r.get("energy_engine"), r.get("std_engine")
            out = dict(r)
            # add paper-unit per-particle columns (the factor-of-two bridge + /N)
            out["e_per_n_paper"] = to_paper_energy(e_eng) / N if e_eng is not None else None
            out["sigma_per_n_paper"] = to_paper_energy(s_eng) / N if s_eng is not None else None
            w.writerow({k: out.get(k) for k in fields})


def _write_params(path: str, result) -> None:
    snaps = result.snapshots
    if not snaps:
        return
    payload = {
        "select": result.hp.select,
        "steps": [int(s["step"]) for s in snaps],
        "metrics": [float(s["metric"]) for s in snaps],
        "params": [s["params"] for s in snaps],  # list of host pytrees, best (lowest metric) first
    }
    with open(path, "wb") as fh:
        fh.write(serialization.msgpack_serialize(payload))


def load_params(path: str) -> dict:
    """Reload a ``best_params.msgpack`` written by :func:`_write_params`.

    Returns ``{"select", "steps", "metrics", "params": [pytree, ...]}`` with numpy arrays. Feed
    any ``params[i]`` straight into your reconstructed ``model.apply({"params": ...}, x)``.
    """
    with open(path, "rb") as fh:
        return serialization.msgpack_restore(fh.read())


def _json_safe(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(type(v).__name__)
    return out
