"""Per-run output artifacts — the portable, self-contained layout under ``outputs/``.

A sweep produces one **run directory** per ``(potential, x, N, seed, hp)`` under
``outputs/runs/<run_id>/``, alongside the index DB ``outputs/soft_sphere.db``. The whole
``outputs/`` tree is self-contained and copyable between a cluster and a PC (see ``RUNNING.md``):
nothing here references an absolute path or a live JAX device.

``run_id`` (the label)
----------------------
``<potential>_x<gas-param>_N<particles>_s<seed>_<hp8>`` — e.g. ``SS10_x1.000e-04_N64_s0_3f9a1c2b``.
The physics axes (potential / x / N / seed) are human-readable; ``hp8`` is the first 8 hex of a
hash of the *full* hyperparameter dict, so two runs that differ only in solver settings get
distinct dirs. The full HP is recorded in ``meta.json`` and the DB row, so the hash never has to
be inverted.

Run-dir contents
----------------
* ``meta.json``         — identity, units note, full HP, box L, and the paper benchmarks
                          (4πx lower bound, Lee-Yang, Eq.31 upper bound) at this point.
* ``history.csv``       — per epoch: engine-unit energy/std/error + paper-unit E/N and σ/N,
                          acceptance, step_size, cm_*, wall_time.
* ``verdict.json``      — the full three-referee ``result.diagnose()`` dict.
* ``best_params.msgpack`` — the best ``snapshot_frac`` of epochs by ``select`` (default σ_E):
                          ``{"steps", "metrics", "select", "params": [pytree, ...]}`` (best first).
                          Reload with :func:`load_params`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os

from dilute_gas import lee_yang_energy_per_particle, to_paper_energy
from flax import serialization

RUNS_SUBDIR = "runs"


def run_id(potential, x: float, N: int, seed: int, hp) -> str:
    """Filesystem-safe, human-readable label, unique per ``(potential, x, N, seed, hp)``."""
    hp8 = hashlib.sha1(json.dumps(hp.to_dict(), sort_keys=True).encode()).hexdigest()[:8]
    return f"{potential.label}_x{x:.3e}_N{N}_s{seed}_{hp8}"


def run_dir(out_root: str, result) -> str:
    """Relative path (under ``out_root``) of the run dir for ``result``."""
    rid = run_id(result.potential, result.x, result.N, result.seed, result.hp)
    return os.path.join(RUNS_SUBDIR, rid)


def write_run_artifacts(out_root: str, result) -> str:
    """Write meta/history/verdict/params for one run. Returns the run dir *relative* to out_root."""
    rel = run_dir(out_root, result)
    full = os.path.join(out_root, rel)
    os.makedirs(full, exist_ok=True)

    _write_meta(os.path.join(full, "meta.json"), result)
    _write_history(os.path.join(full, "history.csv"), result)
    with open(os.path.join(full, "verdict.json"), "w") as fh:
        json.dump(_json_safe(result.verdict), fh, indent=2)
    _write_params(os.path.join(full, "best_params.msgpack"), result)
    return rel


def _write_meta(path: str, result) -> None:
    x, N = result.x, result.N
    meta = {
        "run_id": os.path.basename(os.path.dirname(path)),
        "units": "paper (hbar^2/2m=1, a=1, energies in hbar^2/2m a^2); see CONVENTIONS.md",
        "potential": {"label": result.potential.label, "R": result.potential.R,
                      "V0_paper": result.potential.V0_paper},
        "x": x, "N": N, "seed": result.seed, "L": result.L,
        "hp": result.hp.to_dict(),
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
              "cm_mean", "cm_std", "wall_time"]
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
