"""Per-run output artifacts — the portable, self-contained layout under ``outputs/``.

A sweep produces one **run directory** per ``(physics, seed, hp)`` under ``outputs/runs/<run_id>/``,
alongside the index DB ``outputs/cs.db``. The whole ``outputs/`` tree is self-contained and copyable
between machines: nothing here references an absolute path or a live JAX device.

``run_id``
----------
``<physics-label>_s<seed>_<hp8>`` — e.g. ``L0.8_N5_s0_3f9a1c2b``. The physics axes are
human-readable; ``hp8`` is the first 8 hex of a hash of the *full* hyperparameter dict, so two runs
that differ only in solver settings get distinct dirs. The full HP + physics are in ``meta.json``
and the DB, so the hash is never inverted.

Run-dir contents: ``meta.json`` (identity + full physics/hp + exact energy + result),
``history.csv`` (per-epoch metrics), ``verdict.json`` (three-referee diagnose dict),
``best_params.msgpack`` (best ``snapshot_frac`` of epochs by ``select``, best first).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

from flax import serialization

RUNS_SUBDIR = "runs"


def run_id(physics, seed: int, hp) -> str:
    """Filesystem-safe, human-readable label, unique per ``(physics, seed, hp)``."""
    hp8 = hashlib.sha1(json.dumps(hp.to_dict(), sort_keys=True).encode()).hexdigest()[:8]
    return f"{physics.label}_s{seed}_{hp8}"


def run_dir(result) -> str:
    """Relative path (under out_root) of the run dir for ``result``."""
    return os.path.join(RUNS_SUBDIR, run_id(result.physics, result.seed, result.hp))


def write_run_artifacts(out_root: str, result) -> str:
    """Write meta/history/verdict/params for one run. Returns the run dir *relative* to out_root."""
    rel = run_dir(result)
    full = os.path.join(out_root, rel)
    os.makedirs(full, exist_ok=True)

    _write_meta(os.path.join(full, "meta.json"), result)
    _write_history(os.path.join(full, "history.csv"), result)
    with open(os.path.join(full, "verdict.json"), "w") as fh:
        json.dump(_json_safe(result.verdict), fh, indent=2)
    _write_params(os.path.join(full, "best_params.msgpack"), result)
    return rel


def _write_meta(path: str, result) -> None:
    meta = {
        "run_id": os.path.basename(os.path.dirname(path)),
        "units": "code convention (hbar^2/m=1, omega=1); H = -sum d^2/dx^2 + sum x^2 + 2L(L-1) sum 1/(xi-xj)^2",
        "physics": result.physics.to_dict(),
        "seed": result.seed,
        "hp": result.hp.to_dict(),
        "exact_energy": result.e_exact,
        "result": {
            "e_total": result.e_total, "e_per_n": result.e_per_n,
            "err_total": result.err_total, "err_per_n": result.err_per_n,
            "sigma_e": result.sigma_e, "acceptance": result.acceptance,
            "passed": bool(result.passed), "gap": result.gap,
        },
    }
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)


def _write_history(path: str, result) -> None:
    rows = result.history_rows
    if not rows:
        return
    fields = ["epoch", "energy", "std", "error_of_mean", "acceptance",
              "step_size", "cm_mean", "cm_std", "wall_time"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _write_params(path: str, result) -> None:
    snaps = result.snapshots
    if not snaps:
        return
    payload = {
        "select": result.hp.select,
        "steps": [int(s["step"]) for s in snaps],
        "metrics": [float(s["metric"]) for s in snaps],
        "params": [s["params"] for s in snaps],  # host pytrees, best (lowest metric) first
    }
    with open(path, "wb") as fh:
        fh.write(serialization.msgpack_serialize(payload))


def load_params(path: str) -> dict:
    """Reload a ``best_params.msgpack``: ``{"select","steps","metrics","params":[pytree,...]}``."""
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
