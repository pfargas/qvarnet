"""Per-run output artifacts, written into the run directory that runq hands the target.

runq owns the run-dir naming/layout (``outputs/runs/<label>/`` next to the DB); this module
only knows *what* a CS run leaves behind:

* ``meta.json``           — identity + full physics/hyper-params + exact energy + result;
* ``history.csv``         — per-epoch metrics;
* ``verdict.json``        — the full three-referee ``diagnose()`` dict;
* ``best_params.msgpack`` — the best retained param snapshots by ``select`` (best first).

Nothing here references an absolute path or a live JAX device, so the whole ``outputs/``
tree stays copyable between machines.
"""

from __future__ import annotations

import csv
import json
import os

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
    meta = {
        "run_id": os.path.basename(os.path.dirname(path)),
        "units": "code convention (hbar^2/m=1, omega=1); H = -sum d^2/dx^2 + sum x^2 + 2L(L-1) sum 1/(xi-xj)^2",
        "physics": result.physics.to_dict(),
        "seed": result.seed,
        "hyper_params": result.hp.to_dict(),
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
              "step_size", "grad_norm", "theta_ratio", "cm_mean", "cm_std", "wall_time"]
    with open(path, "w", newline="") as fh:
        # '#'-prefixed header: gnuplot treats it as a comment (plot "..." u 1:2 just
        # works with `set datafile separator ','`); pandas readers strip it back —
        # see RUNNING.md §3.
        fh.write("# " + ",".join(fields) + "\n")
        w = csv.DictWriter(fh, fieldnames=fields)
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
