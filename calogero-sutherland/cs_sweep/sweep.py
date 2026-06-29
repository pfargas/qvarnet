"""Grid-agnostic sweep orchestration over ``point.run_point``, resumably.

The harness does **not** hardcode which axes vary. ``build_grid`` takes a dict of physics axes and a
dict of hp axes and returns the cartesian product of ``(Physics, HP)`` points; ``sweep_grid`` /
``enqueue_grid`` then run (or queue) every ``(point, seed)``. Grid over L and N, or over lr and
model ``kind``, or any mix — the DB key ``(physics, seed, hp)`` keeps them all distinct.

    grid = build_grid(physics_axes={"L": [0.5, 0.8, 1.0], "N": [2, 5, 10]},
                      hp_axes={"kind": ["jastrow", "mlp_jastrow"]})
    sweep_grid(conn, grid, seeds=[0, 1, 2], out_root="outputs")
"""

from __future__ import annotations

import itertools
import json
import os
import traceback
from dataclasses import replace

import artifacts
import db
from point import HP, Physics, run_point


# ── grid construction ────────────────────────────────────────────────────────────────


def build_grid(
    physics_axes: dict | None = None,
    hp_axes: dict | None = None,
    *,
    base_physics: Physics | None = None,
    base_hp: HP | None = None,
) -> list[tuple[Physics, HP]]:
    """Cartesian product of physics axes × hp axes, off a base point.

    ``physics_axes`` / ``hp_axes`` map a field name to the list of values it takes; any field not
    listed keeps its base value. Returns ``[(Physics, HP), ...]``. Agnostic to *which* fields vary.
    """
    base_physics = base_physics or Physics()
    base_hp = base_hp or HP()
    physics_axes = physics_axes or {}
    hp_axes = hp_axes or {}

    physics_pts = _expand(base_physics, physics_axes)
    hp_pts = _expand(base_hp, hp_axes)
    return [(p, h) for p in physics_pts for h in hp_pts]


def _expand(base, axes: dict) -> list:
    if not axes:
        return [base]
    names = list(axes)
    out = []
    for combo in itertools.product(*(axes[n] for n in names)):
        out.append(replace(base, **dict(zip(names, combo))))
    return out


# ── the runs ──────────────────────────────────────────────────────────────────────


def execute_claimed(conn, physics, seed, hp, *, out_root: str) -> None:
    """Train one already-claimed (``running``) point, write artifacts, save the result.

    Shared by the serial driver (``run_one``) and the parallel ``worker.py``. Assumes the row is
    already ``running``. On failure the row is marked ``failed`` (with traceback) and re-raised.
    """
    rel = os.path.join(artifacts.RUNS_SUBDIR, artifacts.run_id(physics, seed, hp))
    run_path = os.path.join(out_root, rel)
    os.makedirs(run_path, exist_ok=True)
    try:
        result = run_point(physics, seed, hp, checkpoint_dir=run_path)
        artifacts.write_run_artifacts(out_root, result)
        db.save_result(conn, result, run_dir=rel)
    except Exception:
        db.mark_failed(conn, physics, seed, hp, traceback.format_exc())
        raise


def run_one(conn, physics, seed, hp, *, out_root: str, force: bool = False) -> None:
    """Run a single point unless already done (resumable, serial)."""
    if not force and db.status_of(conn, physics, seed, hp) == "done":
        return
    db.enqueue(conn, physics, seed, hp)
    db.mark_running(conn, physics, seed, hp)
    execute_claimed(conn, physics, seed, hp, out_root=out_root)


def sweep_grid(conn, grid, seeds, *, out_root: str = "outputs", force: bool = False) -> None:
    """Serially run every ``(point, seed)`` in ``grid``. Resumable (skips done)."""
    os.makedirs(out_root, exist_ok=True)
    for physics, hp in grid:
        for seed in seeds:
            run_one(conn, physics, seed, hp, out_root=out_root, force=force)
            print(f"[{db.status_of(conn, physics, seed, hp)}] {physics.label} "
                  f"kind={hp.kind} seed={seed}")


def enqueue_grid(conn, grid, seeds) -> int:
    """Insert all ``todo`` rows for ``grid`` × ``seeds`` (for the worker-per-GPU runner)."""
    n = 0
    for physics, hp in grid:
        for seed in seeds:
            db.enqueue(conn, physics, seed, hp)
            n += 1
    return n


# ── aggregation / retrieval ──────────────────────────────────────────────────────────


def load_table(conn):
    """All done runs as a pandas DataFrame, with ``physics_json``/``hp_json`` expanded to columns.

    Physics fields are returned as-is (``L``, ``N``, …); hp fields are prefixed ``hp_`` to avoid
    collisions. Fully grid-agnostic — whatever axes you swept become columns you can group by.
    """
    import pandas as pd

    df = pd.read_sql("SELECT * FROM runs WHERE status='done'", conn)
    if df.empty:
        return df
    phys = pd.json_normalize(df["physics_json"].map(json.loads))
    hp = pd.json_normalize(df["hp_json"].map(json.loads)).add_prefix("hp_")
    out = pd.concat([df.drop(columns=["physics_json", "hp_json"]), phys, hp], axis=1)
    return out


def load_curve(conn, x_axis: str = "L", *, fixed: dict | None = None,
               require_passed: bool = True):
    """Seed-averaged ``E/N`` vs one physics axis (``x_axis``), holding other axes ``fixed``.

    ``fixed`` selects a slice (e.g. ``{"N": 5}``). Combines seeds per x-value: the error is the
    larger of the across-seed SEM and the mean within-seed error (never understated). Returns a
    dict of sorted numpy arrays: ``x, e_per_n, err, e_exact, n_per_x``.
    """
    import numpy as np

    df = load_table(conn)
    if df.empty:
        return {"x": np.array([]), "e_per_n": np.array([]), "err": np.array([]),
                "e_exact": np.array([]), "n_per_x": []}
    if require_passed:
        df = df[df["passed"] == 1]
    for k, v in (fixed or {}).items():
        df = df[df[k] == v]

    xs, e, err, ex, npx = [], [], [], [], []
    for xval, grp in df.groupby(x_axis):
        evals = grp["e_per_n"].to_numpy()
        within = grp["err_per_n"].to_numpy()
        mean = float(evals.mean())
        across = float(evals.std(ddof=1) / np.sqrt(len(evals))) if len(evals) > 1 else 0.0
        xs.append(xval)
        e.append(mean)
        err.append(max(across, float(within.mean())))
        ex.append(float(grp["e_exact"].iloc[0]) / float(grp["N"].iloc[0]))
        npx.append(len(grp))
    order = np.argsort(xs)
    return {"x": np.array(xs)[order], "e_per_n": np.array(e)[order],
            "err": np.array(err)[order], "e_exact": np.array(ex)[order],
            "n_per_x": [npx[i] for i in order]}
