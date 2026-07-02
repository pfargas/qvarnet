"""Contract tests for the runq target: the flat run_point signature can never drift
from the HyperParams dataclass it constructs. No training happens here.

    uv run pytest soft_sphere_gas/test_point_contract.py
"""

import inspect
from dataclasses import fields

import pytest
from point import HyperParams, Potential, _parse_hidden, feasible_x_grid, run_point
from runq import ParamSpace, Skip

PHYSICS_NAMES = {"R", "x", "N"}

# flat-interface representations that differ from the dataclass (documented in point.py)
TRANSFORMED = {
    "phi_hidden": ("64", (64,)),     # dash-separated string -> tuple
    "F_hidden": ("64", (64,)),
    "jastrow_R": (0.0, None),        # 0.0 sentinel -> None (matched Jastrow)
}


def flat_defaults() -> dict:
    return {
        name: p.default
        for name, p in inspect.signature(run_point).parameters.items()
        if name not in ("seed", "run_dir")
    }


def test_flat_signature_covers_hyper_params_exactly():
    flat = flat_defaults()
    assert set(flat) == PHYSICS_NAMES | {f.name for f in fields(HyperParams)}


def test_flat_defaults_match_dataclass_defaults():
    flat = flat_defaults()
    for f in fields(HyperParams):
        if f.name in TRANSFORMED:
            flat_default, dataclass_value = TRANSFORMED[f.name]
            assert flat[f.name] == flat_default, f"HyperParams.{f.name} (flat side)"
            assert f.default == dataclass_value, f"HyperParams.{f.name} (dataclass side)"
        else:
            assert flat[f.name] == f.default, f"HyperParams.{f.name}"


def test_parse_hidden():
    assert _parse_hidden("64") == (64,)
    assert _parse_hidden("128-128") == (128, 128)
    with pytest.raises(ValueError):
        _parse_hidden("banana")


def test_runq_accepts_the_target():
    space = ParamSpace.from_function(run_point)
    assert space.accepts_run_dir
    assert space.coerce("x", "1e-4") == 1e-4
    assert space.coerce("N", "128") == 128
    assert space.coerce("phi_hidden", "128-128") == "128-128"  # stays a string
    assert space.coerce("use_jastrow", "true") is True
    assert space.coerce("jastrow_R", "5") == 5.0  # float default keeps the float type


def test_infeasible_box_raises_skip():
    # R=10 needs x < N/(8R^3) = 64/8000 = 8e-3; x=0.5 is far beyond -> Skip, no training
    with pytest.raises(Skip, match="box too small"):
        run_point(R=10.0, x=0.5, N=64)


def test_feasible_x_grid_respects_box_ceiling():
    potential = Potential.from_R(10.0)
    xs = feasible_x_grid(potential, N=32, x_lo=1e-5, x_hi=1e-2, n=7)
    assert len(xs) == 7
    assert xs.max() < 32 / (8 * 10.0**3)  # every x feasible at the smallest N
