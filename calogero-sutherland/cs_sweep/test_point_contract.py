"""Contract tests for the runq target: the flat run_point signature can never drift
from the Physics / HyperParams dataclasses it constructs. No training happens here.

    uv run pytest calogero-sutherland/cs_sweep/test_point_contract.py
"""

import inspect
from dataclasses import fields

from point import HyperParams, Physics, run_point
from runq import ParamSpace

# HyperParams fields deliberately not exposed on the flat interface
NOT_EXPOSED = {"mlp_hidden"}  # explicit per-layer widths; use mlp_width × mlp_layers


def flat_defaults() -> dict:
    return {
        name: p.default
        for name, p in inspect.signature(run_point).parameters.items()
        if name not in ("seed", "run_dir")
    }


def test_flat_signature_covers_both_dataclasses_exactly():
    flat = flat_defaults()
    physics_names = {f.name for f in fields(Physics)}
    hyper_names = {f.name for f in fields(HyperParams)} - NOT_EXPOSED
    assert set(flat) == physics_names | hyper_names


def test_flat_defaults_match_dataclass_defaults():
    flat = flat_defaults()
    for f in fields(Physics):
        assert flat[f.name] == f.default, f"Physics.{f.name}"
    for f in fields(HyperParams):
        if f.name in NOT_EXPOSED:
            continue
        assert flat[f.name] == f.default, f"HyperParams.{f.name}"


def test_runq_accepts_the_target():
    space = ParamSpace.from_function(run_point)
    assert space.accepts_run_dir
    assert "seed" in space.defaults
    # CLI coercion works for a representative axis of each type
    assert space.coerce("L", "0.5") == 0.5
    assert space.coerce("N", "10") == 10
    assert space.coerce("kind", "jastrow") == "jastrow"
    assert space.coerce("early_stop", "true") is True


def test_dataclasses_build_from_flat_defaults():
    flat = flat_defaults()
    physics = Physics(**{f.name: flat[f.name] for f in fields(Physics)})
    hyper_params = HyperParams(
        **{f.name: flat[f.name] for f in fields(HyperParams) if f.name not in NOT_EXPOSED}
    )
    assert physics == Physics()
    assert hyper_params == HyperParams()
    assert physics.exact_energy() == 5 * (1.0 + 0.8 * 4)  # E0 = N(1 + L(N-1))
