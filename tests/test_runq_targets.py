"""Frozen-signature guard for the runq sweep targets.

runq keys every completed run by ``key_json(params)`` -- the canonical JSON of the
fully resolved parameter dict -- and names its artifact directory by that dict's
hash. So adding, removing, renaming or re-defaulting any ``run_point`` parameter
re-keys every point in that sweep: finished runs stop matching and re-run.

At the time of writing that is 305 completed runs across cs-new, soft-bosons and
soft-hard-bosons. This test pins the signatures against a committed snapshot so a
library refactor cannot silently invalidate them.

The signature is read **statically, with ast** -- importing the targets would need
runq, jax and their sibling modules on the path, and would break for targets living
outside this repo. Nothing here imports qvarnet.

To re-snapshot after a *deliberate* signature change (which re-keys the sweep --
see docs/adr/), run:

    uv run python tests/test_runq_targets.py --update
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).parent / "data" / "runq_signatures.json"

# Sweep targets, relative to the repo root. Targets outside the repo are reached by
# ``..`` and skipped when absent (a fresh clone will not have the sibling projects).
TARGETS = [
    "soft_sphere_gas/point.py",
    "calogero-sutherland/cs_sweep/point.py",
    "../cs-new/cs_sweep/point.py",
]

# runq injects this one; it is never part of the parameter space or the key.
RESERVED = ("run_dir",)


def _sentinel(node: ast.AST) -> str:
    """Readable stand-in for a default that is not a plain literal."""
    return f"<expr:{ast.dump(node)}>"


def run_point_defaults(path: Path) -> dict:
    """The ``run_point`` parameter space of ``path``, read without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        (
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "run_point"
        ),
        None,
    )
    if fn is None:
        raise AssertionError(f"{path} has no top-level run_point()")

    args = fn.args
    positional = args.posonlyargs + args.args
    # Defaults right-align onto the positional parameters.
    paired = list(zip(positional[len(positional) - len(args.defaults) :], args.defaults))
    paired += list(zip(args.kwonlyargs, args.kw_defaults))

    out = {}
    for arg, default in paired:
        if arg.arg in RESERVED or default is None:
            continue
        try:
            out[arg.arg] = ast.literal_eval(default)
        except ValueError:
            out[arg.arg] = _sentinel(default)
    return out


def collect() -> dict:
    """Signatures of every target that exists, keyed by the path spelled in TARGETS."""
    found = {}
    for rel in TARGETS:
        path = (REPO / rel).resolve()
        if path.is_file():
            found[rel] = run_point_defaults(path)
    return found


@pytest.mark.parametrize("rel", TARGETS)
def test_run_point_signature_is_frozen(rel):
    path = (REPO / rel).resolve()
    if not path.is_file():
        pytest.skip(f"sweep target not present: {rel}")

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if rel not in expected:
        pytest.skip(f"{rel} is not in the snapshot; re-run with --update to add it")

    actual = run_point_defaults(path)
    want = expected[rel]

    added = sorted(set(actual) - set(want))
    removed = sorted(set(want) - set(actual))
    changed = {
        k: (want[k], actual[k]) for k in sorted(set(want) & set(actual)) if want[k] != actual[k]
    }

    assert not (added or removed or changed), (
        f"{rel}: run_point signature changed -- this re-keys every completed run "
        f"in that sweep.\n"
        f"  added:   {added}\n"
        f"  removed: {removed}\n"
        f"  changed: {changed}\n"
        f"If the change is deliberate, re-snapshot with "
        f"`uv run python tests/test_runq_targets.py --update` and record why."
    )


def test_snapshot_covers_every_present_target():
    """A target that exists but is missing from the snapshot is unprotected."""
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    missing = sorted(set(collect()) - set(expected))
    assert not missing, f"sweep targets present but unprotected: {missing}"


if __name__ == "__main__":
    import sys

    if "--update" in sys.argv:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        data = collect()
        SNAPSHOT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for name, params in sorted(data.items()):
            print(f"{name}: {len(params)} parameters")
        print(f"\nwrote {SNAPSHOT}")
    else:
        print(json.dumps(collect(), indent=2, sort_keys=True))
