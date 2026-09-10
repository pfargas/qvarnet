"""Fail if any qvarnet module imports from its own layer or above.

The package is a stack. A module may import from layers strictly below it and
nowhere else, which is what keeps the dependency graph acyclic -- the alternative
is what this codebase had before: callbacks <-> vmc, callbacks <-> diagnostics,
and a config package importing the samplers it configured.

    uv run python scripts/check_layers.py

Reads imports statically, so it needs neither jax nor an installed package.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "qvarnet"

# Low to high. A module may import only from layers with a strictly lower index.
LAYERS = [
    "core",
    "physics",
    "ansatz",
    "sampling",
    "optim",
    "analysis",
    "callbacks",
    "vmc",
]
# Top-level modules that sit above everything (the public surface).
TOP = {"__init__", "recipes"}


def layer_of(module: str) -> int | None:
    """Index of the layer a dotted qvarnet module belongs to, or None if top-level."""
    parts = module.split(".")
    if len(parts) < 2:
        return None
    head = parts[1]
    if head in TOP:
        return None
    return LAYERS.index(head) if head in LAYERS else None


def module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_modules(path: Path) -> list[tuple[str, int]]:
    """(module, lineno) for every qvarnet import in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.startswith("qvarnet"):
                out.append((node.module, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("qvarnet"):
                    out.append((alias.name, node.lineno))
    return out


def main() -> int:
    violations = []
    relative = []

    for path in sorted(SRC.rglob("*.py")):
        name = module_name(path)
        own = layer_of(name)

        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                relative.append((name, node.lineno))

        if own is None:  # __init__ / recipes may import anything
            continue

        for target, lineno in imported_modules(path):
            other = layer_of(target)
            if other is None or other < own:
                continue
            if other == own:
                # Same layer is fine only within the same top-level package,
                # which is how a package's own submodules talk to each other.
                continue
            violations.append(
                f"  {name}:{lineno}  imports {target}\n"
                f"      {LAYERS[own]} (layer {own}) may not import {LAYERS[other]} (layer {other})"
            )

    if relative:
        print(f"{len(relative)} relative import(s) found; qvarnet uses absolute imports:")
        for name, lineno in relative[:10]:
            print(f"  {name}:{lineno}")

    if violations:
        print(f"\n{len(violations)} layering violation(s):\n")
        print("\n".join(violations))
        print("\nLayer order (low to high): " + " < ".join(LAYERS))
        return 1

    if relative:
        return 1

    n = sum(1 for _ in SRC.rglob("*.py"))
    print(f"OK: {n} modules, no upward or sideways imports across layers.")
    print("Layer order (low to high): " + " < ".join(LAYERS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
