"""YAML-based experiment runner.

Usage:
    uv run python -m qvarnet.runner experiments/harmonic_oscillator.yaml
    uv run python -m qvarnet.runner experiments/harmonic_oscillator.yaml \\
        --set training.n_epochs=2000 optimizer.learning_rate=5e-4
    qvarnet experiments/harmonic_oscillator.yaml        # after install
"""

import ast
import csv
import sys
from pathlib import Path

import optax
import yaml

from .config.coord_mode import JacobiCoords, LabCoords
from .config.training_setup import CuspConfig, TrainingConfig
from .hamiltonian import get_hamiltonian
from .models import MODEL_REGISTRY
from .train import train


# ---------------------------------------------------------------------------
# Builders: config section → qvarnet object
# ---------------------------------------------------------------------------

def _build_shape(cfg):
    """(n_chains, dof)  — dof can be given directly or as n_particles * n_dim."""
    n_chains = cfg["n_chains"]
    dof = cfg.get("dof") or cfg.get("n_particles", 1) * cfg.get("n_dim", 1)
    return (int(n_chains), int(dof))


def _build_model(cfg):
    cfg = dict(cfg)
    mtype = cfg["type"]
    if mtype not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY)
        raise ValueError(f"Unknown model {mtype!r}. Available: {available}")
    return MODEL_REGISTRY[mtype].from_config(cfg)


def _build_hamiltonian(cfg):
    cfg = dict(cfg)
    htype = cfg.pop("type")
    return get_hamiltonian(htype, **cfg)


def _build_optimizer(cfg):
    cfg = dict(cfg)
    otype = cfg.pop("type")
    lr = float(cfg.pop("learning_rate", 1e-3))
    if otype == "adam":
        return optax.adam(lr, **cfg)
    if otype == "adamw":
        return optax.adamw(lr, **cfg)
    if otype == "sgd":
        return optax.sgd(lr, **cfg)
    raise ValueError(f"Unknown optimizer {otype!r}. Supported: adam, adamw, sgd")


def _build_training_config(cfg, output_dir):
    cfg = dict(cfg)
    cusp_raw = cfg.pop("cusp", None)
    cusp = CuspConfig(**cusp_raw) if cusp_raw else None
    return TrainingConfig(
        checkpoint_path=str(output_dir),
        cusp=cusp,
        **cfg,
    )


def _build_coord_mode(cfg):
    if cfg is None:
        return LabCoords()
    ctype = cfg.get("type", "lab")
    if ctype == "lab":
        return LabCoords()
    if ctype == "jacobi":
        return JacobiCoords(
            n_particles_physical=int(cfg["n_particles_physical"]),
            n_dim=int(cfg["n_dim"]),
        )
    raise ValueError(f"Unknown coord_mode {ctype!r}. Supported: lab, jacobi")


# ---------------------------------------------------------------------------
# Config override: --set training.n_epochs=5000
# ---------------------------------------------------------------------------

def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply dot-path key=value overrides to a nested dict (in-place)."""
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"Override must be key=value, got {kv!r}")
        key, _, raw = kv.partition("=")
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            val = raw  # keep as string
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_energy_history(output_dir: Path, result) -> None:
    import numpy as np
    path = output_dir / "energy_history.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "energy", "std", "acceptance_rate", "step_size"])
        for i, s in enumerate(result.history):
            acc = float(np.mean(s.acceptance_rate)) if s.acceptance_rate is not None else ""
            writer.writerow([
                i,
                float(s.energy),
                float(s.std),
                acc,
                float(s.step_size) if s.step_size is not None else "",
            ])
    print(f"Saved  energy_history.csv → {path}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run(config_path: str, overrides: list[str] | None = None):
    """Run a VMC experiment from a YAML config file.

    Args:
        config_path: path to a YAML config file.
        overrides:   list of "section.key=value" strings (from --set).

    Returns:
        TrainResult
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        _apply_overrides(cfg, overrides)

    name = cfg.get("name", Path(config_path).stem)
    output_dir = Path(cfg.get("output_dir", f"./outputs/{name}/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the resolved config for reproducibility
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    shape = _build_shape(cfg["shape"])
    model = _build_model(cfg["model"])
    hamiltonian = _build_hamiltonian(cfg["hamiltonian"])
    optimizer = _build_optimizer(cfg["optimizer"])
    training_config = _build_training_config(cfg["training"], output_dir)
    sampler_params = dict(cfg["sampler"])
    coord_mode = _build_coord_mode(cfg.get("coord_mode"))

    print("=" * 55)
    print(f"  Experiment : {name}")
    print(f"  Output     : {output_dir}")
    print(f"  Shape      : n_chains={shape[0]}, dof={shape[1]}")
    print(f"  Model      : {cfg['model']['type']}")
    print(f"  Hamiltonian: {cfg['hamiltonian']['type']}")
    print(f"  Epochs     : {training_config.n_epochs}")
    print("=" * 55)

    result = train(
        shape=shape,
        model=model,
        model_name=cfg["model"]["type"],
        model_args=dict(cfg["model"]),
        optimizer=optimizer,
        hamiltonian=hamiltonian,
        training_config=training_config,
        sampler_params=sampler_params,
        coord_mode=coord_mode,
    )

    _save_energy_history(output_dir, result)

    best = result.best(n=1, metric="energy")[0]
    print(f"Best energy : {float(best.energy):.6f}  ±  {float(best.std):.6f}")

    return result


def main():
    """CLI entry point: qvarnet <config.yaml> [--set key=value ...]"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="qvarnet",
        description="Run a VMC experiment from a YAML config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  qvarnet experiments/harmonic_oscillator.yaml
  qvarnet experiments/harmonic_oscillator.yaml --set training.n_epochs=2000
  qvarnet experiments/harmonic_oscillator.yaml --set optimizer.learning_rate=5e-4 shape.n_chains=2000
        """,
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        nargs="*",
        default=[],
        help="Override config values, e.g. --set training.n_epochs=2000",
    )
    args = parser.parse_args()
    run(args.config, overrides=args.set or [])


if __name__ == "__main__":
    main()
