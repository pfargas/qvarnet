"""Recording and reloading a run: run_config.json plus the saved parameters."""

import collections
import dataclasses
import json
import os

import flax
import jax
import jax.numpy as jnp

from qvarnet.core.coords import CoordMode, JacobiCoords, LabCoords
from qvarnet.core.serialization import load_checkpoint, save_checkpoint  # noqa: F401
from qvarnet.vmc.config import CuspConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Run config serialisation helpers
# ---------------------------------------------------------------------------


def _coord_mode_to_dict(coord_mode: CoordMode) -> dict:
    if isinstance(coord_mode, LabCoords):
        return {"type": "LabCoords"}
    if isinstance(coord_mode, JacobiCoords):
        return {
            "type": "JacobiCoords",
            "n_particles_physical": coord_mode.n_particles_physical,
            "n_dim": coord_mode.n_dim,
        }
    raise ValueError(f"Cannot serialise CoordMode of type {type(coord_mode)}")


def _coord_mode_from_dict(d: dict) -> CoordMode:
    t = d["type"]
    if t == "LabCoords":
        return LabCoords()
    if t == "JacobiCoords":
        return JacobiCoords(
            n_particles_physical=d["n_particles_physical"],
            n_dim=d["n_dim"],
        )
    raise ValueError(f"Unknown CoordMode type: {t!r}")


def _training_config_to_dict(cfg: TrainingConfig) -> dict:
    d = dataclasses.asdict(cfg)  # handles nested CuspConfig automatically
    return d


def _training_config_from_dict(d: dict) -> TrainingConfig:
    d = dict(d)
    cusp_dict = d.pop("cusp", None)
    cusp = CuspConfig(**cusp_dict) if cusp_dict is not None else None
    return TrainingConfig(**d, cusp=cusp)


# ---------------------------------------------------------------------------
# High-level run config save / load
# ---------------------------------------------------------------------------


def save_run_config(path, model_name, model_args, sample_shape, coord_mode, training_config):
    """Write a run_config.json that captures everything needed to reconstruct the run.

    Call once at the start of training. Enables load_run() later.

    Args:
        path: checkpoint_path from TrainingConfig (base directory for the run).
        model_name: provenance label for the ansatz (e.g. "mlp", "deep-set").
            Recorded for the human reading the run later; nothing reconstructs
            a model from it.
        model_args: dict of model constructor kwargs — must be JSON-serialisable.
        sample_shape: tuple (n_chains, dof) passed to train().
        coord_mode: CoordMode instance used for this run.
        training_config: TrainingConfig instance.
    """
    config = {
        "model_name": model_name,
        "model_args": model_args,
        "sample_shape": list(sample_shape),
        "coord_mode": _coord_mode_to_dict(coord_mode),
        "training_config": _training_config_to_dict(training_config),
    }
    checkpoint_dir = os.path.join(path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_path = os.path.join(checkpoint_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


LoadedRun = collections.namedtuple(
    "LoadedRun", ["model", "params", "training_config", "coord_mode"]
)


def load_run(path, model, checkpoint_filename="checkpoint.msgpack"):
    """Load the parameters and config of a finished run into ``model``.

    Args:
        path: the checkpoint_path used during training.
        model: the ansatz to load into — construct the same one you trained with.
        checkpoint_filename: which checkpoint to load (default "checkpoint.msgpack").

    Returns:
        LoadedRun(model, params, training_config, coord_mode)

    Example:
        model = LogWavefunction(network=MLP(hidden=[64, 64]), ...)
        run = load_run("./outputs/my_run/", model)
    """
    checkpoint_dir = os.path.join(path, "checkpoints")
    config_path = os.path.join(checkpoint_dir, "run_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No run_config.json found at {config_path}. "
            "Was save_run_config() called during training? "
            "train() writes it only when model_name and model_args are passed."
        )

    with open(config_path) as f:
        config = json.load(f)

    # Reconstruct components
    coord_mode = _coord_mode_from_dict(config["coord_mode"])
    training_config = _training_config_from_dict(config["training_config"])
    sample_shape = tuple(config["sample_shape"])

    # Init model to get parameter tree shape, then restore only the params
    # from the checkpoint (ignore optimizer state — load_run is for inference).
    init_shape = coord_mode.model_input_shape(sample_shape)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones(init_shape))

    checkpoint_dir = os.path.join(path, "checkpoints")
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_filename)
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            bytes_data = f.read()
        raw = flax.serialization.msgpack_restore(bytes_data)
        params = flax.serialization.from_state_dict(params, raw["params"])

    return LoadedRun(
        model=model,
        params=params,
        training_config=training_config,
        coord_mode=coord_mode,
    )
