"""Msgpack save/load of a Flax TrainState.

Deliberately dependency-free: ``callbacks`` needs checkpoint writing, and
reaching up into ``vmc`` for it is what made callbacks and vmc mutually
dependent. Run-config I/O, which does need the configs, lives in vmc/run_io.py.
"""

import os

import flax


def save_checkpoint(state, path, filename="checkpoint.msgpack"):
    """Serialise a Flax TrainState and write it to disk atomically."""
    bytes_output = flax.serialization.to_bytes(state)
    checkpoint_dir = os.path.join(path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    tmp_file = os.path.join(checkpoint_dir, filename + ".tmp")
    with open(tmp_file, "wb") as f:
        f.write(bytes_output)
    os.replace(tmp_file, os.path.join(checkpoint_dir, filename))


def load_checkpoint(state, path, filename="vmc_last_state.msgpack"):
    """Load a previously saved checkpoint, returning the original state if none exists."""
    checkpoint_dir = os.path.join(path, "checkpoints")
    fpath = os.path.join(checkpoint_dir, filename)
    if os.path.exists(fpath):
        with open(fpath, "rb") as f:
            bytes_data = f.read()
        return flax.serialization.from_bytes(state, bytes_data)
    return state
