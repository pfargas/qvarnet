import flax
import os


def save_checkpoint(state, path, filename="checkpoint.msgpack"):
    """Serialise a Flax TrainState and write it to disk atomically.

    Writes to a ``.tmp`` file first, then renames it to ``filename`` to avoid
    partial writes on interrupt.  Checkpoints are stored under
    ``<path>/checkpoints/<filename>``.

    Args:
        state: Flax ``TrainState`` (or ``VMCState``) to save.
        path: Base directory for the experiment output.
        filename: Target filename inside the ``checkpoints/`` subdirectory.
    """
    # 1. Convert the TrainState PyTree into bytes
    bytes_output = flax.serialization.to_bytes(state)

    # 2. Write to a temporary file first (Safety first!)
    tmp_file = filename + ".tmp"
    checkpoint_dir = os.path.join(path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(tmp_file, "wb") as f:
        f.write(bytes_output)

    # 3. Rename tmp to final (Atomic swap)
    os.replace(tmp_file, os.path.join(checkpoint_dir, filename))


def load_checkpoint(state, path, filename="vmc_last_state.msgpack"):
    """Load a previously saved checkpoint, returning the original state if none exists.

    Args:
        state: Template ``VMCState`` used to reconstruct the PyTree structure from bytes.
        path: Base directory where checkpoints are stored.
        filename: Filename inside the ``checkpoints/`` subdirectory.

    Returns:
        Restored ``VMCState`` if the checkpoint file exists, otherwise the original ``state``.
    """
    checkpoint_dir = os.path.join(path, "checkpoints")
    if os.path.exists(os.path.join(checkpoint_dir, filename)):
        with open(os.path.join(checkpoint_dir, filename), "rb") as f:
            bytes_data = f.read()
        # This updates the 'state' object with the saved values
        return flax.serialization.from_bytes(state, bytes_data)
    return state
