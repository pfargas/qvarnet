import flax
import os


def save_checkpoint(state, path, filename="checkpoint.msgpack"):
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
    checkpoint_dir = os.path.join(path, "checkpoints")
    if os.path.exists(os.path.join(checkpoint_dir, filename)):
        with open(os.path.join(checkpoint_dir, filename), "rb") as f:
            bytes_data = f.read()
        # This updates the 'state' object with the saved values
        return flax.serialization.from_bytes(state, bytes_data)
    return state
