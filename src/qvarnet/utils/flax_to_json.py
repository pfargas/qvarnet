from flax import serialization
import jax
import json


def save_flax_to_json(params, filename):
    # 1. Convert Flax structure (or TrainState) to a plain nested dict
    state_dict = serialization.to_state_dict(params)

    # 2. Convert JAX/NumPy arrays to Python lists for JSON
    serializable_dict = jax.tree.map(lambda x: x.tolist(), state_dict)

    with open(filename, "w") as f:
        json.dump(serializable_dict, f, indent=4)
