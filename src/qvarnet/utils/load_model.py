import json
import flax.serialization as serialization
from flax import linen as nn
import jax
import jax.numpy as jnp


def load_flax_from_json(path: str, model: nn.Module, input_dim: int):
    """Load Flax parameters from a JSON file.

    Args:
        path: Path to the JSON file.
        model: The Flax model (nn.Module) instance.
        input_dim: The input dimension used to initialize the model.

    Returns:
        The loaded parameters in the structure expected by the model.
    """
    with open(path, "r") as f:
        new_params_dict = json.load(f)

    # Initialize model with dummy data to get the expected parameter structure
    dummy_input = jnp.zeros((1, input_dim))
    variables = model.init(jax.random.PRNGKey(0), dummy_input)

    # Check if we should use 'params' or the whole variables dict
    # Most models in this codebase seem to use 'params' key
    if "params" in variables and "params" not in new_params_dict:
        # If the model has a 'params' level but the JSON doesn't, wrap the JSON
        new_params_dict = {"params": new_params_dict}

    final_new_params_dict = jax.tree_util.tree_map(jnp.array, new_params_dict)

    params = serialization.from_state_dict(variables, final_new_params_dict)
    return params
