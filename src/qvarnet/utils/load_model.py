import json
import flax.serialization as serialization
from flax import linen as nn
import jax
import jax.numpy as jnp
from ..models import get_model
import os


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


def load_model_from_results(results_path: str):
    """Load a model and its parameters from a results directory.

    Returns:
        (model, params, input_dim)
    """
    config_path = os.path.join(results_path, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found in {results_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    model_args = config["model"]
    model_name = model_args.get("type")

    # Replicate model instantiation logic from run_experiment
    if model_name == "fermionic-mlp":
        model = get_model(
            model_name,
            architecture=model_args["architecture"],
            n_fermions=model_args["n_fermions"],
            n_dim=model_args["n_dim"],
        )
        input_dim = model_args["n_fermions"] * model_args["n_dim"]
    else:
        model = get_model(model_name, architecture=model_args["architecture"])
        input_dim = model_args["architecture"][0]

    params_path = os.path.join(results_path, "parameters.json")
    if not os.path.exists(params_path):
        raise ValueError(f"Parameters file not found in {results_path}")

    raw_params = load_flax_from_json(params_path, model, input_dim)
    jax_params = jax.tree_util.tree_map(
        jnp.array,
        raw_params,
        is_leaf=lambda x: isinstance(
            x, list
        ),  # This converts lists to JAX arrays. i don't get the is_leaf thing.
    )

    # Ensure params is a dictionary with a 'params' key if the model expects it,
    # and return input_dim as well since it's useful for calling the model.
    return model, jax_params, input_dim
