import os

from .models import get_model
from .train import train
import jax
import jax.numpy as jnp
import optax
import time
from .utils import (
    save_flax_to_json,
    save_energy_history,
    save_metrics,
    save_config,
    create_output_directory,
    load_custom_module,
)
from .hamiltonian import get_hamiltonian


def run_experiment(args=None, profile=False):
    """Run a quantum variational Monte Carlo experiment.
    Args:

        args: An object containing model, training, sampler, and optimizer arguments.
        profile: Boolean flag to enable JAX profiling.
    Returns:
        None"""
    if args is None:
        raise ValueError("Arguments must be provided to run_experiment")

    model_args = args.get_model_args()
    train_args = args.get_training_args()
    sampler_args = args.get_sampler_args()
    optimizer_args = args.get_optimizer_args()
    output_args = args.get_output_args()
    master_seed = args.get_seed()
    exp_info = args.get_info_experiment()
    hami_args = args.get_hamiltonian_args()

    output_base_path = output_args.get("save_dir", "tmp/qvarnet/results")
    experiment_name = exp_info.get("name", None)
    experiment_description = (
        f"#{exp_info.get('description', 'No description provided.')}"
    )

    experiment_description += f"""

Experiment info:
- Model: {model_args.get('type', 'N/A')}
- Hamiltonian: {hami_args.get('name', 'N/A')}
- Sampler info: {sampler_args}
- Optimizer info: {optimizer_args}
- Training info: {train_args}
"""

    base_path = create_output_directory(
        output_base_path, experiment_name, load_config=False
    )
    with open(os.path.join(base_path, "description.txt"), "w") as f:
        f.write(experiment_description)

    print("=" * 40)
    print(f"Results will be saved to: {base_path}")
    print("=" * 40)

    # **************************************************
    # ****                Choose model              ****
    # **************************************************

    # if model name is custom, load from custom path
    if args.args.custom_model:
        load_custom_module(args.args.custom_model)
        print("Custom model loaded.")
        from qvarnet.models.registry import MODEL_REGISTRY

        print("Available models:", list(MODEL_REGISTRY.keys()))
        model_name = "new_model"  # FIXME: custom_model is hardcoded here
    else:
        model_name = model_args.get("type", "exponential-mlp-fourth-decay")
    if model_name == "fermionic-mlp":
        is_fermionic = True
        model = get_model(
            model_name,
            architecture=model_args["architecture"],
            n_fermions=model_args["n_fermions"],
            n_dim=model_args["n_dim"],
        )
        print(
            f"Using FermionicMLP with {model_args['n_fermions']} fermions and {model_args['n_dim']} dimensions."
        )
    else:
        model = get_model(model_name, architecture=model_args["architecture"])
    # **************************************************

    # **************************************************
    # ****             Choose hamiltonian           ****
    # **************************************************
    if args.args.custom_hamiltonian:
        load_custom_module(args.args.custom_hamiltonian)
        print("Custom hamiltonian loaded.")
        hamiltonian_name = "local_potential"  # FIXME: local_potential is hardcoded here
        # I propose that the name should be the name of the file without extension, but this is a quick fix for now
    else:
        hamiltonian_name = hami_args.get("name", "harmonic-oscillator")
    hamiltonian = get_hamiltonian(hamiltonian_name, **hami_args.get("params", {}))
    # **************************************************

    if optimizer_args["type"] == "adam":
        optimizer = optax.adam(learning_rate=optimizer_args["learning_rate"])
    elif optimizer_args["type"] == "sgd":
        optimizer = optax.sgd(learning_rate=optimizer_args["learning_rate"])
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_args['type']}")

    shape = (
        train_args["batch_size"],  # number of parallel chains
        model_args["architecture"][0],  # input dimension (degrees of freedom)
    )
    if is_fermionic:
        shape = (
            train_args["batch_size"],
            model_args["n_fermions"]
            * model_args[
                "n_dim"
            ],  # For the FermionicMLP, we have 4 fermions, so input dimension is 4 * 3 dimensions = 12
        )

    if profile:
        jax.profiler.start_trace("/tmp/profile-data")

    time_start = time.perf_counter()
    energy_hist, best_state = train(
        n_epochs=train_args["num_epochs"],
        shape=shape,
        model=model,
        optimizer=optimizer,
        sampler_params=sampler_args,
        rng_seed=master_seed,
        hamiltonian=hamiltonian,
        checkpoint_path=base_path,
        save_checkpoints=output_args.get("save_checkpoints", False),
    )
    time_end = time.perf_counter()

    if profile:
        jax.profiler.stop_trace()

    best_params = best_state.params
    best_score = best_state.score
    # remove zeroes from energy_hist
    energy_hist = energy_hist[energy_hist != 0.0]

    print(f"Total training time: {time_end - time_start: <.4f} seconds")
    print(f"Best score: {best_score:<.4f}")
    print("=" * 40)
    print("Saving results...")

    # Save model parameters
    try:
        save_flax_to_json(best_params, os.path.join(base_path, "parameters.json"))
        print("Saved parameters to parameters.json")
    except Exception as e:
        print(f"Error saving parameters: {e}")

    # Save energy history
    try:
        save_energy_history(base_path, energy_hist)
        print("Saved energy history to energy_history.csv")
    except Exception as e:
        print(f"Error saving energy history: {e}")

    # Save final metrics
    try:
        save_metrics(
            base_path,
            {
                "total_energy": float(best_state.energy),
                "std": float(best_state.std),
                "training_time_seconds": time_end - time_start,
                "best_score": float(best_score),
            },
        )
        print("Saved metrics to metrics.json")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    # Save experiment configuration
    try:
        save_config(
            base_path,
            {
                "model": model_args,
                "training": train_args,
                "sampler": sampler_args,
                "optimizer": optimizer_args,
                "hamiltonian": hami_args,
                "seed": master_seed,
            },
        )
        print("Saved config to config.json")
    except Exception as e:
        print(f"Error saving config: {e}")
