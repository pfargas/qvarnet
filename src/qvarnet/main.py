import os

from tqdm import std

from .models import MLP, ExponentialMLPwithPenalty
from .train import train
import jax
import jax.numpy as jnp
import optax
import time
from .utils import save_flax_to_json, save_results, create_output_directory


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
    experiment_description = exp_info.get("description", "No description provided.")
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
    # model = MLP(architecture=model_args["architecture"])
    model = ExponentialMLPwithPenalty(architecture=model_args["architecture"])
    # **************************************************

    if optimizer_args["type"] == "adam":
        optimizer = optax.adam(learning_rate=optimizer_args["learning_rate"])
    elif optimizer_args["type"] == "sgd":
        optimizer = optax.sgd(learning_rate=optimizer_args["learning_rate"])
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_args['type']}")

    rng = jax.random.PRNGKey(0)  # Random key for parameter initialization
    input_shape = (
        train_args["batch_size"],
        model_args["architecture"][0],
    )
    params = model.init(rng, jnp.ones(input_shape) * 0.1)  # Initialize parameters

    if profile:
        jax.profiler.start_trace("/tmp/profile-data")

    time_start = time.perf_counter()
    energy_hist, best_state = train(
        n_epochs=train_args["num_epochs"],
        init_params=params,
        shape=input_shape,
        model_apply=model.apply,
        optimizer=optimizer,
        sampler_params=sampler_args,
        rng_seed=master_seed,
        hamiltonian_params=hami_args,
        checkpoint_path=base_path,
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
    print("Saving best params")
    try:
        save_flax_to_json(best_params, os.path.join(base_path, "best_params.json"))
    except Exception as e:
        print(f"Error saving best params: {e}")

    if not save_results(
        base_path,
        energy_history=energy_hist,
        final_values=[
            best_state.energy,
            best_state.std,
            f"Total training time: {time_end - time_start} seconds",
        ],
    ):
        print("Error saving results.")
