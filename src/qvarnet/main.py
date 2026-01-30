import os

from tqdm import std
from .models import MLP
from .train import train
import jax
import jax.numpy as jnp
import optax
import time


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
    base_path = create_output_directory(output_base_path, experiment_name)
    with open(os.path.join(base_path, "description.txt"), "w") as f:
        f.write(experiment_description)

    print("=" * 40)
    print(f"Results will be saved to: {base_path}")
    print("=" * 40)

    # **************************************************
    # ****                Choose model              ****
    # **************************************************
    model = MLP(architecture=model_args["architecture"])
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
    params_fin, energy_hist, _, best_energy = train(
        n_epochs=train_args["num_epochs"],
        init_params=params,
        shape=input_shape,
        model_apply=model.apply,
        optimizer=optimizer,
        sampler_params=sampler_args,
        rng_seed=master_seed,
        split_sampler=args.args.split,
        hamiltonian_params=hami_args,
    )
    time_end = time.perf_counter()

    if profile:
        jax.profiler.stop_trace()

    # remove zeroes from energy_hist
    energy_hist = energy_hist[energy_hist != 0.0]

    print(f"last energy: {energy_hist[-1]}, best {best_energy}")

    print(
        f"mean of last 10%: {jnp.mean(energy_hist[-int(0.1 * len(energy_hist)) :])}, std: {jnp.std(energy_hist[-int(0.1 * len(energy_hist)) :])}"
    )

    energy_10_percent = energy_hist[-int(0.1 * len(energy_hist)) :]
    mean_10_percent = jnp.mean(energy_10_percent)
    std_10_percent = jnp.std(energy_10_percent)

    if not save_results(
        base_path,
        energy_history=energy_hist,
        final_params=params_fin,
        final_values=[mean_10_percent, std_10_percent, time_end - time_start],
    ):
        print("Error saving results.")


def save_results(base_path, **kwargs) -> bool:
    try:
        os.makedirs(base_path, exist_ok=True)
        # Save other results as needed
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

    for key, value in kwargs.items():
        try:
            with open(os.path.join(base_path, f"{key}.txt"), "w") as f:
                for num in value:
                    f.write(str(num))
                    f.write("\n")
        except Exception as e:
            print(f"Error saving {key}: {e}")
            return False
    return True


def create_output_directory(base_path: str, experiment_name: str) -> str:
    """Create output directory for experiment results. The directory will be created in the
    specified base path. If experiment_name is not provided, a unique run ID will be generated.

    That means the relative directory structure will be:
        ./base_path/experiment_name/
    or
        ./base_path/run_000/
        ./base_path/run_001/
        ...
    depending on whether experiment_name is provided.

    Args:
        base_path: Base path where results should be saved.
        experiment_name: Name of the experiment (used for subdirectory).
    Returns:
        The full path to the created output directory."""
    if not os.path.isabs(base_path):
        cwd = os.getcwd()
        base_path = os.path.join(cwd, base_path)

    if experiment_name is None or experiment_name.strip() == "":
        directories_in_base = os.listdir(base_path)
        run_id = 0
        while f"run_{run_id:03d}" in directories_in_base:
            run_id += 1
        base_path = os.path.join(base_path, f"run_{run_id:03d}")
        os.makedirs(base_path, exist_ok=True)

    else:
        base_path = os.path.join(base_path, experiment_name)
    # if the directory does exists, throw a warning
    if os.path.exists(base_path):
        print(
            f"Warning: Directory {base_path} already exists. Creating different run ID."
        )
        directories_in_base = os.listdir(os.path.dirname(base_path))
        run_id = 0
        while f"{experiment_name}_{run_id:03d}" in directories_in_base:
            run_id += 1
        base_path = os.path.join(
            os.path.dirname(base_path), f"{experiment_name}_{run_id:03d}"
        )
    os.makedirs(base_path, exist_ok=True)
    return os.path.abspath(os.path.normpath(base_path))
