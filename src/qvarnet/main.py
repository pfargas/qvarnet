import os
from .models import MLP, WavefunctionOneParameter, ExponentialWavefunction
from .train import train
import jax
import jax.numpy as jnp
import optax
from jax.scipy.integrate import trapezoid


def run_experiment(args=None, profile=False):
    """Run a quantum variational Monte Carlo experiment.
    Args:

        args: An object containing model, training, sampler, and optimizer arguments.
        profile: Boolean flag to enable JAX profiling.
    Returns:
        None"""
    if args is None:
        raise ValueError("Arguments must be provided to run_experiment")

    modelArguments = args.get_model_args()
    trainingArguments = args.get_training_args()
    samplerArguments = args.get_sampler_args()
    optimizerArguments = args.get_optimizer_args()
    outputArguments = args.get_output_args()
    master_seed = args.get_seed()

    output_base_path = outputArguments.get("save_dir", "tmp/qvarnet/results")
    os.makedirs(output_base_path, exist_ok=True)
    directories_in_base = os.listdir(output_base_path)
    run_id = 0
    while f"run_{run_id:03d}" in directories_in_base:
        run_id += 1
    base_path = os.path.join(output_base_path, f"run_{run_id:03d}")
    os.makedirs(base_path, exist_ok=True)
    # **************************************************
    # ****                Choose model              ****
    # **************************************************
    model = MLP(architecture=modelArguments["architecture"])
    # model = ExponentialWavefunction()
    # model = WavefunctionOneParameter()
    # **************************************************

    if optimizerArguments["type"] == "adam":
        optimizer = optax.adam(learning_rate=optimizerArguments["learning_rate"])
    elif optimizerArguments["type"] == "sgd":
        optimizer = optax.sgd(learning_rate=optimizerArguments["learning_rate"])
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizerArguments['type']}")

    rng = jax.random.PRNGKey(0)  # Random key for parameter initialization
    input_shape = (
        trainingArguments["batch_size"],
        modelArguments["architecture"][0],
    )
    params = model.init(rng, jnp.ones(input_shape) * 0.1)  # Initialize parameters
    PBC = samplerArguments.get("PBC", 40.0)  # Periodic Boundary Conditions

    if profile:
        jax.profiler.start_trace("/tmp/profile-data")

    print("SANITY CHECK: PARAMS USED")
    print("params: ", params)
    print("input_shape: ", input_shape)
    print("optimizer: ", optimizer)
    print("samplerArguments: ", samplerArguments)
    print("seed: ", master_seed)

    params_fin, energy_hist, _, _ = train(
        n_epochs=trainingArguments["num_epochs"],
        init_params=params,
        shape=input_shape,
        model_apply=model.apply,
        optimizer=optimizer,
        sampler_params=samplerArguments,
        rng_seed=master_seed,
    )

    if profile:
        jax.profiler.stop_trace()

    # print(f"Best energy: {best_energy}")
    # print(f"Best params: {best_params}")
    import matplotlib.pyplot as plt

    print(f"last energy: {energy_hist[-1]}, before: {energy_hist[-2]}")
    plt.plot(energy_hist)
    plt.xlabel("Training Step")
    plt.ylabel("Energy")
    plt.savefig(f"{base_path}/energy_history.png")
    if not save_results(base_path, energy_hist=energy_hist, params_fin=params_fin):
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
            jnp.save(os.path.join(base_path, f"{key}.npy"), value)
        except Exception as e:
            print(f"Error saving {key}: {e}")
            return False
    return True
