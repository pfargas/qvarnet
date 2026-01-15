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

    modelArguments = args.get_model_args
    trainingArguments = args.get_training_args
    samplerArguments = args.get_sampler_args
    optimizerArguments = args.get_optimizer_args

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
    PBC = 40.0  # Periodic Boundary Conditions

    if profile:
        jax.profiler.start_trace("/tmp/profile-data")
    params_fin, energy, _, best_params, best_energy = train(
        trainingArguments["num_epochs"],
        params,
        input_shape,
        model.apply,
        optimizer,
        sampler_params=samplerArguments,
        PBC=PBC,
        n_steps_sampler=samplerArguments["chain_length"],
    )

    if profile:
        jax.profiler.stop_trace()

    print(f"Best energy: {best_energy}")
    print(f"Best params: {best_params}")
    import matplotlib.pyplot as plt

    print(f"last energy: {energy[-1]}, before: {energy[-2]}")
    plt.plot(energy)
    plt.xlabel("Training Step")
    plt.ylabel("Energy")
    plt.show()
    plt.savefig("energy_history.png")

    # Reconstruct wavefunction
    if modelArguments["architecture"][0] != 1:
        print(
            "Cannot plot wavefunction: input dimension is not 1. Skipping wavefunction plot."
        )
        return
    else:
        x = jnp.linspace(-PBC / 2, PBC / 2, 1000).reshape(-1, 1)
        psi_approx = model.apply(params_fin, x)
        print(type(psi_approx))
        print(psi_approx.shape)
        norm = jnp.sqrt(trapezoid((psi_approx**2).squeeze(), x.squeeze()))
        print(f"Norm: {norm}")
        psi_approx = psi_approx / norm
        print(
            f"Norm after normalization: {jnp.sqrt(trapezoid((psi_approx**2).squeeze(), x.squeeze()))}"
        )
        plt.plot(x, psi_approx**2)
        plt.plot(x, jnp.pi ** (-0.5) * jnp.exp(-(x**2)), linestyle="dashed")
        plt.xlabel("x")
        plt.ylabel(r"$|\psi(x)|^2$")
        plt.legend(["VMC Approximation", "Exact Solution"])
        plt.show()
        plt.savefig("final_wavefunction.png")
