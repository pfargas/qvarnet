import sys
import os

# Compute absolute path to the src/ folder
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
src_path = os.path.join(project_root, "src")

# Add src/ to sys.path
sys.path.insert(0, src_path)

# Now you can import the code
from qvarnet.hamiltonians import GeneralHamiltonian, HarmonicOscillator
from qvarnet.models.mlp import MLP
from qvarnet.samplersv2 import MetropolisHastingsSampler
from qvarnet.utils.callback import EarlyStoppingCallback


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import time
from tqdm import tqdm
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def training(
    EPOCHS=10,
    N_SAMPLES=100_000,
    L_BOX=20.0,
    STEP_SIZE=0.1,
    BURN_IN=500,
    LEARNING_RATE=1e-3,
    MLP_LAYER_DIMS=[1, 2, 1],
):
    samples_history = []
    energy_history = []
    energy_std_history = []
    dict_state_history = []
    sampling_times = []
    energy_times = []
    acceptance_rate_history = []

    # ----------------------- DEFINE MODEL TOPOLOGY -----------------------
    model = MLP(layer_dims=MLP_LAYER_DIMS)
    model.to(device)

    model_new = MLP(layer_dims=MLP_LAYER_DIMS)
    model_new.to(device)

    # init weights
    for layer in model.children():
        if isinstance(layer, nn.Linear):

            nn.init.xavier_uniform_(layer.weight, nonlinearity="tanh")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    # ------------------------ DEFINE HAMILTONIAN -----------------------
    hamiltonian = HarmonicOscillator(model=model)
    hamiltonian.to(device)

    # ------------------------ DEFINE SAMPLER -----------------------
    sampler = MetropolisHastingsSampler(
        model=model,
        n_samples=N_SAMPLES,
        step_size=STEP_SIZE,
        burn_in=BURN_IN,
        is_wf=True,
        L_BOX=L_BOX,
    )
    sampler.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    x0 = torch.tensor([0.0], device=device)  # Initial point for the sampler

    for _ in tqdm(range(EPOCHS)):
        optimizer.zero_grad()

        dict_state_history.append(model.state_dict())

        # Run sampler
        sampler.model = model
        start_sample_time = time.time()
        samples = sampler(x0, method="parallel", n_walkers=N_SAMPLES)
        samples.requires_grad = True
        end_sample_time = time.time()

        sampling_times.append(end_sample_time - start_sample_time)
        samples_history.append(samples)
        acceptance_rate_history.append(sampler.get_acceptance_rate())

        # Compute the mean and std of the local energy
        start_energy_time = time.time()
        hamiltonian.model = model
        local_energy = hamiltonian(samples)
        end_energy_time = time.time()

        energy_times.append(end_energy_time - start_energy_time)

        loss = local_energy.mean()

        energy = copy.deepcopy(loss.item())
        energy_std = local_energy.std().item()
        energy_history.append(energy)
        energy_std_history.append(energy_std)

        # Compute gradients
        loss.squeeze().backward()
        # Update model parameters
        optimizer.step()

        sampler.reset_statistics()

    print("Ended training.")
    return {
        "samples_history": samples_history,
        "energy_history": energy_history,
        "dict_state_history": dict_state_history,
        "sampling_times": sampling_times,
        "energy_times": energy_times,
        "energy_std_history": energy_std_history,
        "acceptance_rate_history": acceptance_rate_history,
    }


def compute_mean_std(times):
    mean_time = np.mean(times)
    std_time = np.std(times)
    return mean_time, std_time


def plot_confidence_interval(x, data, error, **kwargs):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if not isinstance(error, np.ndarray):
        error = np.array(error)
    lower_bound = data - error
    upper_bound = data + error
    kwargs_fill = copy.deepcopy(kwargs)
    if "label" in kwargs:
        kwargs_fill.pop("label")
    plt.fill_between(x, lower_bound, upper_bound, alpha=0.2, **kwargs_fill)
    plt.plot(x, data, **kwargs)


def test_burnin_times(
    min_burnin=50,
    max_burnin=1000,
    step_burnin=50,
    epochs=10,
    n_samples=100_000,
    layer_dims=[1, 2, 1],
):
    burn_in = range(min_burnin, max_burnin + 1, step_burnin)
    results_burnin = []
    for b in burn_in:
        print(f"Training with burn-in: {b}")
        result = training(
            EPOCHS=epochs,
            N_SAMPLES=n_samples,
            L_BOX=20.0,
            STEP_SIZE=0.5,
            BURN_IN=b,
            LEARNING_RATE=1e-3,
            MLP_LAYER_DIMS=layer_dims,
        )
        results_burnin.append(result)
    burnin_energy_times = []
    burnin_sampling_times = []
    burnin_std_energy_times = []
    burnin_std_sampling_times = []

    for result in results_burnin:
        energy_times = result["energy_times"]
        sampling_times = result["sampling_times"]

        mean_energy, std_energy = compute_mean_std(energy_times)
        mean_sampling, std_sampling = compute_mean_std(sampling_times)

        burnin_energy_times.append(mean_energy)
        burnin_std_energy_times.append(std_energy)
        burnin_sampling_times.append(mean_sampling)
        burnin_std_sampling_times.append(std_sampling)

    plt.figure(figsize=(12, 6))
    plot_confidence_interval(
        burn_in, burnin_energy_times, burnin_std_energy_times, label="Mean Energy Time"
    )
    plot_confidence_interval(
        burn_in,
        burnin_sampling_times,
        burnin_std_sampling_times,
        label="Mean Sampling Time",
    )
    plt.xlabel("Burn-in Steps")
    plt.ylabel("Time (seconds)")
    plt.xticks(burn_in[::2], rotation=45)
    plt.legend()


def test_sampling_with_burnins(
    min_burnin=50,
    max_burnin=1050,
    step_burnin=50,
    n_samples=100_000,
    layer_dims=[1, 2, 1],
):
    burn_in = range(min_burnin, max_burnin, step_burnin)
    results_burnin_2 = []
    for b in burn_in:
        print(f"Training with burn-in: {b}")
        result = training(
            EPOCHS=1,
            N_SAMPLES=n_samples,
            L_BOX=20.0,
            STEP_SIZE=0.5,
            BURN_IN=b,
            LEARNING_RATE=1e-3,
            MLP_LAYER_DIMS=layer_dims,
        )
        results_burnin_2.append(result)

    x_linspace = torch.linspace(-10, 10, 1000, device=device).reshape(-1, 1)
    print(len(results_burnin_2))

    # generate a big pic of 4x5

    plt.figure(figsize=(20, 16))
    # Plot the results

    for i in range(len(results_burnin_2)):
        plt.subplot(4, 5, i + 1)
        plt.title(f"Burn-in: {burn_in[i]}")
        result = results_burnin_2[i]
        samples = result["samples_history"]

        model = MLP(layer_dims=[1, 2, 1])
        model.load_state_dict(result["dict_state_history"][0])
        model.to(device)
        y = model(x_linspace).detach().cpu().numpy()

        normalization_constant = np.sqrt(
            np.sum(y**2) * (x_linspace[1] - x_linspace[0]).item()
        )
        y /= normalization_constant

        plt.plot(x_linspace.cpu().numpy(), y**2, label=f"Sample {i+1}")
        samples = samples[0].detach().cpu().numpy()
        plt.hist(samples, bins=50, density=True, alpha=0.5, label=f"Histogram {i+1}")
    plt.tight_layout()

    plt.show()


def test_n_samples(
    min_samples=10_000,
    max_samples=2_600_000,
    step_samples=50_000,
    epochs=10,
    layer_dims=[1, 2, 1],
):
    results_n_samples = []
    n_samples = range(min_samples, max_samples, step_samples)

    for n in n_samples:
        print(f"Training with n_samples: {n}")
        result = training(
            EPOCHS=epochs,
            N_SAMPLES=n,
            L_BOX=20.0,
            STEP_SIZE=0.5,
            BURN_IN=100,
            LEARNING_RATE=1e-3,
            MLP_LAYER_DIMS=layer_dims,
        )
        results_n_samples.append(result)
    print(len(results_n_samples))

    samples_times = []
    mean_energy_times = []
    samples_times_std = []
    energy_times_std = []

    for result in results_n_samples:
        sampling_times = result["sampling_times"]
        energy_times = result["energy_times"]
        mean_sampling, std_sampling = compute_mean_std(sampling_times)
        mean_energy, std_energy = compute_mean_std(energy_times)
        samples_times.append(mean_sampling)
        samples_times_std.append(std_sampling)
        mean_energy_times.append(mean_energy)
        energy_times_std.append(std_energy)

    print(len(mean_energy_times))
    print(len(energy_times_std))

    plt.figure(figsize=(12, 6))
    plot_confidence_interval(
        n_samples, samples_times, samples_times_std, label="Mean Sampling Time"
    )
    plot_confidence_interval(
        n_samples, mean_energy_times, energy_times_std, label="Mean Energy Time"
    )
