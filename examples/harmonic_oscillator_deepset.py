"""
Example: Variational Monte Carlo for Harmonic Oscillator with DeepSet Architecture

This script demonstrates how to use the refactored qvarnet package to optimize
a neural quantum state for a harmonic oscillator system (10 particles in 1D).

**Key Features**:
1. DeepSet architecture with configurable φ and F networks
2. Log-domain training (is_log_model=True) for numerical stability
3. Comprehensive wavefunction visualization and analysis
4. Comparison with exact analytical solution

Model operates in log-domain for stability:
- Model outputs log|ψ(x)| instead of ψ(x)
- Probability is computed as |ψ|² = exp(2 * log|ψ|)
- Local energy uses log-space kinetic energy formula

Run this script to train the model and generate a dashboard showing:
- Energy convergence vs exact ground state
- Learned wavefunction amplitude vs exact solution
- Single-particle and pair-particle correlations
"""

import os
import json
from pathlib import Path
from datetime import datetime

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Import refactored modules
from qvarnet.models.deep_set import DeepSet
from qvarnet.hamiltonian.continuous import (
    HarmonicOscillatorHamiltonian,
    NN_OscillatorHamiltonian,
)
from qvarnet.train import train
from qvarnet.probability import build_prob_fn
from qvarnet.sampling_step import sample_and_process
from qvarnet.training_step import compute_step
from qvarnet.vmc_state import VMCState
from qvarnet.config.training_setup import parse_sampler_params


def create_deepset_model(
    n_particles: int = 10,
    n_dim: int = 1,
    phi_hidden_units: int = 16,
    f_hidden_units: int = 16,
    shared_dim: int = 16,
) -> DeepSet:
    """
    Create a DeepSet model for quantum state representation.

    Architecture:
    - φ network (per-particle): 1D → φ_hidden → shared_dim
    - F network (aggregation): shared_dim → F_hidden → 1 (log amplitude)
    - Final output: exp(F(Σφ(x_i)))

    Args:
        n_particles: Number of particles
        n_dim: Spatial dimensions per particle
        phi_hidden_units: Hidden units in φ network
        f_hidden_units: Hidden units in F network
        shared_dim: Dimension of aggregated representation

    Returns:
        DeepSet model
    """
    return DeepSet(
        n_particles=n_particles,
        n_dim=n_dim,
        phi_hidden_architecture=[phi_hidden_units],  # Single hidden layer
        F_hidden_architecture=[f_hidden_units],  # Single hidden layer
        hidden_internal_dimension=shared_dim,  # Aggregation dimension
        phi_hidden_activation=nn.tanh,
        F_hidden_activation=nn.tanh,
        kernel_init=nn.initializers.lecun_normal(),
        bias_init=nn.initializers.zeros_init(),
    )


def create_experiment_config(system_name: str, n_epochs: int = 500):
    """Create experiment configuration dictionary."""
    return {
        "experiment": {
            "name": f"harmonic_{system_name}_deepset",
            "description": f"VMC optimization for {system_name} with DeepSet model",
            "seed": 42,
        },
        "model": {
            "type": "deep-set",
            "n_particles": 10,
            "n_dim": 1,
        },
        "training": {
            "num_epochs": n_epochs,
            "batch_size": 500,
            "init_positions": "normal",
        },
        "optimizer": {
            "type": "adam",
            "learning_rate": 1e-3,
        },
        "sampler": {
            "step_size": 0.5,
            "chain_length": 200,
            "thermalization_steps": 50,
            "thinning_factor": 2,
            "PBC": 40.0,
        },
        "hamiltonian": {
            "name": system_name,
            "params": {
                "omega": 1.0 if "nn" not in system_name else 1.0,
                "omega_interaction": 0.1 if "nn" in system_name else None,
            },
        },
        "output": {
            "save_dir": "./results",
            "save_checkpoints": False,
        },
    }


def train_deepset_model(
    hamiltonian,
    system_name: str,
    n_epochs: int = 500,
    seed: int = 42,
) -> dict:
    """
    Train a DeepSet model using Variational Monte Carlo in log-domain.

    The model operates in log-space for numerical stability:
    - Model outputs: log|ψ(x)|
    - Probability: |ψ|² = exp(2 * log|ψ|)
    - Local energy computed using log-space kinetic energy formula

    Args:
        hamiltonian: Hamiltonian object (HarmonicOscillatorHamiltonian or similar)
        system_name: Name of the system (for logging)
        n_epochs: Number of training epochs
        seed: Random seed

    Returns:
        Dictionary with training results including energy history and final state
    """
    print(f"\n{'='*60}")
    print(f"Training DeepSet for {system_name} (LOG-DOMAIN)")
    print(f"{'='*60}")

    # Model and training setup
    model = create_deepset_model()
    n_particles = 10
    n_dim = 1
    shape = (4, n_particles * n_dim)  # 4 chains, 10 DoF

    # Optimizer
    optimizer = optax.adam(learning_rate=1e-3)

    # Sampler configuration
    sampler_params = {
        "step_size": 0.5,
        "chain_length": 200,
        "thermalization_steps": 50,
        "thinning_factor": 2,
        "PBC": 40.0,
    }

    # Train in log-domain for numerical stability
    key = jax.random.PRNGKey(seed)
    state_history = train(
        n_epochs=n_epochs,
        shape=shape,
        model=model,
        optimizer=optimizer,
        sampler_params=sampler_params,
        hamiltonian=hamiltonian,
        rng_seed=seed,
        checkpoint_path="./checkpoints",
        save_checkpoints=False,
        is_log_model=True,  # Use log-domain (model outputs log|ψ|)
    )

    # Extract results
    # state_history is a list of VMCState objects, extract energy from each
    energies = jnp.array([float(state.energy) for state in state_history])

    # Generate samples from the trained model
    print(f"    Generating samples from trained model...")
    samples = sample_from_model(
        model, state_history[-1].params, n_samples=5000, seed=seed
    )

    # Check wavefunction symmetry
    print(f"    Checking wavefunction symmetry...")
    symmetry_metrics = check_wavefunction_symmetry(model, state_history[-1].params)

    return {
        "system": system_name,
        "model": model,
        "energies": energies,
        "n_epochs": len(energies),
        "final_energy": float(energies[-1]) if len(energies) > 0 else 0.0,
        "min_energy": float(jnp.min(energies)) if len(energies) > 0 else 0.0,
        "std_final": (float(jnp.std(energies[-50:])) if len(energies) > 50 else 0.0),
        "samples": samples,
        "symmetry": symmetry_metrics,
    }


def check_wavefunction_symmetry(
    model,
    params,
    n_test_configs: int = 20,
) -> dict:
    """
    Check permutation symmetry of the learned wavefunction.

    For DeepSet, the wavefunction should be invariant under permutations of particles:
    |ψ(x₁, x₂, ..., x₁₀)| = |ψ(x₂, x₁, ..., x₁₀)| (any permutation)

    Args:
        model: Trained Flax model
        params: Model parameters
        n_test_configs: Number of random configurations to test

    Returns:
        Dictionary with permutation symmetry metrics
    """
    from qvarnet.probability import build_prob_fn

    prob_fn = build_prob_fn(model.apply, is_log_model=True)

    key = jax.random.PRNGKey(42)
    perm_errors = []

    for _ in range(n_test_configs):
        # Generate random configuration
        key, subkey = jax.random.split(key)
        config = jax.random.normal(subkey, (10,)) * 0.5

        # Compute probability of original config
        prob_orig = float(np.exp(2 * prob_fn(config, params)))

        # Test a few random permutations
        for _ in range(3):
            key, subkey = jax.random.split(key)
            perm_indices = jax.random.permutation(subkey, 10)
            permuted_config = config[perm_indices]

            # Compute probability of permuted config
            prob_perm = float(np.exp(2 * prob_fn(permuted_config, params)))

            # Relative error
            if max(prob_orig, prob_perm) > 1e-10:
                rel_error = abs(prob_orig - prob_perm) / max(prob_orig, prob_perm)
                perm_errors.append(rel_error)

    perm_errors = np.array(perm_errors)

    return {
        "mean_perm_error": float(np.mean(perm_errors)),
        "max_perm_error": float(np.max(perm_errors)),
        "min_perm_error": float(np.min(perm_errors)),
        "std_perm_error": float(np.std(perm_errors)),
    }


def sample_from_model(
    model,
    params,
    n_samples: int = 5000,
    seed: int = 42,
) -> jnp.ndarray:
    """
    Generate samples from the trained model using MCMC sampling.

    Args:
        model: Trained Flax model
        params: Model parameters
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        Samples of shape (n_samples, DoF)
    """
    from qvarnet.probability import build_prob_fn
    from qvarnet.samplers import mh_chain

    key = jax.random.PRNGKey(seed)
    prob_fn = build_prob_fn(model.apply, is_log_model=True)

    # Initialize positions
    shape = (10,)  # 10 particles in 1D
    init_pos = jax.random.normal(key, shape) * 0.5

    # Generate samples using MH
    # Need more steps due to thinning factor
    n_steps = n_samples * 5  # Account for thinning
    rand_nums = jax.random.uniform(key, (n_steps, 11))

    # Call mh_chain with positional arguments in correct order:
    # (random_values, PBC, prob_fn, prob_params, init_position, step_size, is_log_prob)
    samples, _ = mh_chain(
        rand_nums,  # random_values
        40.0,  # PBC
        prob_fn,  # prob_fn
        params,  # prob_params
        init_pos,  # init_position
        0.5,  # step_size
        True,  # is_log_prob
    )

    # Apply thinning (keep every 5th sample)
    return samples[::5][:n_samples]


def create_dashboard(results_list: list, save_path: str = "vmc_dashboard.png"):
    """
    Create a comprehensive dashboard for harmonic oscillator analysis.

    Args:
        results_list: List containing single result dictionary from train_deepset_model()
        save_path: Path to save the dashboard figure
    """
    results = results_list[0]  # Use only the first (and only) system

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.38, wspace=0.3)

    # Color palette
    color_neural = "#1f77b4"
    color_exact = "#d62728"

    # ========== Row 1: Energy convergence (span 2 cols) ==========
    ax1 = fig.add_subplot(gs[0, :2])
    energies = np.array(results["energies"])
    ax1.plot(
        energies,
        label="Neural Network Energy",
        linewidth=2.5,
        color=color_neural,
        alpha=0.8,
    )
    ax1.fill_between(
        range(len(energies)),
        energies,
        alpha=0.15,
        color=color_neural,
    )
    # Add theoretical ground state line
    ax1.axhline(
        y=5.0,
        color=color_exact,
        linestyle="--",
        linewidth=2.5,
        label="Exact E₀ = 5.0",
        alpha=0.8,
    )

    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Energy", fontsize=12, fontweight="bold")
    ax1.set_title("Energy Convergence During Training", fontsize=13, fontweight="bold")
    ax1.legend(loc="best", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ========== Row 1, Col 3: Training info ==========
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis("off")
    info_text = f"Training Summary\n\n"
    info_text += f"Final E: {results['final_energy']:.6f}\n"
    info_text += f"Min E: {results['min_energy']:.6f}\n"
    info_text += f"Exact E₀: 5.000000\n"
    info_text += f"Error: {abs(results['final_energy'] - 5.0):.6f}\n"
    info_text += f"Epochs: {results['n_epochs']}\n\n"
    info_text += f"System: 10 particles, 1D\n"
    info_text += f"ω = 1.0\n"
    ax_info.text(
        0.05,
        0.5,
        info_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.6),
    )
    ax_info.set_title("Training Info", fontsize=12, fontweight="bold")

    # ========== Row 2, Col 1: Single-particle marginal density ==========
    ax2 = fig.add_subplot(gs[1, 0])
    if "samples" in results:
        samples = np.array(results["samples"])
        # Extract first particle coordinate
        particle_1_samples = samples[:, 0]
        ax2.hist(
            particle_1_samples,
            bins=35,
            alpha=0.7,
            label="Neural Network (particle 1)",
            color=color_neural,
            density=True,
            edgecolor="black",
            linewidth=0.8,
        )

    # Overlay exact harmonic oscillator ground state
    x_range = np.linspace(-4, 4, 150)
    psi0 = (1.0 / np.pi) ** 0.25 * np.exp(-0.5 * x_range**2)
    psi0_squared = psi0**2
    ax2.plot(
        x_range,
        psi0_squared,
        color=color_exact,
        linewidth=2.5,
        label="Exact |ψ₀|²",
        alpha=0.9,
        linestyle="--",
    )

    ax2.set_xlabel("Position", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax2.set_title("Single-Particle Marginal Density", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10, loc="best")
    ax2.grid(True, alpha=0.3)

    # ========== Row 2, Col 2: Pair correlation (particles 0-1) ==========
    ax3 = fig.add_subplot(gs[1, 1])
    if "samples" in results:
        samples = np.array(results["samples"])
        # Extract pair: particles 0 and 1
        p0 = samples[:, 0]
        p1 = samples[:, 1]

        # 2D histogram of pair correlation
        h = ax3.hist2d(p0, p1, bins=30, cmap="viridis", alpha=0.9)
        ax3.set_xlabel("Particle 0 Position", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Particle 1 Position", fontsize=11, fontweight="bold")
        ax3.set_title(
            "Pair Correlation (Particles 0 & 1)", fontsize=12, fontweight="bold"
        )
        cbar = plt.colorbar(h[3], ax=ax3, label="Count")
        cbar.ax.tick_params(labelsize=9)

    # ========== Row 2, Col 3: Smoothed energy convergence ==========
    ax_smooth = fig.add_subplot(gs[1, 2])
    energies = np.array(results["energies"])
    window = min(30, len(energies) // 10)
    if window > 1:
        running_avg = np.convolve(energies, np.ones(window) / window, mode="valid")
        ax_smooth.plot(
            range(len(running_avg)),
            running_avg,
            linewidth=2.5,
            label="Running Average",
            color=color_neural,
            alpha=0.8,
        )
    ax_smooth.axhline(
        y=5.0,
        color=color_exact,
        linestyle="--",
        linewidth=2.5,
        label="Exact E₀ = 5.0",
        alpha=0.8,
    )
    ax_smooth.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_smooth.set_ylabel("Energy", fontsize=11, fontweight="bold")
    ax_smooth.set_title("Smoothed Convergence", fontsize=12, fontweight="bold")
    ax_smooth.legend(fontsize=9)
    ax_smooth.grid(True, alpha=0.3)

    # ========== Row 3, Col 1: Energy metrics comparison ==========
    ax4 = fig.add_subplot(gs[2, 0])
    energy_types = ["Final\nEnergy", "Min\nEnergy", "Exact\nE₀"]
    energy_values = [results["final_energy"], results["min_energy"], 5.0]
    colors_bars = [color_neural, color_neural, color_exact]

    bars = ax4.bar(
        energy_types,
        energy_values,
        color=colors_bars,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
        width=0.6,
    )

    ax4.set_ylabel("Energy", fontsize=11, fontweight="bold")
    ax4.set_title("Energy Comparison", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_ylim([4.5, 5.5])

    # Add value labels on bars
    for bar, val in zip(bars, energy_values):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # ========== Row 3, Col 2: Training statistics table ==========
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    energies = np.array(results["energies"])
    last_50_std = float(np.std(energies[-50:])) if len(energies) > 50 else 0.0

    table_data = [
        ["Metric", "Value"],
        ["Final Energy", f"{results['final_energy']:.6f}"],
        ["Min Energy", f"{results['min_energy']:.6f}"],
        ["Exact E₀", "5.000000"],
        ["Error", f"{abs(results['final_energy'] - 5.0):.6f}"],
        ["Δ E (Last 50)", f"{last_50_std:.6f}"],
        ["Epochs", f"{results['n_epochs']}"],
    ]

    table = ax5.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.45, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor("#2E86AB")
        table[(0, i)].set_text_props(weight="bold", color="white", fontsize=9)

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")
            else:
                table[(i, j)].set_facecolor("#ffffff")

    ax5.set_title("Detailed Statistics", fontsize=11, fontweight="bold", pad=10)

    # ========== Row 3, Col 3: Energy distribution ==========
    ax6 = fig.add_subplot(gs[2, 2])
    energies = np.array(results["energies"])
    # Use last 150 epochs for distribution
    recent_energies = energies[-150:] if len(energies) > 150 else energies
    ax6.hist(
        recent_energies,
        bins=25,
        alpha=0.7,
        color=color_neural,
        edgecolor="black",
        linewidth=1,
    )
    ax6.axvline(
        x=5.0,
        color=color_exact,
        linestyle="--",
        linewidth=2.5,
        label="Exact E₀ = 5.0",
        alpha=0.9,
    )

    ax6.set_xlabel("Energy", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax6.set_title(
        "Energy Distribution (Last 150 Epochs)", fontsize=12, fontweight="bold"
    )
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis="y")

    # ========== Row 4: Wavefunction amplitude on grid (span all columns) ==========
    ax7 = fig.add_subplot(gs[3, :])
    if "samples" in results:
        samples = np.array(results["samples"])
        # Create 1D grid for position
        x_grid = np.linspace(-4.5, 4.5, 150)

        # Compute wavefunction amplitude via kernel density estimation
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(samples[:, 0])  # Use first particle
        psi_squared = kde(x_grid)

        ax7.plot(
            x_grid,
            np.sqrt(np.abs(psi_squared)),
            linewidth=3,
            label="Neural Network |ψ|",
            color=color_neural,
            alpha=0.9,
        )
        ax7.fill_between(
            x_grid, np.sqrt(np.abs(psi_squared)), alpha=0.2, color=color_neural
        )

    # Plot exact ground state
    x_exact = np.linspace(-4.5, 4.5, 150)
    psi0_exact = (1.0 / np.pi) ** 0.25 * np.exp(-0.5 * x_exact**2)
    ax7.plot(
        x_exact,
        psi0_exact,
        color=color_exact,
        linewidth=3,
        linestyle="--",
        label="Exact Ground State |ψ₀|",
        alpha=0.9,
    )

    ax7.set_xlabel("Position (x)", fontsize=12, fontweight="bold")
    ax7.set_ylabel("Wavefunction Amplitude |ψ|", fontsize=12, fontweight="bold")
    ax7.set_title(
        "Learned vs. Exact Wavefunction Amplitude (Single-Particle Marginal)",
        fontsize=13,
        fontweight="bold",
    )
    ax7.legend(loc="best", fontsize=11, framealpha=0.95)
    ax7.grid(True, alpha=0.3)

    # Main title
    fig.suptitle(
        "DeepSet VMC for Harmonic Oscillator: Learning the Wavefunction\n"
        "10 Particles in 1D | ω=1.0 | Neural Network vs. Exact Solution",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Dashboard saved to: {save_path}")
    print(f"  Figure size: 18x14 inches @ 300 DPI")

    return fig


def main():
    """Main execution: train model and compare with analytical solution."""
    print("\n" + "=" * 70)
    print("VARIATIONAL MONTE CARLO WITH DEEPSET ARCHITECTURE")
    print("Harmonic Oscillator | Comparison with Analytical Solution")
    print("=" * 70)

    # Training parameters
    n_epochs = 10_000
    seed = 42

    # Simple harmonic oscillator
    print("\nSetting up harmonic oscillator (ω=1.0, 10 particles, 1D)...")
    hamiltonian_ho = HarmonicOscillatorHamiltonian(omega=1.0)
    results_ho = train_deepset_model(
        hamiltonian=hamiltonian_ho,
        system_name="harmonic-oscillator",
        n_epochs=n_epochs,
        seed=seed,
    )
    print(f"  ✓ Final energy: {results_ho['final_energy']:.6f}")
    print(f"  ✓ Min energy:   {results_ho['min_energy']:.6f}")
    print(f"  ✓ Epochs:       {results_ho['n_epochs']}")
    print(f"\nPermutation Symmetry Check (DeepSet invariance):")
    print(f"  ✓ Mean error:  {results_ho['symmetry']['mean_perm_error']:.2e}")
    print(f"  ✓ Max error:   {results_ho['symmetry']['max_perm_error']:.2e}")
    print(f"  ✓ Std error:   {results_ho['symmetry']['std_perm_error']:.2e}")

    # Create dashboard
    print("\nGenerating dashboard with wavefunction analysis...")
    results_list = [results_ho]
    fig = create_dashboard(results_list, save_path="vmc_dashboard.png")

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nEnergy Results:")
    print(f"  Neural Network Final Energy:  {results_ho['final_energy']:.6f}")
    print(f"  Theoretical Ground State (E₀ = D/2 = 5.0): 5.000000")
    print(f"  Error: {abs(results_ho['final_energy'] - 5.0):.6f}")
    print(f"\nPermutation Symmetry (DeepSet Invariance):")
    print(f"  Relative error under random particle permutations:")
    print(f"    Mean:  {results_ho['symmetry']['mean_perm_error']:.2e}")
    print(f"    Max:   {results_ho['symmetry']['max_perm_error']:.2e}")
    print(f"    Std:   {results_ho['symmetry']['std_perm_error']:.2e}")
    print(
        f"  Interpretation: {'✓ Perfect symmetry' if results_ho['symmetry']['mean_perm_error'] < 1e-6 else '✓ Excellent symmetry' if results_ho['symmetry']['mean_perm_error'] < 1e-4 else '✓ Good symmetry' if results_ho['symmetry']['mean_perm_error'] < 1e-2 else '⚠ Imperfect symmetry'}"
    )
    print(f"\nDashboard saved as: vmc_dashboard.png")
    print("\nKey visualizations:")
    print("  • Energy convergence during training")
    print("  • Single-particle marginal density")
    print("  • Pair correlation between particles")
    print("  • Energy distribution (last 100 epochs)")
    print("  • Training statistics")
    print("  • Learned wavefunction amplitude vs. exact solution")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
