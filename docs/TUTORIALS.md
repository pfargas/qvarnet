# Usage Examples and Tutorials

This section provides practical examples and tutorials for using QVarNet in various research scenarios.

## Quick Start Example

### Basic 1D Harmonic Oscillator

```python
import jax
import jax.numpy as jnp
from qvarnet.models import MLP
from qvarnet.main import run_experiment
from qvarnet.cli.run import FileParser
import optax

# Create a simple configuration
config = {
    "optimizer": {
        "optimizer_type": "adam",
        "learning_rate": 1e-3
    },
    "training": {
        "batch_size": 1000,
        "num_epochs": 2000
    },
    "model": {
        "architecture": [1, 32, 16, 1],  # 1D input
        "activation": "tanh"
    },
    "sampler": {
        "step_size": 0.5,
        "chain_length": 100
    }
}

# Save configuration
import json
with open("simple_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Load and run
parser = FileParser("simple_config.json")
parser.parse()

# Run experiment
params_fin, energy_history, _, best_params, best_energy = run_experiment(parser)

print(f"Best energy: {best_energy:.6f}")
print(f"Exact ground state: 0.5")  # For 1D harmonic oscillator
```

## Advanced Examples

### 2D Quantum Harmonic Oscillator

```python
import jax
import jax.numpy as jnp
from qvarnet.models import MLP
from qvarnet.train import train
from qvarnet.sampler import mh_chain
import optax
import matplotlib.pyplot as plt

# Configuration for 2D oscillator
config = {
    "optimizer": {"optimizer_type": "adam", "learning_rate": 5e-4},
    "training": {"batch_size": 2000, "num_epochs": 5000},
    "model": {"architecture": [2, 64, 64, 32, 1], "activation": "tanh"},
    "sampler": {"step_size": 0.3, "chain_length": 150}
}

# Set up model and optimizer
model = MLP(architecture=config["model"]["architecture"])
optimizer = optax.adam(learning_rate=config["optimizer"]["learning_rate"])

# Initialize parameters
rng = jax.random.PRNGKey(42)
input_shape = (config["training"]["batch_size"], 2)  # 2D
params = model.init(rng, jnp.ones(input_shape) * 0.1)

# Train
params_fin, energy_history, _, best_params, best_energy = train(
    n_steps=config["training"]["num_epochs"],
    init_params=params,
    shape=input_shape,
    model_apply=model.apply,
    optimizer=optimizer,
    sampler_params=config["sampler"],
    PBC=20.0,
    n_steps_sampler=config["sampler"]["chain_length"]
)

print(f"2D Harmonic Oscillator Results:")
print(f"Best energy: {best_energy:.6f}")
print(f"Exact ground state: 1.0")  # E = (n_x + n_y + 1)/2 for ground state
```

### Custom Wavefunction Ansatz

```python
from flax import linen as nn
import jax.numpy as jnp

class CustomWavefunction(nn.Module):
    """Custom ansatz inspired by Slater determinant structure"""
    features: int = 32
    n_particles: int = 2
    
    @nn.compact
    def __call__(self, x):
        # Input x shape: (batch, n_particles * dimensions)
        batch_size = x.shape[0]
        
        # Reshape for particle-wise processing
        x_reshaped = x.reshape(batch_size, self.n_particles, -1)
        
        # Particle-wise encoding
        particle_features = []
        for i in range(self.n_particles):
            xi = x_reshaped[:, i, :]  # (batch, dimensions)
            particle_enc = nn.Dense(self.features)(xi)
            particle_enc = nn.tanh(particle_enc)
            particle_features.append(particle_enc)
        
        # Combine particle features
        combined = jnp.stack(particle_features, axis=1)  # (batch, n_particles, features)
        combined = jnp.mean(combined, axis=1)  # Mean over particles
        
        # Final output
        output = nn.Dense(1)(combined)
        
        return output.squeeze()

# Use the custom model
model = CustomWavefunction(features=64, n_particles=2)

# Continue with training as before...
```

## Performance Optimization

### GPU Memory Management

```python
import jax
import os

# Configure memory usage for large simulations
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  # Use 90% of GPU memory

# Enable memory preallocation (reduces fragmentation)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

# Enable async dispatch for better performance
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math"

# Verify configuration
print(f"JAX devices: {jax.devices()}")
print(f"Memory fraction: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'default')}")
```

### Sampling Performance Tuning

```python
from qvarnet.sampler import mh_chain
from functools import partial
import jax

# Optimize sampler for your specific problem
@partial(jax.jit, static_argnames=("prob_fn", "n_chains"))
def optimized_sampler(random_keys, prob_fn, prob_params, init_positions, 
                     step_size, PBC, n_chains):
    """Vectorized sampler with optimized memory usage"""
    
    @partial(jax.jit, static_argnames=("prob_fn",))
    def single_chain(key, init_pos, step_size):
        return mh_chain(key, PBC, prob_fn, prob_params, init_pos, step_size)
    
    # Vectorize over chains
    batch_sampler = jax.vmap(single_chain, in_axes=(0, 0, None))
    
    return batch_sampler(random_keys, init_positions, step_size)

# Usage in training
def optimized_training_step(state, sampler_params):
    # Generate all random keys at once
    key, subkey = jax.random.split(state.rng)
    
    # Pre-allocate random numbers
    n_chains = state.batch_size
    n_steps = sampler_params["chain_length"]
    random_keys = jax.random.split(subkey, n_chains)
    
    # Optimized sampling
    samples = optimized_sampler(
        random_keys, 
        state.prob_fn, 
        state.params,
        state.init_positions,
        sampler_params["step_size"],
        state.PBC,
        n_chains
    )
    
    # Continue with training...
```

## Research Workflows

### Parameter Sweep Study

```python
import itertools
import json
from pathlib import Path

def create_parameter_sweep():
    """Generate configuration files for systematic parameter study"""
    
    # Parameter ranges
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]
    batch_sizes = [500, 1000, 2000]
    architectures = [
        [1, 16, 8, 1],
        [1, 32, 16, 1], 
        [1, 64, 32, 16, 1]
    ]
    step_sizes = [0.2, 0.5, 1.0]
    
    # Create output directory
    output_dir = Path("sweep_configs")
    output_dir.mkdir(exist_ok=True)
    
    configs = []
    for lr, bs, arch, ss in itertools.product(
        learning_rates, batch_sizes, architectures, step_sizes):
        
        config = {
            "optimizer": {
                "optimizer_type": "adam",
                "learning_rate": lr
            },
            "training": {
                "batch_size": bs,
                "num_epochs": 3000
            },
            "model": {
                "architecture": arch,
                "activation": "tanh"
            },
            "sampler": {
                "step_size": ss,
                "chain_length": 100
            }
        }
        
        # Save configuration
        filename = f"config_lr{lr}_bs{bs}_arch{'x'.join(map(str,arch))}_ss{ss}.json"
        with open(output_dir / filename, "w") as f:
            json.dump(config, f, indent=2)
        
        configs.append((filename, config))
    
    return configs

# Run parameter sweep
configs = create_parameter_sweep()
results = []

for filename, config in configs:
    print(f"Running {filename}...")
    
    # Load and run
    parser = FileParser(f"sweep_configs/{filename}")
    parser.parse()
    
    # Run with timeout for safety
    try:
        params_fin, energy_history, _, best_params, best_energy = run_experiment(parser)
        
        result = {
            "config": filename,
            "best_energy": float(best_energy),
            "final_energy": float(energy_history[-1]),
            "converged": abs(energy_history[-10:].mean() - energy_history[-1]) < 1e-4
        }
        results.append(result)
        
    except Exception as e:
        print(f"Failed {filename}: {e}")
        continue

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("sweep_results.csv", index=False)
```

### Convergence Analysis

```python
import matplotlib.pyplot as plt
import numpy as np

def analyze_convergence(energy_history, window_size=100):
    """Analyze convergence characteristics"""
    
    energies = np.array(energy_history)
    
    # Moving average
    moving_avg = np.convolve(energies, np.ones(window_size)/window_size, mode='valid')
    
    # Gradient of energy
    energy_grad = np.gradient(energies)
    
    # Convergence criteria
    recent_std = energies[-window_size:].std()
    recent_grad_mean = np.abs(energy_grad[-window_size:]).mean()
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Energy history
    axes[0].plot(energies, alpha=0.7, label='Raw energy')
    axes[0].plot(range(window_size-1, len(energies)), moving_avg, 
                 label=f'{window_size}-step moving avg')
    axes[0].set_ylabel('Energy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Energy gradient
    axes[1].plot(energy_grad, alpha=0.7)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('dE/dt')
    axes[1].grid(True)
    
    # Recent stability
    recent_energies = energies[-window_size:]
    axes[2].hist(recent_energies, bins=20, alpha=0.7)
    axes[2].set_xlabel('Energy')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Recent {window_size} steps: std={recent_std:.6f}')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'recent_std': recent_std,
        'recent_grad_mean': recent_grad_mean,
        'converged': recent_std < 1e-4 and recent_grad_mean < 1e-5
    }

# Usage
analysis = analyze_convergence(energy_history)
print(f"Convergence analysis: {analysis}")
```

## Troubleshooting Examples

### Debugging Poor Convergence

```python
def debug_experiment(config):
    """Debug function to identify convergence issues"""
    
    # Step 1: Check configuration
    print("Configuration check:")
    print(f"  Learning rate: {config['optimizer']['learning_rate']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Architecture: {config['model']['architecture']}")
    print(f"  Step size: {config['sampler']['step_size']}")
    
    # Step 2: Test with reduced parameters
    debug_config = config.copy()
    debug_config['training']['num_epochs'] = 100
    
    # Step 3: Monitor intermediate values
    class DebugCallback:
        def __init__(self):
            self.step_count = 0
            self.acceptance_rates = []
            self.energies = []
            
        def __call__(self, step, energy, acceptance_rate):
            if step % 10 == 0:
                print(f"Step {step}: E={energy:.6f}, Acc={acceptance_rate:.3f}")
                self.acceptance_rates.append(acceptance_rate)
                self.energies.append(energy)
    
    # Run with debug
    callback = DebugCallback()
    # ... run experiment with callback
    
    # Analyze results
    if len(callback.acceptance_rates) > 0:
        avg_acceptance = np.mean(callback.acceptance_rates)
        energy_variance = np.var(callback.energies[-10:])
        
        print(f"\nDiagnostics:")
        print(f"  Average acceptance rate: {avg_acceptance:.3f}")
        print(f"  Recent energy variance: {energy_variance:.6f}")
        
        # Recommendations
        if avg_acceptance < 0.3:
            print("  Recommendation: Increase step size (acceptance too low)")
        elif avg_acceptance > 0.8:
            print("  Recommendation: Decrease step size (acceptance too high)")
            
        if energy_variance > 1e-3:
            print("  Recommendation: Decrease learning rate or increase batch size")
    
    return callback
```

### Memory Usage Profiling

```python
import jax
import time
import psutil

def profile_memory_usage():
    """Profile memory usage during training"""
    
    process = psutil.Process()
    
    def memory_callback(step, state):
        if step % 100 == 0:
            # System memory
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # JAX memory (if available)
            try:
                jax_memory = jax.devices()[0].memory_stats()
                jax_used = jax_memory.get('bytes_in_use', 0) / 1024 / 1024
            except:
                jax_used = 0
            
            print(f"Step {step}: System memory: {memory_mb:.1f}MB, "
                  f"JAX memory: {jax_used:.1f}MB")
    
    return memory_callback
```