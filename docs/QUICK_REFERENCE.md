# QVarNet Quick Reference Cheat Sheet

This cheat sheet provides a quick reference for common QVarNet operations and configurations.

## Quick Start Commands

### Installation
```bash
# Clone and install
git clone https://github.com/pfargas/qvarnet.git
cd qvarnet
conda env create -f environment_config.yaml
conda activate jax
pip install -e .

# Quick test
qvarnet --help
```

### Basic Usage
```bash
# Run with default parameters
qvarnet run

# Run with custom config
qvarnet run --filepath my_config.json

# Run on CPU
qvarnet run --device cpu

# Run with profiling
qvarnet run --profile
```

## Configuration Templates

### Minimal 1D Configuration
```json
{
  "optimizer": {
    "optimizer_type": "adam",
    "learning_rate": 0.001
  },
  "training": {
    "batch_size": 1000,
    "num_epochs": 2000
  },
  "model": {
    "architecture": [1, 32, 16, 1],
    "hidden_activation": "tanh"
  },
  "sampler": {
    "step_size": 0.5,
    "chain_length": 100
  }
}
```

### High-Performance 2D Configuration
```json
{
  "optimizer": {
    "optimizer_type": "adam",
    "learning_rate": 0.0005
  },
  "training": {
    "batch_size": 2000,
    "num_epochs": 5000
  },
  "model": {
    "architecture": [2, 128, 64, 32, 1],
    "hidden_activation": "gelu"
  },
  "sampler": {
    "step_size": 0.3,
    "chain_length": 200
  }
}
```

## Common Python Usage

### Basic Experiment
```python
import jax
from qvarnet.models import MLP
from qvarnet.main import run_experiment
from qvarnet.cli.run import FileParser
import optax

# Load configuration
parser = FileParser("config.json")
parser.parse()

# Run experiment
params, energy, _, best_params, best_energy = run_experiment(parser)
print(f"Best energy: {best_energy:.6f}")
```

### Custom Model
```python
from flax import linen as nn
import jax.numpy as jnp

class CustomModel(nn.Module):
    features: int = 64
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.features // 2)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x.squeeze()

# Usage in experiment
model = CustomModel(features=128)
```

### Manual Training Loop
```python
from qvarnet.train import train, create_optimizer

# Setup
model = MLP(architecture=[1, 32, 16, 1])
optimizer = optax.adam(learning_rate=1e-3)

# Initialize
rng = jax.random.PRNGKey(42)
input_shape = (1000, 1)  # batch_size, dimensions
params = model.init(rng, jnp.ones(input_shape))

# Train
params_final, energy_history, _, best_params, best_energy = train(
    n_steps=3000,
    init_params=params,
    shape=input_shape,
    model_apply=model.apply,
    optimizer=optimizer,
    sampler_params={"step_size": 0.5, "chain_length": 100}
)
```

## Parameter Guidelines

### Learning Rate Selection
| Problem Type | Learning Rate | Notes |
|--------------|---------------|-------|
| Simple (1D) | 1e-3 to 5e-3 | Fast convergence |
| Medium (2D) | 5e-4 to 2e-3 | Balanced speed/stability |
| Complex (3D+) | 1e-4 to 5e-4 | Prioritize stability |
| Fine-tuning | 1e-5 to 1e-4 | High precision |

### Architecture Selection
| Dimensions | Recommended Architecture | Parameters |
|------------|------------------------|------------|
| 1D | [1, 32, 16, 1] | ~500 |
| 2D | [2, 64, 32, 16, 1] | ~2,500 |
| 3D | [3, 128, 64, 32, 16, 1] | ~8,000 |

### Sampling Parameters
| Scenario | Step Size | Chain Length | Acceptance Rate |
|----------|------------|---------------|-----------------|
| Conservative | 0.2 | 100 | 70-80% |
| Balanced | 0.5 | 100-200 | 50-70% |
| Aggressive | 1.0 | 150-300 | 30-50% |

## Troubleshooting Quick Fixes

### Poor Convergence
```json
{
  "optimizer": {
    "learning_rate": 1e-4  // Reduce learning rate
  },
  "sampler": {
    "chain_length": 200    // Increase sampling
  }
}
```

### Memory Issues
```json
{
  "training": {
    "batch_size": 500      // Reduce batch size
  },
  "model": {
    "architecture": [1, 16, 8, 1]  // Smaller network
  }
}
```

### Low Acceptance Rate
```json
{
  "sampler": {
    "step_size": 0.2      // Reduce step size
  }
}
```

## Performance Optimization

### GPU Memory Management
```python
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  # Use 90% of GPU memory
```

### JAX Configuration
```python
import jax
jax.config.update("jax_enable_x64", True)  # Double precision
jax.config.update("jax_platform_name", "cuda")  # Force GPU
```

### Profiling
```python
# Enable profiling
with jax.profiler.start_trace("/tmp/profile"):
    results = run_experiment(config)
# Analyze with: tensorboard --logdir /tmp/profile
```

## Common Analysis Patterns

### Energy Convergence Plot
```python
import matplotlib.pyplot as plt

def plot_convergence(energy_history):
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history)
    plt.xlabel('Training Step')
    plt.ylabel('Energy')
    plt.title('Energy Convergence')
    plt.grid(True, alpha=0.3)
    plt.show()
```

### Wavefunction Visualization
```python
def plot_wavefunction(model, params, x_range, PBC=20.0):
    x = jnp.linspace(-PBC/2, PBC/2, 1000).reshape(-1, 1)
    psi = model.apply(params, x)
    prob = jnp.abs(psi)**2
    
    plt.figure(figsize=(10, 6))
    plt.plot(x.squeeze(), prob, label='VMC Wavefunction')
    # Add exact solution if available
    plt.xlabel('Position')
    plt.ylabel(r'$|\psi(x)|^2$')
    plt.legend()
    plt.show()
```

### Statistical Analysis
```python
import numpy as np
from scipy import stats

def analyze_energy_samples(samples):
    mean_energy = np.mean(samples)
    std_energy = np.std(samples, ddof=1)
    sem = std_energy / np.sqrt(len(samples))
    
    # 95% confidence interval
    ci_95 = stats.t.interval(0.95, len(samples)-1, 
                             loc=mean_energy, scale=sem)
    
    return {
        'mean': mean_energy,
        'std': std_energy,
        'sem': sem,
        'ci_95': ci_95
    }
```

## Batch Operations

### Parameter Sweep
```python
import itertools
import json

def create_sweep_configs():
    learning_rates = [1e-4, 5e-4, 1e-3]
    batch_sizes = [500, 1000, 2000]
    architectures = [
        [1, 16, 8, 1],
        [1, 32, 16, 1],
        [1, 64, 32, 16, 1]
    ]
    
    configs = []
    for lr, bs, arch in itertools.product(learning_rates, batch_sizes, architectures):
        config = {
            "optimizer": {"optimizer_type": "adam", "learning_rate": lr},
            "training": {"batch_size": bs, "num_epochs": 2000},
            "model": {"architecture": arch, "hidden_activation": "tanh"},
            "sampler": {"step_size": 0.5, "chain_length": 100}
        }
        configs.append(config)
    
    return configs
```

### Batch Experiment Execution
```python
def run_batch_experiments(configs):
    results = []
    
    for i, config in enumerate(configs):
        print(f"Running experiment {i+1}/{len(configs)}")
        
        # Save config to temp file
        with open(f"temp_config_{i}.json", "w") as f:
            json.dump(config, f)
        
        try:
            parser = FileParser(f"temp_config_{i}.json")
            parser.parse()
            
            params, energy, _, best_params, best_energy = run_experiment(parser)
            
            results.append({
                'config': config,
                'final_energy': float(best_energy),
                'converged': True
            })
            
        except Exception as e:
            results.append({
                'config': config,
                'error': str(e),
                'converged': False
            })
    
    return results
```

## Device Management

### Check Available Devices
```python
import jax
print(f"Available devices: {jax.devices()}")
print(f"Current device: {jax.devices()[0]}")
```

### Force CPU Usage
```python
import jax
jax.config.update("jax_platform_name", "cpu")
```

### Multi-GPU Setup (Advanced)
```python
# Distribute across multiple GPUs
devices = jax.devices()
n_devices = len(devices)

# Parallel training setup
def parallel_train(params, config):
    # Implement multi-device training
    pass
```

## Error Handling

### Common Exceptions
```python
try:
    params, energy, _, best_params, best_energy = run_experiment(parser)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("GPU memory exhausted - reduce batch size")
    elif "CUDA" in str(e):
        print("CUDA error - check GPU setup")
    else:
        print(f"Runtime error: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Debug Helper
```python
def debug_experiment(config):
    print("Configuration check:")
    print(f"  Learning rate: {config['optimizer']['learning_rate']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Architecture: {config['model']['architecture']}")
    print(f"  Step size: {config['sampler']['step_size']}")
    
    # Add more debug checks as needed
    return True
```

---

**Pro Tip**: Save this cheat sheet as `qvarnet_cheatsheet.md` in your working directory for quick reference during experiments!