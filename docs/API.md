# API Reference

This section provides detailed API documentation for all modules in QVarNet.

## Core Modules

### `qvarnet.main`

The main entry point for running quantum VMC experiments.

#### `run_experiment(args=None, profile=False)`

Run a complete quantum variational Monte Carlo experiment.

**Parameters:**
- `args`: Configuration object containing model, training, sampler, and optimizer arguments
- `profile`: Boolean flag to enable JAX profiling

**Returns:**
- `params_fin`: Final trained parameters
- `energy`: Energy history array
- `wf_hist`: Wavefunction parameter history
- `best_params`: Best parameters found during training
- `best_energy`: Lowest energy achieved

**Example:**
```python
from qvarnet.main import run_experiment
from qvarnet.cli.run import FileParser

# Load configuration
parser = FileParser("parameters.json")
parser.parse()

# Run experiment
params_fin, energy, _, best_params, best_energy = run_experiment(parser)
```

### `qvarnet.train`

Training utilities for VMC optimization.

#### Core Functions

##### `train(n_steps, init_params, shape, model_apply, optimizer, sampler_params, PBC=10, n_steps_sampler=500, rng_seed=0)`

Main training function that optimizes wavefunction parameters using gradient descent.

**Mathematical Background:**
The optimizer minimizes the loss function:
$$\mathcal{L}(\theta) = 2 \mathbb{E}_{x \sim |\psi_\theta(x)|^2}\left[(E_{\text{loc}}(x) - \langle E \rangle) \log |\psi_\theta(x)|\right]$$

**Parameters:**
- `n_steps`: Number of training iterations
- `init_params`: Initial model parameters
- `shape`: Input shape `(batch_size, dimensions)`
- `model_apply`: Model's apply function
- `optimizer`: Optax optimizer instance
- `sampler_params`: Dictionary with sampler configuration
- `PBC`: Periodic boundary condition size
- `n_steps_sampler`: MCMC chain length per training step
- `rng_seed`: Random seed for reproducibility

##### `local_energy_batch(params, xs, model_apply)`

Compute local energy for a batch of configurations.

**Physics:**
$$E_{\text{loc}}(x) = -\frac{1}{2}\frac{\nabla^2 \psi(x)}{\psi(x)} + V(x)$$

**Parameters:**
- `params`: Model parameters
- `xs`: Batch of position configurations
- `model_apply`: Model's apply function

**Returns:**
- Local energy values for each configuration

##### `laplace(func, x)`

Compute the Laplacian operator $\nabla^2 f(x)$ for arbitrary functions.

### `qvarnet.sampler`

High-performance Metropolis-Hastings sampling implementation.

#### Key Functions

##### `mh_chain(random_values, PBC, prob_fn, prob_params, init_position, step_size=1.0)`

Execute a single Metropolis-Hastings Markov chain.

**Algorithm:**
1. Generate proposal: $x' = x + \Delta x$ where $\Delta x \sim U(-s, s)$
2. Apply periodic boundary conditions
3. Accept with probability: $\min(1, \frac{|\psi(x')|^2}{|\psi(x)|^2})$

**Parameters:**
- `random_values`: Pre-generated random numbers shape `(n_steps, DoF + 1)`
- `PBC`: Periodic boundary condition size
- `prob_fn`: Probability density function (usually $|\psi|^2$)
- `prob_params`: Parameters for probability function
- `init_position`: Initial position for the chain
- `step_size`: MCMC step size

**Returns:**
- Array of sampled positions shape `(n_steps, DoF)`

##### `mh_kernel(uniform_random_numbers, prob_fn, prob_params, position, prob, step_size, PBC=10.0)`

Single Metropolis-Hastings step optimized for JAX compilation.

**Performance Characteristics:**
- JIT-compiled for maximum efficiency
- Vectorized operations for GPU acceleration
- Minimal memory allocations

### `qvarnet.models`

Neural network architectures for wavefunction approximation.

#### Available Models

##### `MLP(architecture, hidden_activation=nn.tanh, alpha=1.0)`

Multi-layer perceptron for general wavefunction approximation.

**Architecture Definition:**
- `architecture`: List specifying layer sizes, e.g., `[2, 10, 10, 1]`
- `hidden_activation`: Activation function for hidden layers
- `alpha`: Scaling parameter for output

**Example:**
```python
from qvarnet.models import MLP

# Create network for 2D harmonic oscillator
model = MLP(
    architecture=[2, 64, 64, 32, 1],
    hidden_activation=nn.tanh
)
```

##### `WavefunctionOneParameter()`

Simple analytical wavefunction with single parameter:
$$\psi(x) = \frac{1}{\alpha^2 + x^2}$$

**Use Case:**
- Validation and testing
- Analytical benchmarks
- Educational purposes

##### `ExponentialWavefunction()`

Gaussian-like wavefunction:
$$\psi(x) = \exp(-\alpha \sum_i x_i^2)$$

**Use Case:**
- Ground state approximation for harmonic oscillators
- Comparison with analytical solutions

### `qvarnet.models.mlp`

Enhanced MLP implementation with custom layers.

##### `CustomMLP`

Extended MLP with additional features:

- **Kernel Scaling**: Beta parameter for kernel normalization
- **Custom Initialization**: Configurable weight/bias initialization
- **Modular Design**: Compatible with custom layers

**Example:**
```python
from qvarnet.models.mlp import MLP as CustomMLP

model = CustomMLP(
    architecture=[2, 32, 16, 1],
    hidden_activation=nn.gelu,
    beta=0.5  # Scale down kernel weights
)
```

### `qvarnet.layers`

Custom neural network layers.

#### `CustomDense`

Enhanced dense layer with kernel scaling.

**Features:**
- **Kernel Scaling**: Beta parameter for weight normalization
- **Configurable Initialization**: Custom weight/bias initializers
- **JAX Compatibility**: Fully compatible with JAX compilation

**Parameters:**
- `features`: Output dimension
- `kernel_init`: Weight initialization function
- `bias_init`: Bias initialization function
- `beta`: Kernel scaling factor (default: 1.0)

### `qvarnet.cli`

Command-line interface for QVarNet.

#### `FileParser`

Configuration file parser for JSON parameter files.

**Usage:**
```python
from qvarnet.cli.run import FileParser

parser = FileParser("config.json")
parser.parse()

# Access configuration sections
model_args = parser.get_model_args
training_args = parser.get_training_args
sampler_args = parser.get_sampler_args
optimizer_args = parser.get_optimizer_args
```

#### CLI Commands

##### `qvarnet run [options]`

Run a VMC experiment from command line.

**Options:**
- `--device, -d`: Device to use (`cpu` or `cuda`)
- `--filepath, -f`: Path to parameter file
- `--profile, -p`: Enable JAX profiling

**Example:**
```bash
# Run with default parameters
qvarnet run

# Run with custom configuration on CPU
qvarnet run --device cpu --filepath my_config.json

# Run with profiling
qvarnet run --profile
```

### `qvarnet.callback`

Utility functions for training callbacks.

#### Functions

##### `nan_callback(x)`

Check for NaN values in arrays.

**Returns:** Boolean indicating presence of NaN values

##### `update_best_params(energy, best_energy, params, best_params)`

Conditional update of best parameters based on energy improvement.

**Performance:** JAX-compiled for efficiency during training

## Configuration System

### Parameter File Format

All QVarNet experiments use JSON configuration files:

```json
{
  "optimizer": {
    "optimizer_type": "adam",
    "learning_rate": 1e-3
  },
  "training": {
    "batch_size": 1000,
    "num_epochs": 3000
  },
  "model": {
    "architecture": [2, 10, 1],
    "activation": "tanh"
  },
  "sampler": {
    "step_size": 0.5,
    "chain_length": 100
  }
}
```

### Configuration Sections

#### Optimizer Configuration
- `optimizer_type`: `"adam"` or `"sgd"`
- `learning_rate`: Learning rate (typical range: 1e-4 to 1e-2)

#### Training Configuration
- `batch_size`: Number of MCMC chains
- `num_epochs`: Number of training steps

#### Model Configuration
- `architecture`: Neural network layer sizes
- `activation`: Activation function name

#### Sampler Configuration
- `step_size`: MCMC proposal step size
- `chain_length`: Length of each MCMC chain per training step

## Performance Considerations

### Memory Usage

- **Batch Size**: Larger batches improve GPU utilization but increase memory usage
- **Chain Length**: Longer chains provide better sampling but use more memory
- **Model Size**: Deeper networks require more memory and computation

### Sampling Efficiency

- **Acceptance Rate**: Target 50-70% acceptance for optimal mixing
- **Step Size**: Adjust based on acceptance rate and energy landscape
- **GPU Utilization**: Ensure sufficient parallelism with large batch sizes

### JAX Compilation

- **First Run**: Initial compilation takes time due to JIT
- **Static Arguments**: Use `static_argnames` for functions with changing Python arguments
- **Memory**: JAX caches compiled functions for reuse