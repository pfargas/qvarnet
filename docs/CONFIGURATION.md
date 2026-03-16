# Configuration Reference

This section provides a comprehensive reference for all configuration options available in QVarNet.

## Configuration File Structure

QVarNet uses JSON configuration files organized into four main sections:

```json
{
  "optimizer": { ... },
  "training": { ... }, 
  "model": { ... },
  "sampler": { ... }
}
```

## Optimizer Configuration

### Available Optimizers

#### Adam Optimizer

```json
{
  "optimizer": {
    "optimizer_type": "adam",
    "learning_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-08,
    "weight_decay": 0.0
  }
}
```

**Parameters:**
- `optimizer_type` (string): `"adam"` - Use Adam optimizer
- `learning_rate` (float): Learning rate (default: 0.001)
  - Typical range: 1e-5 to 1e-2
  - Smaller values for stable convergence
  - Larger values for faster learning (may be unstable)
- `beta1` (float): Exponential decay rate for first moment (default: 0.9)
- `beta2` (float): Exponential decay rate for second moment (default: 0.999)
- `eps` (float): Small constant for numerical stability (default: 1e-08)
- `weight_decay` (float): L2 regularization coefficient (default: 0.0)

#### SGD Optimizer

```json
{
  "optimizer": {
    "optimizer_type": "sgd", 
    "learning_rate": 0.01,
    "momentum": 0.0,
    "nesterov": false
  }
}
```

**Parameters:**
- `optimizer_type` (string): `"sgd"` - Use Stochastic Gradient Descent
- `learning_rate` (float): Learning rate (default: 0.01)
- `momentum` (float): Momentum factor (default: 0.0)
- `nesterov` (bool): Use Nesterov momentum (default: false)

### Optimizer Selection Guide

| Optimizer | Use Case | Learning Rate Range | Stability |
|-----------|----------|-------------------|-----------|
| Adam | General purpose, robust | 1e-4 - 1e-2 | High |
| SGD | Large datasets, fine-tuning | 1e-3 - 1e-1 | Medium |

## Training Configuration

```json
{
  "training": {
    "batch_size": 1000,
    "num_epochs": 3000,
    "rng_seed": 42,
    "validation_interval": 100,
    "checkpoint_interval": 500
  }
}
```

**Parameters:**
- `batch_size` (int): Number of MCMC chains to run in parallel
  - Larger values improve GPU utilization
  - Limited by GPU memory
  - Typical values: 500-5000
- `num_epochs` (int): Number of training iterations
  - Convergence typically achieved in 1000-5000 steps
  - More epochs for complex problems
- `rng_seed` (int): Random seed for reproducibility (default: 42)
- `validation_interval` (int): How often to compute validation metrics
- `checkpoint_interval` (int): How often to save model checkpoints

### Batch Size Guidelines

| System | Recommended Batch Size | Memory Usage (approx.) |
|--------|------------------------|-----------------------|
| 1D Harmonic | 1000-2000 | 1-2 GB |
| 2D Harmonic | 500-1500 | 2-4 GB |
| 3D Harmonic | 300-1000 | 3-6 GB |
| Complex potentials | 500-2000 | 4-8 GB |

## Model Configuration

### Multi-Layer Perceptron (MLP)

```json
{
  "model": {
    "type": "mlp",
    "architecture": [2, 64, 32, 16, 1],
    "hidden_activation": "tanh",
    "alpha": 1.0,
    "kernel_init": "lecun_normal",
    "bias_init": "zeros"
  }
}
```

**Parameters:**
- `type` (string): Model type (`"mlp"`, `"exponential"`, `"onelinear"`)
- `architecture` (list): Layer sizes from input to output
  - First element: input dimension
  - Last element: output dimension (always 1 for wavefunction)
  - Middle elements: hidden layer sizes
- `hidden_activation` (string): Activation function for hidden layers
  - `"tanh"`: Hyperbolic tangent (default, good for wavefunctions)
  - `"relu"`: Rectified linear unit
  - `"gelu"`: Gaussian error linear unit
  - `"sigmoid"`: Logistic sigmoid
- `alpha` (float): Scaling factor for output (default: 1.0)
- `kernel_init` (string): Weight initialization method
  - `"lecun_normal"`: LeCun normal initialization (default)
  - `"glorot_normal"`: Xavier/Glorot normal
  - `"he_normal"`: He initialization
- `bias_init` (string): Bias initialization method
  - `"zeros"`: Initialize to zero (default)
  - `"ones"`: Initialize to one
  - `"normal"`: Normal distribution

### Architecture Examples

#### 1D Harmonic Oscillator
```json
{
  "model": {
    "architecture": [1, 32, 16, 8, 1],
    "hidden_activation": "tanh"
  }
}
```

#### 2D Harmonic Oscillator
```json
{
  "model": {
    "architecture": [2, 64, 32, 16, 1],
    "hidden_activation": "tanh"
  }
}
```

#### 3D Harmonic Oscillator
```json
{
  "model": {
    "architecture": [3, 128, 64, 32, 1],
    "hidden_activation": "tanh"
  }
}
```

#### High-Expressivity Network
```json
{
  "model": {
    "architecture": [2, 256, 128, 64, 32, 16, 1],
    "hidden_activation": "gelu"
  }
}
```

### Specialized Models

#### Exponential Wavefunction
```json
{
  "model": {
    "type": "exponential",
    "alpha": 1.0
  }
}
```

**Mathematical form:** $\psi(x) = \exp(-\alpha x^2)$

#### One-Parameter Wavefunction
```json
{
  "model": {
    "type": "onelinear",
    "alpha": 0.5
  }
}
```

**Mathematical form:** $\psi(x) = \frac{1}{\alpha^2 + x^2}$

## Sampler Configuration

```json
{
  "sampler": {
    "step_size": 0.5,
    "chain_length": 100,
    "pbc": 40.0,
    "adapt_step_size": true,
    "target_acceptance": 0.65,
    "burn_in": 0,
    "thinning": 1
  }
}
```

**Parameters:**
- `step_size` (float): MCMC proposal step size
  - Larger values: lower acceptance, faster exploration
  - Smaller values: higher acceptance, slower exploration
  - Typical range: 0.1 - 2.0
  - Default: 0.5
- `chain_length` (int): Length of each MCMC chain per training step
  - Longer chains: better sampling, more computation
  - Typical values: 50-500
  - Should be >10x autocorrelation time
- `pbc` (float): Periodic boundary condition size
  - Should be large enough to contain wavefunction
  - For harmonic oscillator: 20-40 a.u.
  - Default: 40.0
- `adapt_step_size` (bool): Enable adaptive step size (default: true)
- `target_acceptance` (float): Target acceptance rate (default: 0.65)
  - Optimal range: 0.5-0.7
  - Used for adaptive step size
- `burn_in` (int): Number of burn-in steps to discard (default: 0)
- `thinning` (int): Keep every k-th sample (default: 1)

### Step Size Tuning Guidelines

| Target Acceptance | Recommended Step Size | Effect |
|-------------------|----------------------|--------|
| 0.3-0.4 | 0.8-1.5 | Aggressive exploration, lower correlation |
| 0.4-0.6 | 0.4-0.8 | Balanced approach (recommended) |
| 0.6-0.8 | 0.2-0.4 | Conservative exploration, higher correlation |
| >0.8 | <0.2 | Too conservative, very slow exploration |

## Complete Configuration Examples

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
    "learning_rate": 0.0005,
    "weight_decay": 1e-6
  },
  "training": {
    "batch_size": 2000,
    "num_epochs": 5000,
    "rng_seed": 42
  },
  "model": {
    "architecture": [2, 128, 64, 32, 16, 1],
    "hidden_activation": "gelu",
    "kernel_init": "he_normal"
  },
  "sampler": {
    "step_size": 0.3,
    "chain_length": 200,
    "pbc": 30.0,
    "adapt_step_size": true,
    "target_acceptance": 0.65
  }
}
```

### Research-Grade 3D Configuration

```json
{
  "optimizer": {
    "optimizer_type": "adam",
    "learning_rate": 0.0002,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 1e-5
  },
  "training": {
    "batch_size": 1500,
    "num_epochs": 8000,
    "rng_seed": 12345,
    "validation_interval": 50,
    "checkpoint_interval": 1000
  },
  "model": {
    "architecture": [3, 256, 128, 64, 32, 16, 8, 1],
    "hidden_activation": "tanh",
    "alpha": 1.0
  },
  "sampler": {
    "step_size": 0.25,
    "chain_length": 150,
    "pbc": 25.0,
    "adapt_step_size": true,
    "target_acceptance": 0.6,
    "burn_in": 20,
    "thinning": 2
  }
}
```

## Parameter Sweep Templates

### Learning Rate Sweep

```json
{
  "optimizer": {
    "optimizer_type": "adam",
    "learning_rate": "{{learning_rate}}"
  },
  "training": {
    "batch_size": 1000,
    "num_epochs": 3000
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

Use with learning rates: [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]

### Architecture Sweep

```json
{
  "optimizer": {
    "optimizer_type": "adam",
    "learning_rate": 0.001
  },
  "training": {
    "batch_size": 1000,
    "num_epochs": 3000
  },
  "model": {
    "architecture": "{{architecture}}",
    "hidden_activation": "tanh"
  },
  "sampler": {
    "step_size": 0.5,
    "chain_length": 100
  }
}
```

Use with architectures:
- `[1, 16, 8, 1]` (small)
- `[1, 32, 16, 1]` (medium)
- `[1, 64, 32, 16, 1]` (large)
- `[1, 128, 64, 32, 16, 1]` (very large)

## Configuration Validation

### Required Parameters

| Section | Required Parameters | Default Values |
|---------|--------------------|----------------|
| optimizer | `optimizer_type`, `learning_rate` | adam, 0.001 |
| training | `batch_size`, `num_epochs` | 1000, 3000 |
| model | `architecture` | [1, 32, 16, 1] |
| sampler | `step_size`, `chain_length` | 0.5, 100 |

### Validation Rules

1. **Architecture validation**:
   - Input dimension must match physical system
   - Output dimension must be 1 (wavefunction amplitude)
   - Hidden layers should have decreasing sizes

2. **Learning rate validation**:
   - Must be positive
   - Recommended range: 1e-6 to 1e-1

3. **Batch size validation**:
   - Must be positive integer
   - Should be compatible with GPU memory

4. **Sampler validation**:
   - Step size must be positive
   - Chain length must be reasonable (>10)
   - PBC must be large enough

## Troubleshooting Configurations

### Common Issues

#### Poor Convergence
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

#### Memory Issues
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

#### Low Acceptance Rate
```json
{
  "sampler": {
    "step_size": 0.2      // Reduce step size
  }
}
```

#### High Autocorrelation
```json
{
  "sampler": {
    "step_size": 1.0,      // Increase step size
    "chain_length": 300    // Longer chains
  }
}
```

This configuration reference provides all the information needed to set up and optimize QVarNet experiments for various research scenarios.