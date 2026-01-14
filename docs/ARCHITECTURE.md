# Architecture and Design

This section provides an in-depth analysis of QVarNet's architecture, design decisions, and the theoretical foundations underlying the implementation.

## System Architecture Overview

QVarNet is built with a modular architecture that separates concerns while maintaining high performance through JAX/Flax integration.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Layer     │    │   Training      │    │   Sampling      │
│                 │    │   Engine        │    │   Engine        │
│ • Parameter     │◄──►│ • Optimization  │◄──►│ • Metropolis-   │
│   parsing       │    │ • Loss          │    │   Hastings      │
│ • Device mgmt    │    │ • Gradients     │    │ • GPU accel.    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Model Layer   │
                       │                 │
                       │ • Neural nets   │
                       │ • Custom layers │
                       │ • Wavefunction  │
                       └─────────────────┘
```

## Core Design Principles

### 1. JAX-First Philosophy

**Rationale**: JAX provides automatic differentiation, Just-In-Time compilation, and GPU acceleration essential for quantum VMC simulations.

**Implementation Decisions**:
- All numerical operations use JAX numpy for GPU compatibility
- Functions are designed for JAX JIT compilation with `static_argnames`
- Random number generation follows JAX's explicit key management
- Memory layout optimized for JAX's vectorization capabilities

```python
# Example of JAX-optimized design
@partial(jax.jit, static_argnames=("prob_fn",))
def mh_chain(random_values, PBC, prob_fn, prob_params, init_position, step_size):
    # Fully JIT-compiled for maximum performance
    # Static arguments avoid recompilation
```

### 2. Functional Programming Approach

**Rationale**: Functional style aligns with JAX's paradigm and enables better optimization.

**Key Features**:
- Pure functions without side effects
- Explicit parameter passing (no global state)
- Immutable data structures
- Composable function pipelines

### 3. Modular Sampling Architecture

**Design**: The sampling system is separated into:
- **Kernel Level**: Single MCMC step operations
- **Chain Level**: Complete Markov chains
- **Batch Level**: Vectorized chain execution

```python
# Hierarchical sampling design
mh_kernel()      # Single step
    ↓
mh_chain()       # Complete chain using kernel
    ↓
jax.vmap()       # Vectorized batch sampling
```

## Theoretical Foundation

### Variational Monte Carlo Framework

QVarNet implements the standard VMC approach with neural network wavefunctions:

#### Wavefunction Ansatz
$$\psi_\theta(\mathbf{x}) = f_{\text{NN}}(\mathbf{x}; \theta)$$

where $\theta$ are neural network parameters and $f_{\text{NN}}$ is the neural network function.

#### Local Energy
$$E_{\text{loc}}(\mathbf{x}) = \frac{\hat{H}\psi_\theta(\mathbf{x})}{\psi_\theta(\mathbf{x})} = -\frac{1}{2}\frac{\nabla^2\psi_\theta(\mathbf{x})}{\psi_\theta(\mathbf{x})} + V(\mathbf{x})$$

#### Energy Minimization
The variational principle states that:
$$\langle E \rangle_\theta = \int |\psi_\theta(\mathbf{x})|^2 E_{\text{loc}}(\mathbf{x}) d\mathbf{x} \geq E_0$$

#### Loss Function
Using the variance minimization approach:
$$\mathcal{L}(\theta) = 2\mathbb{E}_{x \sim |\psi_\theta(x)|^2}\left[(E_{\text{loc}}(x) - \langle E \rangle)\log|\psi_\theta(x)|\right]$$

### Metropolis-Hastings Sampling

**Algorithm Implementation**:

1. **Proposal**: $x' = x + \Delta x$, $\Delta x \sim U[-s, s]$
2. **Boundary Conditions**: Apply periodic boundary conditions
3. **Acceptance**: $A = \min\left(1, \frac{|\psi(x')|^2}{|\psi(x)|^2}\right)$
4. **Update**: $x_{n+1} = \begin{cases} x' & \text{with probability } A \\ x & \text{with probability } 1-A \end{cases}$

**Design Optimizations**:
- Pre-generated random numbers for reproducibility and performance
- Vectorized proposal generation
- JAX-compatible acceptance logic
- Configurable step size adaptation

## Implementation Details

### Neural Network Architecture

#### Multi-Layer Perceptron Design

```python
class MLP(nn.Module):
    architecture: list           # Layer dimensions
    hidden_activation: callable  # Non-linearity
    alpha: float                 # Output scaling
    
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.architecture) - 1):
            x = nn.Dense(features=self.architecture[i + 1])(x)
            if i < len(self.architecture) - 2:  # No activation on output layer
                x = self.hidden_activation(x)
        return x
```

**Design Considerations**:
- Flexible architecture specification
- Configurable activation functions
- Output normalization through alpha parameter
- Flax-compatible for JAX integration

#### Custom Layer Implementation

The `CustomDense` layer provides additional control over weight initialization and scaling:

```python
class CustomDense(nn.Module):
    features: int
    beta: float = 1.0           # Kernel scaling factor
    kernel_init: callable
    bias_init: callable
    
    @nn.compact
    def __call__(self, inputs):
        kernel = self.param("kernel", self.kernel_init, (inputs.shape[-1], self.features))
        y = jnp.dot(inputs, self.beta * kernel)  # Scaled kernel
        bias = self.param("bias", self.bias_init, (self.features,))
        return y + bias
```

### Training System Architecture

#### State Management

QVarNet uses Flax's `TrainState` for consistent parameter management:

```python
state = train_state.TrainState.create(
    apply_fn=model_apply,
    params=init_params,
    tx=optimizer
)
```

#### Training Loop Design

The training loop integrates sampling and optimization:

```python
for step in range(n_steps):
    # 1. Generate samples using current parameters
    batch = sampler(...)
    
    # 2. Compute energy and gradients
    state, energy = train_step(state, batch)
    
    # 3. Track best parameters
    if energy < best_energy:
        best_energy = energy
        best_params = state.params
```

### Sampling Engine Performance

#### Vectorization Strategy

The sampling engine uses three levels of vectorization:

1. **Chain Level**: Individual MCMC chains
2. **Batch Level**: Multiple parallel chains
3. **Step Level**: Vectorized operations within each step

#### Memory Efficiency

```python
# Pre-allocate random numbers for efficiency
rand_nums = random.uniform(key, (n_chains, n_steps_sampler, DoF + 1))

# Vectorized sampling
batch = jax.vmap(mh_chain, in_axes=(0, None, None, None, 0, None))(
    rand_nums, PBC, prob_fn, state.params, init_position, step_size
)
```

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Scaling |
|-----------|------------|---------|
| Forward Pass | O(N·L) | Linear in batch size N and network size L |
| Backward Pass | O(N·L) | Same as forward pass |
| MCMC Sampling | O(N·S·D) | Batch N × Steps S × Dimensions D |
| Local Energy | O(N·L·D) | Additional gradient computations |

### Memory Usage Analysis

**Memory Components**:
- Model parameters: O(L) where L is network size
- Batch storage: O(N·S·D) for sampling
- Gradient storage: O(L) during training
- JAX compilation cache: Variable, depends on function diversity

**Optimization Strategies**:
- Use `@partial(jax.jit, static_argnames=...)` to reduce recompilation
- Batch size tuning for GPU utilization
- Memory-efficient sampling with pre-allocated arrays

### GPU Utilization

The architecture is designed for maximum GPU throughput:

1. **Large Batch Operations**: Vectorized linear algebra
2. **Minimal CPU-GPU Transfer**: Keep data on GPU when possible
3. **JAX Compilation**: Optimized CUDA kernels
4. **Async Operations**: Overlap computation and transfer

## Configuration System

### JSON-Based Configuration

The configuration system uses JSON for:

- **Human Readability**: Easy to understand and modify
- **Language Independence**: Compatible with any tooling
- **Validation**: Structured with clear type expectations
- **Extensibility**: Easy to add new parameters

### Parameter Hierarchy

```
Root
├── optimizer      # Training optimization
├── training       # Training hyperparameters  
├── model         # Neural network architecture
└── sampler       # MCMC sampling parameters
```

**Design Benefits**:
- Clear separation of concerns
- Easy parameter sweeps
- Reproducible experiments
- Validation by construction

## Error Handling and Robustness

### Numerical Stability

**Strategies Implemented**:
- Small constants in denominators: `psi + 1e-12`
- Log-space operations: `log(|psi| + 1e-8)`
- Gradient clipping (configurable)
- NaN detection and handling

### Graceful Degradation

```python
# Example of robust implementation
psi_safe = psi_vals + 1e-12  # Prevent division by zero
kinetic = -0.5 * (d2psi / psi_safe)  # Safe division
```

## Extensibility Design

### Plugin Architecture

The system supports extensions through:

1. **Custom Models**: Inherit from `nn.Module`
2. **Custom Samplers**: Implement compatible interface
3. **Custom Optimizers**: Use Optax ecosystem
4. **Custom Potentials**: Modify `V(x)` function

### Future Compatibility

Design decisions ensure future extensions can:
- Add new quantum systems through potential functions
- Implement different sampling algorithms
- Support various neural network architectures
- Integrate with quantum chemistry packages

## Testing and Validation Strategy

### Unit Testing Philosophy

Each component is tested for:
- **Numerical Accuracy**: Compare with analytical solutions
- **Performance**: Benchmark against baselines
- **Stability**: Stress test with edge cases
- **Compatibility**: Verify JAX optimizations

### Integration Testing

End-to-end testing validates:
- Complete training workflows
- CLI functionality
- Configuration parsing
- GPU/CPU compatibility

## Security and Safety

### Safe Defaults

- Reasonable default hyperparameters
- Memory usage limits
- Training duration bounds
- Error condition handling

### Reproducibility

- Explicit random seed management
- Deterministic JAX operations
- Version-controlled configurations
- Complete experiment logging