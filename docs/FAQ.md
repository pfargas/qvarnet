# FAQ and Troubleshooting

This section addresses common questions, issues, and solutions when using QVarNet for quantum variational Monte Carlo simulations.

## Installation and Setup Issues

### Q: I'm getting "CUDA not found" errors

**A:** This typically indicates JAX cannot find your CUDA installation.

**Solutions:**
1. **Verify CUDA Installation**:
```bash
nvcc --version  # Should show CUDA version
nvidia-smi      # Should show GPU info
```

2. **Reinstall JAX with CUDA**:
```bash
pip uninstall jax jaxlib
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

3. **Check Environment Variables**:
```bash
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
```

4. **Verify JAX Detection**:
```python
import jax
print(jax.devices())  # Should show CUDA devices
```

### Q: Memory allocation errors on GPU

**A:** GPU memory is insufficient for the current configuration.

**Solutions:**
1. **Reduce Batch Size**:
```json
{
  "training": {
    "batch_size": 500  // Reduce from default 1000
  }
}
```

2. **Adjust Memory Fraction**:
```python
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # Use 80% of GPU memory
```

3. **Use Smaller Network Architecture**:
```json
{
  "model": {
    "architecture": [1, 16, 8, 1]  // Smaller network
  }
}
```

### Q: Installation fails with dependency conflicts

**A:** Python environment has conflicting packages.

**Solutions:**
1. **Use Fresh Conda Environment**:
```bash
conda create -n qvarnet python=3.11
conda activate qvarnet
pip install -e .
```

2. **Check Package Versions**:
```bash
pip list | grep -E "(jax|flax|optax)"
```

3. **Update Dependencies**:
```bash
pip install --upgrade jax jaxlib flax optax
```

## Training and Convergence Issues

### Q: Energy is not converging or increasing

**A:** Training configuration or model issues.

**Diagnostic Steps:**
1. **Check Learning Rate**:
```json
{
  "optimizer": {
    "learning_rate": 1e-4  // Try smaller value
  }
}
```

2. **Verify Sampling Quality**:
```python
# Check acceptance rate
acceptance_rate = calculate_acceptance_rate(samples)
print(f"Acceptance rate: {acceptance_rate:.3f}")
# Should be between 0.4-0.8 for optimal mixing
```

3. **Examine Energy History**:
```python
import matplotlib.pyplot as plt
plt.plot(energy_history)
plt.title("Energy Convergence")
plt.xlabel("Training Step")
plt.ylabel("Energy")
plt.show()
```

**Common Solutions:**
- Lower learning rate (1e-4 to 1e-3 range)
- Increase chain length for better sampling
- Check network architecture is appropriate
- Verify periodic boundary conditions

### Q: Sampling acceptance rate is too low (< 30%)

**A:** Step size is too large for the energy landscape.

**Solutions:**
1. **Reduce Step Size**:
```json
{
  "sampler": {
    "step_size": 0.2  // Reduce from default 0.5
  }
}
```

2. **Adaptive Step Size**:
```python
def adaptive_step_size(acceptance_rate, current_step_size):
    if acceptance_rate < 0.4:
        return current_step_size * 0.9
    elif acceptance_rate > 0.8:
        return current_step_size * 1.1
    return current_step_size
```

### Q: Autocorrelation time is very high

**A:** Samples are highly correlated, reducing effective sampling efficiency.

**Solutions:**
1. **Increase Step Size**:
```json
{
  "sampler": {
    "step_size": 1.0  // Increase step size
  }
}
```

2. **Use Longer Chains**:
```json
{
  "sampler": {
    "chain_length": 200  // Increase from default 100
  }
}
```

3. **Thinning**:
```python
# Keep every k-th sample to reduce correlation
thinned_samples = samples[::5]  # Keep every 5th sample
```

## Performance Issues

### Q: Training is very slow

**A:** Various performance bottlenecks.

**Diagnostic Steps:**
1. **Check Device Usage**:
```python
import jax
print(f"Using device: {jax.devices()[0]}")
```

2. **Profile the Training**:
```bash
qvarnet run --profile
# Then analyze with: tensorboard --logdir /tmp/profile-data
```

3. **Benchmark Sampling Speed**:
```python
import time
start = time.perf_counter()
samples = generate_samples(100000)
end = time.perf_counter()
print(f"Samples/second: {100000 / (end - start):.0f}")
```

**Optimization Solutions:**
1. **Increase Batch Size** (if memory allows):
```json
{
  "training": {
    "batch_size": 2000  // Increase from default 1000
  }
}
```

2. **Use GPU Acceleration**:
```bash
qvarnet run --device cuda
```

3. **Reduce Model Complexity**:
```json
{
  "model": {
    "architecture": [1, 32, 16, 1]  // Simpler architecture
  }
}
```

### Q: Out of memory errors during training

**A:** GPU memory is exhausted.

**Solutions:**
1. **Memory-Efficient Training**:
```python
# Process in smaller batches
def memory_efficient_training(state, batch_size):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        state = train_step(state, batch)
        # Clear intermediate results
        jax.clear_caches()
    return state
```

2. **Gradient Accumulation**:
```python
# Simulate larger batch size with gradient accumulation
def gradient_accumulation_step(state, data_chunks):
    accumulated_grads = None
    
    for chunk in data_chunks:
        grads = compute_gradients(state.params, chunk)
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = jax.tree_map(
                lambda x, y: x + y, accumulated_grads, grads
            )
    
    # Apply accumulated gradients
    return state.apply_gradients(grads=accumulated_grads)
```

## Model and Architecture Issues

### Q: Model fails to learn simple problems

**A:** Model architecture or initialization issues.

**Diagnostic Steps:**
1. **Test with Known Solution**:
```python
# Test with analytic wavefunction
from qvarnet.models import ExponentialWavefunction
model = ExponentialWavefunction()
```

2. **Check Initial Predictions**:
```python
# Examine initial model output
params = model.init(rng_key, sample_input)
output = model.apply(params, sample_input)
print(f"Initial output range: [{output.min():.3f}, {output.max():.3f}]")
```

**Solutions:**
1. **Simplify Architecture**:
```json
{
  "model": {
    "architecture": [1, 8, 1]  // Start very simple
  }
}
```

2. **Check Weight Initialization**:
```python
from flax.linen import initializers

# Use appropriate initialization
def init_fn(key, shape, dtype=jnp.float32):
    return initializers.normal(stddev=0.01)(key, shape, dtype)
```

3. **Normalize Inputs**:
```python
# Ensure inputs are appropriately scaled
normalized_inputs = inputs / jnp.std(inputs)
```

### Q: Neural network outputs NaN values

**A:** Numerical instability in the network.

**Common Causes and Solutions:**
1. **Exploding Gradients**:
```json
{
  "optimizer": {
    "learning_rate": 1e-5  // Much smaller learning rate
  }
}
```

2. **Numerical Overflow**:
```python
# Use log-space operations
log_psi = jnp.log(jnp.abs(psi) + 1e-8)  # Add small constant
```

3. **Activation Function Issues**:
```python
# Use more stable activations
hidden_activation = nn.tanh  # Instead of potentially unstable activations
```

## CLI and Configuration Issues

### Q: CLI command not found after installation

**A:** Installation script path issues.

**Solutions:**
1. **Check Installation**:
```bash
pip show qvarnet  # Verify package is installed
```

2. **Use Python Module**:
```bash
python -m qvarnet.cli.run run  # Alternative to direct command
```

3. **Check PATH**:
```bash
echo $PATH | grep -o "[^:]*bin[^:]*"  # Check if install directory is in PATH
```

### Q: Configuration file not found

**A:** File path resolution issues.

**Solutions:**
1. **Use Absolute Path**:
```bash
qvarnet run --filepath /full/path/to/config.json
```

2. **Check Current Directory**:
```bash
pwd  # Verify you're in the correct directory
ls  # Check if config file exists
```

3. **Copy Default Config**:
```bash
cp src/qvarnet/cli/parameters/hyperparams.json my_config.json
qvarnet run --filepath my_config.json
```

## Reproducibility Issues

### Q: Results vary between runs

**A:** Random seed management issues.

**Solutions:**
1. **Set Explicit Seeds**:
```json
{
  "training": {
    "rng_seed": 42  // Fixed seed for reproducibility
  }
}
```

2. **JAX Random Key Management**:
```python
# Proper key splitting
main_key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(main_key)
```

3. **Deterministic Operations**:
```python
import jax
jax.config.update("jax_enable_x64", True)  # Use double precision
```

## Specific Physics Issues

### Q: Computed energy is far from expected value

**A:** Physics implementation or parameter issues.

**Diagnostic Steps:**
1. **Check Potential Function**:
```python
def V(x):
    """Verify potential is correctly implemented"""
    return 0.5 * jnp.sum(x**2, axis=1)  # Harmonic oscillator
```

2. **Verify Analytical Comparison**:
```python
# Compare with known ground state
expected_energy = 0.5  # 1D harmonic oscillator ground state
relative_error = abs(computed_energy - expected_energy) / expected_energy
print(f"Relative error: {relative_error:.6f}")
```

3. **Check Units and Scaling**:
```python
# Ensure consistent units throughout calculation
# Check if periodic boundary conditions affect energy calculation
```

### Q: Wavefunction normalization issues

**A:** Wavefunction is not properly normalized.

**Solutions:**
1. **Numerical Normalization**:
```python
def normalize_wavefunction(psi, x_grid):
    """Numerically normalize wavefunction"""
    norm_squared = jnp.trapz(jnp.abs(psi)**2, x_grid)
    return psi / jnp.sqrt(norm_squared)
```

2. **Check Probability Density**:
```python
# Verify |psi|^2 integrates to 1
probability = jnp.abs(psi)**2
normalization = jnp.trapz(probability, x_grid)
print(f"Normalization: {normalization:.6f}")
```

## Debugging Tips

### General Debugging Approach

1. **Start Simple**: Begin with known working configurations
2. **Isolate Components**: Test individual components separately
3. **Use Logging**: Add print statements to trace execution
4. **Visualize Results**: Plot energy, samples, wavefunctions
5. **Check Dimensions**: Verify tensor shapes throughout pipeline

### Useful Debugging Code

```python
# Comprehensive debugging helper
def debug_training_step(state, batch, step):
    """Debug information for training step"""
    print(f"\nStep {step}:")
    print(f"  Batch shape: {batch.shape}")
    print(f"  Batch range: [{batch.min():.3f}, {batch.max():.3f}]")
    
    # Check model outputs
    psi_vals = state.apply_fn(state.params, batch)
    print(f"  Psi range: [{psi_vals.min():.3f}, {psi_vals.max():.3f}]")
    
    # Check for NaN/Inf
    if jnp.any(jnp.isnan(psi_vals)):
        print("  WARNING: NaN values in psi!")
    if jnp.any(jnp.isinf(psi_vals)):
        print("  WARNING: Inf values in psi!")
    
    # Energy statistics
    energy = local_energy_batch(state.params, batch, state.apply_fn)
    print(f"  Energy: {energy.mean():.6f} ± {energy.std():.6f}")
    
    return state
```

## When to Ask for Help

If you've tried the solutions above and still have issues:

1. **Search GitHub Issues**: Check if someone already reported similar problems
2. **Create Minimal Reproducible Example**: 
   - Simplify configuration to minimal case
   - Include exact error messages
   - Provide system information
3. **Include Context**:
   - Operating system and Python version
   - GPU/CPU information
   - JAX and CUDA versions
   - Full configuration file
4. **Use GitHub Discussions**: For general questions and advice

Community support resources:
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Documentation**: Check existing guides and examples

Remember that quantum VMC is a complex field, and some issues may require theoretical understanding beyond software troubleshooting. Don't hesitate to ask for help from the community!