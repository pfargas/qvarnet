# QVarNet Examples

This directory contains example scripts demonstrating how to use the refactored qvarnet package.

## `harmonic_oscillator_deepset.py`

A comprehensive example showing how to use the DeepSet architecture with Variational Monte Carlo for quantum systems **in log-domain**.

### What It Does

1. **Creates two systems**:
   - Simple harmonic oscillator (10 particles, 1D)
   - Harmonic oscillator with nearest-neighbor interactions

2. **Uses DeepSet Architecture (Log-Domain)**:
   - Model outputs: log|ψ(x)| (not ψ itself)
   - φ network (per-particle): 1D → 16 hidden → 16 shared
   - F network (aggregation): 16 → 16 hidden → 1 output (log amplitude)
   - Final output: log|ψ| from which probability |ψ|² = exp(2 * log|ψ|) is computed
   - **Advantage**: Numerically more stable than direct ψ representation

3. **Trains with VMC (Log-Domain)**:
   - 4 parallel MCMC chains
   - 500 training epochs
   - Adam optimizer (lr=1e-3)
   - Adaptive step size
   - Log-space kinetic energy formula for stability

4. **Generates Dashboard**:
   - Energy convergence plots
   - Comparison of final vs. minimum energies
   - Smoothed convergence curves
   - Training statistics table
   - Energy distribution histogram

### Why Log-Domain?

This example uses **log-domain training** (is_log_model=True) by default for several important reasons:

1. **Numerical Stability**: 
   - Direct ψ can have extreme values (very small or very large)
   - Log|ψ| is bounded and numerically stable
   - Prevents overflow/underflow in gradient computation

2. **Better Convergence**:
   - Log-space kinetic energy: T = -½(∇²log|ψ| + |∇log|ψ||²)
   - More stable than T = -½(∇²ψ / ψ)
   - Reduces numerical noise in energy evaluation

3. **Faster Training**:
   - Smoother optimization landscape in log-space
   - Better gradient flow
   - Fewer training instabilities

**For your application**: Always use `is_log_model=True` in production code unless you have a specific reason not to.

### Running the Example

```bash
# Make sure you're in the qvarnet root directory
cd /home/pau/PhD/qvarnet

# Run the example (will take ~5-10 minutes)
python examples/harmonic_oscillator_deepset.py
```

### Output

- **vmc_dashboard.png**: High-resolution (300 DPI) dashboard with all plots
- **Console output**: Training progress and summary statistics

### Expected Results

For a harmonic oscillator with ω=1, the exact ground state energy is:
- **E₀ = D/2** where D is degrees of freedom
- For 10 particles in 1D: **E₀ = 5.0**

With the DeepSet architecture and 500 epochs:
- Simple HO should converge to ~5.0
- HO with NN interactions should be slightly higher due to the interaction term

### Customizing the Example

Edit the script to change:

```python
# Change number of particles
n_particles = 10  # Line in create_deepset_model()

# Change network architecture
phi_hidden_units = 16      # Hidden units in φ network
f_hidden_units = 16        # Hidden units in F network
shared_dim = 16            # Aggregation dimension

# Change training parameters
n_epochs = 500             # Number of training epochs
learning_rate = 1e-3       # Optimizer learning rate
chain_length = 200         # MCMC chain length
thermalization_steps = 50  # Burn-in steps
```

### Understanding the Plots

1. **Energy Convergence** (top): Shows how energy decreases during training
   - Should be monotonic or noisy but generally decreasing
   - Plateauing suggests convergence

2. **Final vs. Minimum Energy** (middle-left): 
   - Bars compare final epoch energy vs. best achieved
   - Small gap suggests good convergence
   - Large gap might indicate oscillation

3. **Smoothed Convergence** (middle-right):
   - Running average (window=20 epochs) smooths noise
   - Clearer trend visible

4. **Statistics Table** (bottom-left):
   - ΔE = Standard deviation of last 50 epochs
   - Shows training stability

5. **Energy Distribution** (bottom-right):
   - Histogram of energy values from last 100 epochs
   - Should be narrow (concentrated) if converged
   - Broader distribution indicates ongoing learning

### Key Insights

**Comparing HO vs. HO-NN**:
- Nearest-neighbor interactions add energy (repulsion between particles)
- ω_interaction = 0.1 adds ~0.1-0.2 to ground state energy
- DeepSet should capture both permutation symmetry and interaction effects

**DeepSet Architecture Benefits**:
- **Permutation invariant**: Same energy for any particle ordering
- **Scalable**: Can handle variable number of particles
- **Efficient**: φ is per-particle, F aggregates → O(N) complexity

### Troubleshooting

**Training is very slow**:
- Reduce `n_epochs` to 100 for testing
- Reduce `chain_length` to 100

**Energy doesn't converge**:
- Increase learning rate (try 5e-3)
- Increase chain length (try 300)
- Increase thermalization steps (try 100)

**Memory errors**:
- Reduce batch size (in train() call)
- Reduce number of chains (in shape parameter)

### Next Steps

1. **Modify the architecture**: Try different φ/F hidden sizes
2. **Test on other systems**: Create custom Hamiltonian subclass
3. **Compare with exact solutions**: For harmonic oscillator, E₀ is known
4. **Implement callbacks**: Add custom metrics or visualizations
5. **Use QGT**: Enable natural gradient descent for faster convergence

### References

- **DeepSet paper**: Zaheer et al., "Deep Sets" (NIPS 2017)
- **VMC review**: Carleo & Troyer, "Solving the quantum many-body problem with ANNs" (Science 2017)
- **QVarNet docs**: See `docs/` folder for architecture details
