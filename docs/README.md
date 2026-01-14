# QVarNet: Quantum Variational Monte Carlo with Artificial Neural Networks

## Overview

QVarNet is a high-performance Python library for quantum variational Monte Carlo (VMC) simulations using artificial neural network ansätze. Built on JAX/Flax for efficient GPU acceleration, QVarNet is specifically designed for research-level quantum mechanics calculations, focusing on accuracy, scalability, and reproducibility.

### Key Features

- **JAX/Flax Integration**: Full JAX compatibility with automatic differentiation and Just-In-Time compilation
- **High-Performance Sampling**: Optimized Metropolis-Hastings sampler with configurable acceptance rates
- **Modular Architecture**: Pluggable neural network models and activation functions
- **GPU Acceleration**: Native CUDA support for large-scale simulations
- **Research-Grade**: Designed for PhD-level quantum VMC research with emphasis on numerical stability
- **CLI Interface**: Command-line tool for quick experiments and parameter sweeps

## Research Context

QVarNet is developed for solving quantum many-body problems using the variational Monte Carlo method. The library focuses on:

1. **Wavefunction Approximation**: Using neural networks to approximate quantum states $\psi_\theta(\mathbf{x})$
2. **Energy Minimization**: Finding optimal parameters $\theta$ that minimize the expectation value $\langle E \rangle_\theta$
3. **Sampling Efficiency**: High-throughput generation of configurations according to $|\psi_\theta(\mathbf{x})|^2$

## Quick Start

```bash
# Install the environment
conda env create -f environment_config.yaml
conda activate jax

# Install the package
pip install -e .

# Run a basic experiment
qvarnet run
```

## Documentation Structure

This documentation is organized for different user types:

- **For Researchers**: In-depth theory, architecture details, and performance considerations
- **For Users**: Installation guides, API reference, and usage examples  
- **For Contributors**: Development setup and contribution guidelines

## Citation

If you use QVarNet in your research, please cite:

```bibtex
@software{qvarnet2025,
  title={QVarNet: Quantum Variational Monte Carlo with Artificial Neural Networks},
  author={Fargas, Pau},
  year={2025},
  url={https://github.com/pfargas/qvarnet}
}
```