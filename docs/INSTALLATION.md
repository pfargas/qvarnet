# Installation Guide

This guide covers the complete installation process for QVarNet, including environment setup, dependencies, and verification.

## System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum (16GB+ recommended for large simulations)
- **Storage**: 2GB free space for installation and results
- **GPU**: CUDA-compatible GPU (optional but recommended)

### Recommended Setup
- **GPU**: NVIDIA GPU with CUDA 12.x support
- **Memory**: 16GB+ RAM
- **Storage**: SSD with 10GB+ free space

## Environment Setup

### Method 1: Conda Environment (Recommended)

1. **Clone the repository**:
```bash
git clone https://github.com/pfargas/qvarnet.git
cd qvarnet
```

2. **Create and activate the conda environment**:
```bash
conda env create -f environment_config.yaml
conda activate jax
```

3. **Install the package in editable mode**:
```bash
pip install -e .
```

### Method 2: Manual Installation

1. **Install core dependencies**:
```bash
pip install jax[jax_cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax numpy matplotlib tqdm
```

2. **Clone and install QVarNet**:
```bash
git clone https://github.com/pfargas/qvarnet.git
cd qvarnet
pip install -e .
```

## GPU Configuration

### NVIDIA GPU Setup

1. **Install CUDA Toolkit** (version 11.8 or 12.x recommended)
2. **Install cuDNN** compatible with your CUDA version
3. **Verify GPU access**:
```bash
python -c "import jax; print(jax.devices())"
```

Expected output should show CUDA devices:
```
[CudaDevice(id=0, process_index=0)]
```

### CPU-Only Configuration

If you don't have a GPU or prefer CPU execution:

```bash
pip install jax[cpu]
```

## Verification

### Basic Installation Test

```python
import jax
import qvarnet
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"QVarNet imported successfully")
```

### CLI Test

```bash
qvarnet --help
```

### Sample Experiment

Run a minimal test experiment:

```bash
# Copy a simple parameter file
cp src/qvarnet/cli/parameters/hyperparams.json test_params.json

# Run experiment
qvarnet run --filepath test_params.json
```

## Environment Configuration

### JAX Configuration

The library automatically configures JAX, but you can customize:

```python
import jax
# Force CPU usage
jax.config.update("jax_platform_name", "cpu")

# Enable 64-bit precision (recommended for scientific computing)
jax.config.update("jax_enable_x64", True)

# Disable JAX compilation tracing (useful for debugging)
jax.config.update("jax_debug_infs", True)
```

### Memory Management

For large-scale simulations, consider:

```python
# Adjust JAX memory allocation
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # Use 80% of GPU memory
```

## Common Issues

### Installation Problems

**Issue**: CUDA not found
```bash
# Solution: Reinstall JAX with CUDA
pip uninstall jax jaxlib
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Issue**: Out of memory errors
```bash
# Solution: Reduce batch size or memory fraction
# In your parameter file:
{
  "training": {
    "batch_size": 500  # Reduce from default 1000
  }
}
```

### Performance Issues

**Issue**: Slow sampling
- Verify GPU is being used: `jax.devices()`
- Check batch size configuration
- Ensure JAX compilation is working

**Issue**: Poor convergence
- Adjust learning rate in optimizer configuration
- Increase chain length in sampler parameters
- Check model architecture suitability

## Development Installation

For contributors who want to modify the codebase:

```bash
# Clone with development dependencies
git clone https://github.com/pfargas/qvarnet.git
cd qvarnet

# Create development environment
conda env create -f environment_config.yaml
conda activate jax

# Install with development dependencies
pip install -e ".[docs,tests]"
```

## Docker Installation (Advanced)

For reproducible research environments:

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git
RUN pip install jax[cuda12_pip] flax optax numpy matplotlib tqdm

# Clone and install QVarNet
RUN git clone https://github.com/pfargas/qvarnet.git /opt/qvarnet
WORKDIR /opt/qvarnet
RUN pip install -e .

# Set entrypoint
ENTRYPOINT ["qvarnet"]
```

Build and run:
```bash
docker build -t qvarnet .
docker run --gpus all qvarnet run
```