# Contributing to QVarNet

This guide provides comprehensive information for contributors who want to help improve QVarNet. Whether you're fixing bugs, adding features, or improving documentation, we welcome your contributions!

## Getting Started

### Development Environment Setup

1. **Fork and Clone the Repository**:
```bash
git clone https://github.com/your-username/qvarnet.git
cd qvarnet
git remote add upstream https://github.com/pfargas/qvarnet.git
```

2. **Create Development Environment**:
```bash
# Create conda environment with development dependencies
conda env create -f environment_config.yaml
conda activate jax

# Install in development mode with extra dependencies
pip install -e ".[docs,tests]"

# Install pre-commit hooks for code quality
pre-commit install
```

3. **Verify Development Setup**:
```bash
# Run tests
pytest tests/

# Check code formatting
black --check src/
flake8 src/

# Build documentation
cd docs && make html
```

### Project Structure

```
qvarnet/
├── src/qvarnet/           # Main source code
│   ├── cli/              # Command-line interface
│   ├── layers/           # Custom neural network layers
│   ├── models/           # Neural network architectures
│   ├── __init__.py       # Package initialization
│   ├── callback.py       # Training callbacks
│   ├── main.py           # Main experiment runner
│   ├── models.py         # Basic model definitions
│   ├── sampler.py        # MCMC sampling engine
│   └── train.py          # Training algorithms
├── docs/                 # Documentation
├── scripts/              # Utility and benchmarking scripts
├── tests/                # Test suite
├── environment_config.yaml  # Conda environment file
├── pyproject.toml        # Project configuration
└── README.md            # Project overview
```

## Code Standards and Guidelines

### Python Code Style

We follow standard Python conventions with some project-specific requirements:

#### Formatting
- Use **Black** for code formatting (line length 88)
- Use **isort** for import sorting
- Follow PEP 8 for naming conventions

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

#### Documentation Standards

**Docstring Format**: Use Google-style docstrings

```python
def local_energy_batch(params, xs, model_apply):
    """Compute local energy for a batch of configurations.
    
    Args:
        params: Model parameters dictionary.
        xs: Batch of position configurations with shape (batch_size, dimensions).
        model_apply: Model's apply function for forward passes.
        
    Returns:
        Local energy values for each configuration with shape (batch_size, 1).
        
    Raises:
        ValueError: If input dimensions are incompatible.
        
    Example:
        >>> energies = local_energy_batch(params, positions, model.apply)
        >>> print(f"Mean energy: {energies.mean():.6f}")
    """
```

**Type Hints**: Use type hints for all public functions

```python
from typing import Dict, List, Tuple, Optional, Any
import jax.numpy as jnp

def train(
    n_steps: int,
    init_params: Dict[str, Any],
    shape: Tuple[int, ...],
    model_apply: callable,
    optimizer: optax.GradientTransformation,
    sampler_params: Dict[str, Any],
    PBC: float = 10.0,
    n_steps_sampler: int = 500,
    rng_seed: int = 0,
) -> Tuple[Dict[str, Any], jnp.ndarray, List, Dict[str, Any], float]:
```

#### JAX-Specific Guidelines

1. **Function Purity**: All JAX-compiled functions must be pure
2. **Random Number Management**: Use explicit PRNG keys
3. **Static Arguments**: Properly use `static_argnames` in `@partial(jax.jit, ...)`
4. **Device Management**: Consider device placement in performance-critical code

```python
# Good: Pure function with explicit random keys
@partial(jax.jit, static_argnames=("prob_fn",))
def mh_chain(random_values, PBC, prob_fn, prob_params, init_position, step_size):
    # Pure implementation - no side effects
    pass

# Bad: Using global random state or impure functions
def bad_sampler():
    global random_state  # Don't use globals
    random.random()      # Don't use Python's random
```

### Testing Guidelines

#### Test Structure

```
tests/
├── test_models.py      # Model architecture tests
├── test_sampler.py     # Sampling algorithm tests  
├── test_train.py       # Training function tests
├── test_cli.py         # Command-line interface tests
├── conftest.py         # Shared test fixtures
└── integration/        # Integration tests
```

#### Writing Tests

**Unit Tests**: Test individual functions and methods

```python
import pytest
import jax.numpy as jnp
from qvarnet.sampler import mh_kernel

class TestMHKernel:
    def test_acceptance_probability(self):
        """Test that acceptance probability is correctly computed."""
        # Setup
        prob_current = 1.0
        prob_proposed = 2.0
        expected_accept_prob = min(1.0, prob_proposed / prob_current)
        
        # Test implementation would go here
        assert accept_prob == expected_acceptance
        
    def test_periodic_boundary_conditions(self):
        """Test periodic boundary condition application."""
        # Test boundary wrapping behavior
        pass
    
    @pytest.mark.parametrize("step_size", [0.1, 0.5, 1.0, 2.0])
    def test_step_size_impact(self, step_size):
        """Test behavior with different step sizes."""
        # Parameterized test for different configurations
        pass
```

**Integration Tests**: Test complete workflows

```python
def test_complete_training_workflow():
    """Test end-to-end training process."""
    # Setup model and data
    config = get_test_config()
    
    # Run training
    results = run_complete_training(config)
    
    # Verify results
    assert results["energy"] < 1.0  # Should converge below threshold
    assert results["converged"] is True
```

**Performance Tests**: Verify performance requirements

```python
import time

def test_sampler_performance():
    """Ensure sampler meets performance benchmarks."""
    n_samples = 1000000
    
    start_time = time.perf_counter()
    samples = generate_test_samples(n_samples)
    end_time = time.perf_counter()
    
    throughput = n_samples / (end_time - start_time)
    assert throughput > 10e6  # Should handle >10M samples/second
```

#### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/qvarnet --cov-report=html

# Run specific test file
pytest tests/test_sampler.py

# Run with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m gpu         # Run only GPU tests
```

## Contribution Workflow

### 1. Create an Issue

Before starting work:
- Check if the feature/fix is already being worked on
- Create an issue describing your proposed change
- Discuss the approach with maintainers

### 2. Set Up Your Branch

```bash
# Sync with upstream
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or git checkout -b fix/issue-number-description
```

### 3. Develop Your Contribution

#### Code Development

1. **Write Tests First**: Write tests for your new functionality
2. **Implement Code**: Write the implementation following the style guidelines
3. **Run Tests**: Ensure all tests pass
4. **Update Documentation**: Update relevant documentation

#### Example Feature Addition

Let's say you're adding a new optimizer type:

```python
# 1. Add tests first
def test_adamw_optimizer():
    """Test AdamW optimizer implementation."""
    config = {
        "optimizer": {
            "optimizer_type": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4
        },
        # ... rest of config
    }
    results = run_experiment_with_config(config)
    assert results["converged"]

# 2. Implement the feature
def create_optimizer(optimizer_args):
    """Create optimizer instance from configuration."""
    if optimizer_args["optimizer_type"] == "adam":
        return optax.adam(learning_rate=optimizer_args["learning_rate"])
    elif optimizer_args["optimizer_type"] == "adamw":
        return optax.adamw(
            learning_rate=optimizer_args["learning_rate"],
            weight_decay=optimizer_args.get("weight_decay", 1e-4)
        )
    # ... existing cases

# 3. Update documentation
# Add to API.md and configuration documentation
```

### 4. Submit Your Contribution

#### Pre-commit Checks

```bash
# Run all quality checks
pre-commit run --all-files

# Manual checks
black src/ tests/
isort src/ tests/
flake8 src/ tests/
pytest
mypy src/  # If using type checking
```

#### Commit Guidelines

Use conventional commit messages:

```
feat: Add AdamW optimizer support
fix: Resolve memory leak in large-scale sampling
docs: Update installation guide for CUDA 12
refactor: Optimize sampler vectorization
test: Add integration tests for CLI
perf: Improve sampling throughput by 15%
```

#### Pull Request Process

1. **Push Your Branch**:
```bash
git push origin feature/your-feature-name
```

2. **Create Pull Request**:
- Use the GitHub interface
- Link to relevant issues
- Fill out the PR template
- Request reviews from maintainers

3. **PR Template**:
```markdown
## Description
Brief description of changes and motivation.

## Changes
- [ ] New feature implementation
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Performance benchmarks run

## Testing
- [ ] All tests pass
- [ ] Manual testing completed
- [ ] Performance verified

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Tests are passing
- [ ] No breaking changes (or documented)
```

## Specific Contribution Areas

### Algorithm Development

When contributing new sampling algorithms or optimization methods:

1. **Theoretical Foundation**: Include mathematical background in documentation
2. **Benchmarking**: Compare against existing methods
3. **Parameter Studies**: Document hyperparameter sensitivity
4. **Reproducibility**: Include seed and configuration information

```python
def new_sampling_algorithm(params, config):
    """
    Implement novel sampling algorithm.
    
    Theoretical basis:
    Based on [Reference], this algorithm improves sampling efficiency by...
    
    Performance characteristics:
    - Acceptance rate: ~60%
    - Autocorrelation time: 0.5x standard MH
    - Computational overhead: +10%
    
    Parameters:
    - param1: Description and typical values
    - param2: Description and typical values
    """
    # Implementation
```

### Model Architecture Contributions

When adding new neural network architectures:

1. **Compatibility**: Ensure JAX/Flax compatibility
2. **Performance**: Benchmark against existing models
3. **Documentation**: Include architecture diagrams and motivation
4. **Examples**: Provide usage examples

```python
class NovelWavefunction(nn.Module):
    """Novel wavefunction architecture with [key feature].
    
    Architecture diagram:
    Input → [Layer 1] → [Layer 2] → [Special Layer] → Output
    
    Performance:
    - Parameters: ~1000 for typical configuration
    - Training speed: Comparable to standard MLP
    - Expressiveness: Improved for [specific problems]
    """
    # Implementation
```

### Documentation Improvements

When improving documentation:

1. **Accuracy**: Verify code examples work
2. **Completeness**: Include all relevant parameters
3. **Clarity**: Use clear, accessible language
4. **Examples**: Provide practical examples

```markdown
## New Feature Documentation

### Overview
Brief description of what the feature does.

### Usage
```python
# Working example
import qvarnet
result = qvarnet.new_feature(config)
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| param1 | float | 1.0 | Parameter description |

### Performance
| Configuration | Performance |
|---------------|-------------|
| Small         | Fast        |
| Large         | Moderate    |
```

## Code Review Process

### Reviewer Guidelines

When reviewing contributions:

1. **Functionality**: Does the code work correctly?
2. **Performance**: Are there performance implications?
3. **Style**: Does it follow project guidelines?
4. **Documentation**: Is it well-documented?
5. **Testing**: Are tests comprehensive?

### Responding to Reviews

As a contributor:

1. **Address Feedback**: Make requested changes promptly
2. **Ask Questions**: Clarify any unclear feedback
3. **Provide Context**: Explain design decisions
4. **Update PR**: Keep PR description and status current

## Release Process

### Version Management

We use semantic versioning:
- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes (backward compatible)

### Changelog Maintenance

Maintain `CHANGELOG.md` with:
```markdown
## [1.2.0] - 2024-01-15

### Added
- New optimizer: AdamW
- Enhanced CLI with batch mode
- Performance profiling tools

### Fixed
- Memory leak in large-scale sampling
- GPU compatibility issue with CUDA 12

### Changed
- Improved default hyperparameters
- Updated documentation structure
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For general questions and ideas
- **Documentation**: Check existing docs first
- **Examples**: Look at example notebooks and scripts

### Recognition

Contributors are recognized in:
- `AUTHORS.md` file
- Release notes
- Documentation acknowledgments
- Conference presentations (when applicable)

Thank you for contributing to QVarNet! Your efforts help advance quantum computational research and make powerful VMC tools accessible to the scientific community.