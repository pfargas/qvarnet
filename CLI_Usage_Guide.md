# Enhanced CLI Configuration System

This document describes the refactored CLI system for qvarnet that supports multiple configuration management, presets, and better organization.

## **🎯 Key Changes**

### **1. Configuration System**
- **New location**: `src/qvarnet/cli/parameters/`
- **Base classes**: Abstract configuration with validation
- **Preset system**: Pre-defined configurations for common experiments
- **Backward compatibility**: Maintained for existing workflows

### **2. Enhanced CLI**
- **Multiple commands**: `run`, `list-presets`
- **Flexible configuration**: `--config`, `--preset`, `--override` options
- **Configuration validation**: JSON schema validation and error handling

## **📁 New Directory Structure**

```
src/qvarnet/cli/parameters/
├── __init__.py                 # Configuration factory functions
├── base.py                     # Base configuration classes
├── presets/                     # Pre-defined configurations
│   ├── harmonic_oscillator_standard.json
│   ├── ising_1d_benchmark.json
│   └── qgt_comparison_test.json
└── hyperparams.json              # Legacy compatibility file
```

## **🚀 Usage Examples**

### **List Available Presets**
```bash
qvarnet list-presets
```
Output:
```
Available preset configurations:
  harmonic_oscillator_standard: Standard harmonic oscillator benchmark with default MLP
    Tags: continuous, benchmark, harmonic_oscillator
  ising_1d_benchmark: 1D Ising model for testing discrete spin systems  
    Tags: discrete, ising, spin_system, benchmark
  qgt_comparison_test: Compare QGT preconditioning against standard optimizers
    Tags: qgt, natural_gradient, optimization, comparison
```

### **Run with Preset**
```bash
# Use standard harmonic oscillator preset
qvarnet run --preset harmonic_oscillator_standard

# Use Ising model benchmark
qvarnet run --preset ising_1d_benchmark

# Use QGT comparison test
qvarnet run --preset qgt_comparison_test
```

### **Run with Custom Configuration**
```bash
# Using a custom JSON config file
qvarnet run --config /path/to/my/config.json

# Using preset with overrides
qvarnet run --preset harmonic_oscillator_standard --override training.num_epochs=5000 --override optimizer.learning_rate=5e-4

# Multiple overrides
qvarnet run --preset qgt_comparison_test \
  --override training.batch_size=1200 \
  --override optimizer.qgt_config.regularization=1e-5
```

### **Configuration Dump and Validation**
```bash
# Show loaded configuration
qvarnet run --preset harmonic_oscillator_standard --config-dump

# Output:
{
  "experiment": {
    "name": "harmonic_oscillator_standard",
    "description": "Standard harmonic oscillator benchmark with default MLP",
    "tags": ["continuous", "benchmark", "harmonic_oscillator"]
  },
  "model": {
    "type": "mlp",
    "architecture": [2, 10, 1],
    "activation": "tanh"
  },
  ...
}
```

## **📊 Preset Configurations**

### **harmonic_oscillator_standard**
- **Purpose**: Default benchmark for continuous systems
- **System**: 2D harmonic oscillator with ω=1.0
- **Model**: MLP [2,10,1] with tanh activation
- **Training**: 3000 epochs, batch size 1000
- **Optimizer**: Adam with lr=1e-3

### **ising_1d_benchmark**
- **Purpose**: Benchmark for discrete spin systems
- **System**: 1D Ising model with L=20 spins
- **Model**: MLP [20,10,1] for larger spin space
- **Training**: 2000 epochs, batch size 500
- **Optimizer**: Adam with lr=5e-4 (more conservative)

### **qgt_comparison_test**
- **Purpose**: Test QGT preconditioning implementation
- **System**: 2D harmonic oscillator
- **Model**: MLP [10,20,1] for QGT scaling tests
- **Training**: 1500 epochs, batch size 800
- **Optimizer**: QGT with Cholesky solver

## **🔧 Configuration Override System**

### **Override Syntax**
```bash
# Simple string overrides
--override key=value

# JSON overrides (parsed automatically)
--override 'training.num_epochs=5000'
--override 'optimizer.qgt_config.solver_options.maxiter=2000'

# Multiple overrides
--override key1=value1 --override key2=value2 --override key3=value3
```

### **Override Examples**
```bash
# Change training parameters
qvarnet run --preset harmonic_oscillator_standard \
  --override training.batch_size=2000 \
  --override training.num_epochs=5000 \
  --override optimizer.learning_rate=5e-4

# Modify model architecture
qvarnet run --preset ising_1d_benchmark \
  --override model.architecture=[30,15,1] \
  --override model.activation=relu

# QGT configuration
qvarnet run --preset qgt_comparison_test \
  --override optimizer.qgt_config.solver=gmres \
  --override optimizer.qgt_config.regularization=1e-7
```

## **🔄 Backward Compatibility**

### **Legacy Support**
The old `--filepath` argument still works:
```bash
# Old way (still supported)
qvarnet run --filepath ./cli/parameters/hyperparams.json

# Equivalent new way
qvarnet run --config ./cli/parameters/hyperparams.json
```

### **Migration Path**
1. **Phase 1**: Both systems work in parallel
2. **Phase 2**: Default to new system, warnings for legacy usage
3. **Phase 3**: Remove legacy support (future version)

## **🛠 Error Handling**

### **Configuration Validation**
- **Required keys**: All top-level sections must exist
- **Type validation**: Numeric ranges, parameter constraints
- **JSON schema**: Structured validation for preset files
- **Clear error messages**: Specific guidance for invalid configurations

### **Graceful Fallbacks**
- **Missing presets**: List available options and exit
- **Invalid JSON**: Show parsing error with line number
- **Validation failures**: Clear requirements and suggestions

## **📈 Advanced Features (Future)**

### **Configuration Templates**
- **Template system**: Generate config from templates
- **Parameter sweeps**: Auto-generate multiple configurations
- **Environment-specific**: Different defaults for different compute environments

### **Result Management**
- **Automatic naming**: Timestamp-based experiment directories
- **Result aggregation**: Collect and compare multiple runs
- **Metadata tracking**: Store experiment provenance and parameters

## **🎯 Benefits**

### **Research Productivity**
- **Quick setup**: Preset configurations for common experiments
- **Parameter sweeps**: Easy systematic exploration
- **Reproducibility**: Complete configuration tracking
- **Collaboration**: Share configurations as JSON files

### **Development Efficiency**
- **Modular design**: Easy to add new configuration options
- **Validation**: Catch configuration errors early
- **Documentation**: Self-documenting preset configurations
- **Testing**: Configuration validation in isolation

This enhanced system maintains full backward compatibility while providing powerful new capabilities for systematic experimentation and configuration management.