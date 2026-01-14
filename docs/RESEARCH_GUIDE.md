# Research Guide for PhD-Level Quantum VMC Work

This guide is specifically designed for PhD researchers using QVarNet for advanced quantum variational Monte Carlo research.

## Research Context and Applications

### Quantum Systems Suitable for QVarNet

QVarNet is particularly well-suited for:

1. **Few-Body Quantum Systems**:
   - Harmonic oscillator (1D, 2D, 3D)
   - Hydrogen-like atoms
   - Quantum dots
   - Small molecules (simplified models)

2. **Model Hamiltonians**:
   - Hubbard model (small clusters)
   - Heisenberg spin chains
   - Transverse field Ising model
   - Quantum billiard problems

3. **Research Applications**:
   - Benchmarking new algorithms
   - Testing novel neural network architectures
   - Method development for VMC
   - Educational demonstrations

### Theoretical Foundation

#### Variational Principle

For any trial wavefunction $\psi_T(\mathbf{x}; \theta)$:
$$E[\theta] = \frac{\langle \psi_T | \hat{H} | \psi_T \rangle}{\langle \psi_T | \psi_T \rangle} \geq E_0$$

where $E_0$ is the true ground state energy.

#### Local Energy Minimization

The energy functional can be written as:
$$E[\theta] = \int d\mathbf{x} \, |\psi_T(\mathbf{x}; \theta)|^2 E_{\text{loc}}(\mathbf{x}; \theta)$$

with local energy:
$$E_{\text{loc}}(\mathbf{x}; \theta) = \frac{\hat{H}\psi_T(\mathbf{x}; \theta)}{\psi_T(\mathbf{x}; \theta)}$$

#### Neural Network Ansätze

General neural network wavefunction:
$$\psi_T(\mathbf{x}; \theta) = \exp(f_{\text{NN}}(\mathbf{x}; \theta))$$

where $f_{\text{NN}}$ is a neural network that outputs log-amplitude.

## Advanced Research Workflows

### 1. Benchmarking New Algorithms

#### Protocol for Algorithm Comparison

```python
def benchmark_algorithm(algorithm_func, test_cases, n_runs=5):
    """Comprehensive benchmarking framework"""
    
    results = []
    
    for case in test_cases:
        case_results = []
        
        for run in range(n_runs):
            # Set different random seed for each run
            rng = jax.random.PRNGKey(42 + run)
            
            # Run algorithm with consistent initial conditions
            start_time = time.perf_counter()
            
            final_energy, convergence_history, metrics = algorithm_func(
                config=case['config'],
                rng_key=rng
            )
            
            end_time = time.perf_counter()
            
            # Collect comprehensive metrics
            result = {
                'case_name': case['name'],
                'run': run,
                'final_energy': final_energy,
                'convergence_steps': len(convergence_history),
                'wall_time': end_time - start_time,
                'energy_variance': jnp.var(convergence_history[-100:]),
                'final_accuracy': abs(final_energy - case['exact_energy']),
                'metrics': metrics
            }
            
            case_results.append(result)
        
        results.append(case_results)
    
    return results

# Example test cases
test_cases = [
    {
        'name': '1D Harmonic Oscillator',
        'config': {'dimensions': 1, 'potential': 'harmonic'},
        'exact_energy': 0.5
    },
    {
        'name': '2D Harmonic Oscillator', 
        'config': {'dimensions': 2, 'potential': 'harmonic'},
        'exact_energy': 1.0
    }
]
```

#### Statistical Analysis

```python
import numpy as np
import scipy.stats as stats

def analyze_benchmark_results(results):
    """Statistical analysis of benchmark results"""
    
    analysis = {}
    
    for case_results in results:
        case_name = case_results[0]['case_name']
        
        # Extract metrics
        final_energies = [r['final_energy'] for r in case_results]
        convergence_times = [r['wall_time'] for r in case_results]
        accuracies = [r['final_accuracy'] for r in case_results]
        
        # Statistical measures
        analysis[case_name] = {
            'energy_mean': np.mean(final_energies),
            'energy_std': np.std(final_energies),
            'energy_ci': stats.t.interval(0.95, len(final_energies)-1,
                                        loc=np.mean(final_energies),
                                        scale=stats.sem(final_energies)),
            'time_mean': np.mean(convergence_times),
            'time_std': np.std(convergence_times),
            'accuracy_mean': np.mean(accuracies),
            'efficiency': np.mean(accuracies) / np.mean(convergence_times)  # accuracy per second
        }
    
    return analysis
```

### 2. Developing Novel Wavefunction Architectures

#### Architecture Design Principles

**Expressivity vs Trainability Trade-off**:
```python
class ExpressiveWavefunction(nn.Module):
    """Highly expressive but potentially hard-to-train wavefunction"""
    
    features: int = 128
    n_layers: int = 4
    use_residual: bool = True
    use_attention: bool = False
    
    @nn.compact
    def __call__(self, x):
        # Feature extraction
        h = x
        for i in range(self.n_layers):
            h_prev = h
            
            # Dense layer with appropriate activation
            h = nn.Dense(self.features)(h)
            h = nn.gelu(h)
            
            # Residual connection
            if self.use_residual and i > 0:
                h = h + h_prev
            
            # Optional attention mechanism
            if self.use_attention:
                h = self.attention_layer(h)
        
        # Output log-probability
        log_psi = nn.Dense(1)(h)
        return log_psi.squeeze()
    
    def attention_layer(self, x):
        """Simple self-attention for particle interactions"""
        # Implement attention mechanism for capturing correlations
        pass
```

**Physics-Informed Architectures**:
```python
class PhysicsInformedWavefunction(nn.Module):
    """Wavefunction with built-in physics constraints"""
    
    n_particles: int
    features: int = 64
    
    @nn.compact
    def __call__(self, x):
        # x shape: (batch, n_particles * dimensions)
        
        # Reshape for particle-wise processing
        batch_size = x.shape[0]
        x_particles = x.reshape(batch_size, self.n_particles, -1)
        
        # Particle-wise encoding
        particle_features = []
        for i in range(self.n_particles):
            xi = x_particles[:, i, :]
            h_i = nn.Dense(self.features)(xi)
            h_i = nn.tanh(h_i)
            particle_features.append(h_i)
        
        # Symmetrization (for bosons) or antisymmetrization (for fermions)
        combined = self.symmetrize(particle_features)
        
        # Jastrow factor for correlations
        jastrow = self.compute_jastrow_factor(x_particles)
        
        # Final output
        log_psi = nn.Dense(1)(combined) + jastrow
        return log_psi.squeeze()
    
    def symmetrize(self, particle_features):
        """Apply appropriate symmetrization"""
        # Implement symmetrization for particle statistics
        return jnp.mean(jnp.stack(particle_features, axis=1), axis=1)
    
    def compute_jastrow_factor(self, x_particles):
        """Compute Jastrow factor for two-body correlations"""
        # Compute pairwise distances and correlations
        return 0.0  # Placeholder
```

#### Training Novel Architectures

```python
def train_novel_architecture(model_class, config, training_data):
    """Specialized training for novel architectures"""
    
    # Initialize model
    model = model_class(**config['model_params'])
    optimizer = optax.adam(learning_rate=config['learning_rate'])
    
    # Custom training loop with monitoring
    def train_step(state, batch):
        def loss_fn(params):
            log_psi = model.apply(params, batch['x'])
            
            # Custom loss incorporating architectural specifics
            energy_loss = compute_energy_loss(log_psi, batch)
            regularization = compute_architecture_regularization(params)
            
            return energy_loss + 0.01 * regularization
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, loss
    
    # Training with extensive logging
    training_log = []
    state = train_state.create(model, optimizer)
    
    for epoch in range(config['n_epochs']):
        # Sample configurations
        batch = sample_configurations(state.params, batch_size=config['batch_size'])
        
        # Training step
        state, loss = train_step(state, batch)
        
        # Extensive monitoring
        if epoch % 100 == 0:
            metrics = {
                'epoch': epoch,
                'loss': loss,
                'energy': compute_expectation_energy(state.params),
                'variance': compute_energy_variance(state.params),
                'norm': compute_wavefunction_norm(state.params),
                'grad_norm': compute_gradient_norm(grads)
            }
            training_log.append(metrics)
            
            print(f"Epoch {epoch}: E = {metrics['energy']:.6f} ± {metrics['variance']:.6f}")
    
    return state, training_log
```

### 3. Systematic Parameter Studies

#### Hyperparameter Optimization

```python
import optuna  # Modern hyperparameter optimization framework

def objective(trial):
    """Objective function for hyperparameter optimization"""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [500, 1000, 2000])
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 2, 5)
    step_size = trial.suggest_float('step_size', 0.1, 2.0)
    
    # Configure model
    config = {
        'optimizer': {'learning_rate': learning_rate, 'type': 'adam'},
        'training': {'batch_size': batch_size, 'n_epochs': 2000},
        'model': {
            'architecture': [2] + [hidden_size] * n_layers + [1],
            'activation': 'tanh'
        },
        'sampler': {'step_size': step_size, 'chain_length': 100}
    }
    
    # Run training
    try:
        final_energy, energy_history = run_experiment_with_config(config)
        
        # Multi-objective optimization
        final_accuracy = abs(final_energy - exact_energy)
        convergence_speed = len(energy_history)
        
        return final_accuracy
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')  # Penalize failed trials

# Run optimization study
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner()
)

study.optimize(objective, n_trials=100)

# Analyze results
print(f"Best trial: {study.best_trial.params}")
print(f"Best accuracy: {study.best_trial.value:.6f}")
```

#### Architecture Search

```python
def architecture_search_space():
    """Define architecture search space"""
    
    architectures = [
        # Shallow networks
        [1, 16, 8, 1],
        [1, 32, 16, 1],
        [1, 64, 32, 1],
        
        # Deep networks
        [1, 32, 32, 32, 1],
        [1, 64, 64, 64, 1],
        
        # Wide networks
        [1, 128, 64, 1],
        [1, 256, 128, 1],
        
        # Specialized architectures
        [1, 8, 8, 8, 8, 1],  # Many small layers
        [1, 256, 1],          # Single wide layer
    ]
    
    results = {}
    
    for arch in architectures:
        config = create_config_for_architecture(arch)
        
        try:
            final_energy, metrics = run_experiment_with_config(config)
            
            results[str(arch)] = {
                'final_energy': final_energy,
                'n_parameters': count_parameters(arch),
                'training_time': metrics['training_time'],
                'convergence_speed': metrics['convergence_speed'],
                'expressivity': metrics['expressivity_measure']
            }
            
        except Exception as e:
            print(f"Architecture {arch} failed: {e}")
            continue
    
    return results
```

## Research Best Practices

### 1. Reproducibility

#### Complete Experiment Documentation

```python
def document_experiment(config, results, metadata):
    """Comprehensive experiment documentation"""
    
    documentation = {
        'experiment_id': generate_uuid(),
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': results,
        'system_info': {
            'python_version': sys.version,
            'jax_version': jax.__version__,
            'cuda_version': get_cuda_version(),
            'gpu_info': str(jax.devices()),
            'environment': get_environment_vars()
        },
        'random_seeds': config.get('random_seeds', {}),
        'git_commit': get_git_commit(),
        'code_diff': get_code_diff(),
        'metadata': metadata
    }
    
    # Save documentation
    with open(f"experiment_{documentation['experiment_id']}.json", 'w') as f:
        json.dump(documentation, f, indent=2, default=str)
    
    return documentation
```

### 2. Statistical Rigor

#### Error Analysis

```python
def statistical_error_analysis(energy_samples):
    """Comprehensive statistical analysis of energy samples"""
    
    samples = np.array(energy_samples)
    
    # Basic statistics
    mean_energy = np.mean(samples)
    std_energy = np.std(samples, ddof=1)
    sem = std_energy / np.sqrt(len(samples))
    
    # Confidence intervals
    ci_95 = stats.t.interval(0.95, len(samples)-1, 
                             loc=mean_energy, scale=sem)
    
    # Autocorrelation analysis
    autocorr = compute_autocorrelation(samples)
    autocorr_time = estimate_autocorrelation_time(autocorr)
    
    # Effective sample size
    n_eff = len(samples) / (2 * autocorr_time)
    effective_error = std_energy / np.sqrt(n_eff)
    
    # Blocking analysis
    blocking_errors = blocking_analysis(samples)
    
    return {
        'mean': mean_energy,
        'std': std_energy,
        'standard_error': sem,
        'confidence_interval_95': ci_95,
        'autocorrelation_time': autocorr_time,
        'effective_sample_size': n_eff,
        'effective_error': effective_error,
        'blocking_errors': blocking_errors,
        'convergence_diagnostics': check_convergence(samples)
    }
```

### 3. Publication-Ready Results

#### Figure Generation

```python
def create_publication_figures(results, output_dir):
    """Generate publication-quality figures"""
    
    # Set up publication style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (6, 4),
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # Energy convergence plot
    fig, ax = plt.subplots()
    ax.plot(results['energy_history'], 'b-', linewidth=1, alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Energy (a.u.)')
    ax.set_title('Energy Convergence')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=results['exact_energy'], color='r', linestyle='--', 
               label=f'Exact: {results["exact_energy"]:.4f}')
    ax.legend()
    plt.savefig(f"{output_dir}/energy_convergence.pdf")
    
    # Wavefunction comparison
    if results['final_wavefunction'] is not None:
        fig, ax = plt.subplots()
        ax.plot(results['x_grid'], results['exact_psi'], 'r-', 
               linewidth=2, label='Exact')
        ax.plot(results['x_grid'], results['approximate_psi'], 'b--', 
               linewidth=2, label='VMC')
        ax.set_xlabel('Position (a.u.)')
        ax.set_ylabel(r'$|\psi(x)|^2$')
        ax.set_title('Wavefunction Comparison')
        ax.legend()
        plt.savefig(f"{output_dir}/wavefunction_comparison.pdf")
```

#### LaTeX Table Generation

```python
def generate_latex_table(benchmark_results):
    """Generate LaTeX table for publication"""
    
    latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Performance comparison of different algorithms}
\\label{tab:algorithm_comparison}
\\begin{tabular}{lccc}
\\hline
Algorithm & Energy Error & Convergence Steps & Time (s) \\\\
\\hline
"""
    
    for result in benchmark_results:
        latex_table += f"{result['name']} & {result['energy_error']:.2e} & "
        latex_table += f"{result['convergence_steps']} & {result['time']:.2f} \\\\\n"
    
    latex_table += """\\hline
\\end{tabular}
\\end{table}
"""
    
    return latex_table
```

## Advanced Topics

### 1. Quantum Monte Carlo Beyond Ground State

#### Excited State Calculations

```python
def orthogonalize_to_ground_state(psi_excited, psi_ground):
    """Orthogonalize excited state to ground state"""
    overlap = jnp.trapz(psi_excited * psi_ground, x_grid)
    psi_orthogonal = psi_excited - overlap * psi_ground
    
    # Renormalize
    norm = jnp.sqrt(jnp.trapz(jnp.abs(psi_orthogonal)**2, x_grid))
    return psi_orthogonal / norm

def excited_state_loss(log_psi, overlap_psi, overlap_weight=10.0):
    """Loss function for excited state calculation"""
    # Standard energy term
    energy_loss = compute_energy_loss(log_psi)
    
    # Orthogonalization penalty
    overlap = jnp.mean(jnp.exp(log_psi) * overlap_psi)
    orthogonalization_loss = overlap_weight * overlap**2
    
    return energy_loss + orthogonalization_loss
```

### 2. Multi-Objective Optimization

```python
def multi_objective_training(params, objectives, weights):
    """Multi-objective optimization for VMC"""
    
    total_loss = 0.0
    objective_values = {}
    
    for name, (objective_fn, weight) in zip(objectives.keys(), objectives.values(), weights):
        value = objective_fn(params)
        objective_values[name] = value
        total_loss += weight * value
    
    return total_loss, objective_values

# Example: Optimize both energy and wavefunction smoothness
objectives = {
    'energy': (lambda p: compute_energy(p), 1.0),
    'smoothness': (lambda p: compute_laplacian_variance(p), 0.1),
    'entropy': (lambda p: compute_information_content(p), 0.01)
}
```

This research guide provides PhD researchers with the advanced tools and methodologies needed for cutting-edge quantum VMC research using QVarNet. The combination of rigorous statistical analysis, systematic experimentation, and publication-ready outputs makes it suitable for high-impact research publications.