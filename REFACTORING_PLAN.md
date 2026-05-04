# QVarNet Refactoring & Quality Plan

**Priority Level**: High  
**Estimated Effort**: 40-60 hours  
**Risk Level**: Medium (code should remain functionally identical)

---

## Executive Summary

This codebase has **critical gaps in testing, numerical stability, and maintainability**. While it runs, we cannot trust scientific results without validation. This plan prioritizes testing and correctness, then tackles code quality.

---

## Phase 1: VALIDATION & TESTING (Highest Priority)

### 1.1 Numerical Correctness Tests
**Status**: Not Started  
**Effort**: 8-12 hours  
**Files to Create**: `tests/test_kinetic_energy.py`, `tests/test_hamiltonians.py`

#### 1.1.1 Kinetic Energy Equivalence
- [ ] Test that `kinetic_term_log_wavefunction()` ≈ `kinetic_term_divergence_theorem()` on known models
- [ ] Test that log-space kinetic ≈ direct kinetic (within numerical error)
- [ ] Use finite differences as ground truth for small systems
- [ ] Document why they might differ and acceptable tolerance

**Why**: We have 3 kinetic implementations. If they don't match, we're computing wrong energies.

#### 1.1.2 Hamiltonian Validation
- [ ] Test harmonic oscillator recovers exact GS energy (ω=1 → E_0 = d/2)
- [ ] Test 2D and 3D harmonic oscillators
- [ ] Test nearest-neighbor oscillator on known systems
- [ ] Compare against analytical solutions or literature values

**Why**: Ground truth exists. If we don't recover it on simple systems, results on complex systems are meaningless.

#### 1.1.3 DeepSet Architecture
- [ ] Test permutation invariance: `model(permute(x)) = model(x)` exactly (or to numerical precision)
- [ ] Test aggregation: swapping particles doesn't change output
- [ ] Unit test φ and F networks separately

**Why**: Permutation invariance is the whole point. If it fails, the architecture is broken.

#### 1.1.4 Sampler Diagnostics
- [ ] Verify acceptance rate stays ~30-50% (tune step size)
- [ ] Compute autocorrelation time, report effective sample size
- [ ] Test that MCMC samples match target distribution (KS test or chi-squared)

**Why**: Invalid samples → invalid energies. Need proof chains are actually sampling correctly.

---

### 1.2 Integration Tests
**Status**: Not Started  
**Effort**: 4-6 hours  
**Files to Create**: `tests/test_training.py`, `tests/test_pipeline.py`

- [ ] Full pipeline test: config → train → energy history
- [ ] Energy should monotonically decrease (or stay flat in last epochs)
- [ ] Check for NaN/inf in gradients
- [ ] Verify checkpoint save/load works
- [ ] Test deterministic runs: same seed → identical results

**Why**: Full system tests catch integration bugs that unit tests miss.

---

## Phase 2: NUMERICAL STABILITY (Critical)

### 2.1 Epsilon Consolidation
**Status**: Not Started  
**Effort**: 2-3 hours  
**Files to Modify**: `kinetic.py`, `qgt.py`, `hamiltonian/continuous.py`

- [ ] Create `src/qvarnet/config/numerical_constants.py`:
  ```python
  EPSILON_LOG_STABILITY = 1e-8  # For log(|ψ| + ε) to avoid log(0)
  EPSILON_QGT_REG = 1e-6        # QGT regularization
  EPSILON_GRAD_CLIP = 1e-12     # Gradient clipping threshold
  ```
- [ ] Replace all hardcoded `1e-8`, `1e-12` with imports from this module
- [ ] Add docstring explaining each constant and references
- [ ] Make these configurable via experiment config

**Why**: Currently scattered magic numbers bias results invisibly. Centralized constants are documented and tunable.

---

### 2.2 Kinetic Energy Refactor
**Status**: Not Started  
**Effort**: 4-6 hours  
**Files to Modify**: `kinetic.py`

**Problem**: Three implementations, unclear which to use.

**Solution**:
```python
# kinetic.py
def compute_kinetic_energy(params, samples, model_apply, method='autodiff', is_log_model=False):
    """
    Compute kinetic energy using specified method.
    
    Args:
        method: 'autodiff' (most stable), 'divergence', 'log_space', 'central_diff'
        is_log_model: True if model outputs log(ψ)
    
    Returns:
        kinetic energy of shape (batch,)
    """
    if is_log_model:
        # Only log-space method is valid for log models
        if method != 'autodiff':
            raise ValueError(f"Log model only supports autodiff, got {method}")
        return _kinetic_log_autodiff(params, samples, model_apply)
    
    if method == 'autodiff':
        return _kinetic_autodiff(params, samples, model_apply)
    elif method == 'divergence':
        return _kinetic_divergence(params, samples, model_apply)
    elif method == 'log_space':
        return _kinetic_log_space(params, samples, model_apply)
    elif method == 'central_diff':
        return _kinetic_central_diff(params, samples, model_apply)
    else:
        raise ValueError(f"Unknown method: {method}")
```

**Action Items**:
- [ ] Keep one implementation per method in private functions (`_kinetic_*`)
- [ ] Delete redundant implementations or merge them
- [ ] Add `method` parameter to hamiltonian config
- [ ] Default to log-space (most stable)
- [ ] Test all methods produce ~same result

**Why**: Single interface, explicit choice, tested equivalence.

---

### 2.3 Shape Safety
**Status**: Not Started  
**Effort**: 3-4 hours  
**Files to Modify**: `kinetic.py`, `hamiltonian/`, `models/`

Add shape assertions after reshapes:

```python
def log_psi(x, params, model_apply):
    batch_size = x.shape[0] if x.ndim > 1 else 1
    x_reshaped = x.reshape(batch_size, -1) if x.ndim == 1 else x
    assert x_reshaped.ndim == 2, f"Expected 2D, got {x_reshaped.shape}"
    psi = model_apply(params, x_reshaped)
    assert psi.shape[0] == batch_size, f"Batch mismatch: {psi.shape} vs {batch_size}"
    return jnp.log(jnp.abs(psi) + EPSILON_LOG_STABILITY).squeeze(-1)
```

**Action Items**:
- [ ] Add shape assertions in all forward passes
- [ ] Document expected shapes at function entry points
- [ ] Use helper function for consistent reshaping

**Why**: Catch shape bugs immediately instead of silent broadcasting errors.

---

## Phase 3: TRAIN FUNCTION REFACTORING (See Detailed Plan Below)

**Status**: In Progress (user reports difficulty)  
**Effort**: 6-10 hours  
**Files to Modify**: `train.py` (major restructuring)

See **Section: "How to Split train()" below** for detailed strategy.

---

## Phase 4: CODE CLEANUP

### 4.1 Remove Dead Code
**Status**: Not Started  
**Effort**: 1-2 hours

- [ ] Delete `laplacian_OLD()` (laplacian.py:7-13)
- [ ] Delete commented-out code blocks
- [ ] Remove FIXMEs or convert to issue tickets
- [ ] Check which kinetic implementations are actually used

**Action Items**:
- [ ] Grep for `# TODO`, `# FIXME`, `# XXX` — review each one
- [ ] Grep for commented code blocks — delete or explain
- [ ] Use `git log` if you need history

---

### 4.2 Simplify Vmap Usage
**Status**: Not Started  
**Effort**: 2-3 hours  
**Files to Modify**: `train.py`

**Before**:
```python
sampler_fn = jax.vmap(mh_chain, in_axes=(0, None, None, None, 0, None, None), out_axes=0)
```

**After**:
```python
def vectorized_sampler_fn(random_values, prob_params, init_positions):
    """Vectorize MH chains over random values and initial positions."""
    return jax.vmap(
        mh_chain,
        in_axes=dict(
            random_values=0,
            prob_fn=None,
            prob_params=None, 
            init_position=0,
            step_size=None,
            PBC=None,
            is_log_prob=None,
        ),
        out_axes=0,
    )(...)
```

Or better: make `mh_chain` handle batches natively.

---

### 4.3 Global State Cleanup
**Status**: Not Started  
**Effort**: 1-2 hours  
**Files to Modify**: `train.py`

- [ ] Remove `stop_requested` global
- [ ] Use `KeyboardInterrupt` exception handling instead
- [ ] Or use `multiprocessing.Event()` if you need signals

**Before**:
```python
stop_requested = False
def signal_handler(signum, frame):
    global stop_requested
    stop_requested = True
```

**After**:
```python
class TrainingInterruptedError(Exception):
    pass

def signal_handler(signum, frame):
    raise TrainingInterruptedError("Training interrupted by user")

signal.signal(signal.SIGINT, signal_handler)

try:
    state_history = train(...)
except TrainingInterruptedError:
    print("Gracefully stopped training")
    # state_history has partial results
```

---

### 4.4 Configuration Type Safety
**Status**: Not Started  
**Effort**: 3-4 hours  
**Files to Modify**: `cli_config/base.py`, `cli_config/` (all)

Use `dataclasses` instead of plain dicts:

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass(frozen=True)  # immutable
class ModelConfig:
    type: str
    architecture: List[int]
    activation: str = "tanh"
    n_up: Optional[int] = None
    n_down: Optional[int] = None

@dataclass(frozen=True)
class OptimizerConfig:
    type: str
    learning_rate: float
    
@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    # ... other sections
```

**Why**: Type hints + immutability = safe, self-documenting config.

---

### 4.5 Model Registry: Dependency Injection
**Status**: Not Started  
**Effort**: 2-3 hours  
**Files to Modify**: `models/registry.py`, `main.py`

**Before**: Global `MODEL_REGISTRY` dict  
**After**: Pass registry as argument

```python
# models/registry.py
class ModelRegistry:
    def __init__(self):
        self.models = {}
    
    def register(self, name):
        def decorator(cls):
            self.models[name] = cls
            return cls
        return decorator
    
    def get(self, name):
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found")
        return self.models[name]

# Create singleton instance
DEFAULT_REGISTRY = ModelRegistry()

# In main.py
def run_experiment(args, registry=DEFAULT_REGISTRY):
    model_class = registry.get(model_name)
    ...
```

**Why**: Testable, not polluted by test registrations.

---

## Phase 5: DOCUMENTATION & TESTING INFRASTRUCTURE

### 5.1 Setup Testing Infrastructure
**Status**: Not Started  
**Effort**: 2 hours

- [ ] Create `tests/` directory
- [ ] Create `tests/conftest.py` with fixtures (small test models, hamiltonians)
- [ ] Create `pytest.ini` with JAX configuration
- [ ] Add `tests/README.md` with instructions

```bash
pytest -v                    # Run all tests
pytest tests/test_kinetic_energy.py -k "harmonic"  # Run specific tests
```

---

### 5.2 Document Numerical Choices
**Status**: Not Started  
**Effort**: 2-3 hours  
**Files to Create**: `docs/NUMERICAL_STABILITY.md`

Sections:
- Epsilon values and why
- Float32 vs float64
- Autodiff vs finite differences (trade-offs)
- Why log-space is more stable
- When kinetic energy methods differ

---

### 5.3 Document Known Limitations
**Status**: Not Started  
**Effort**: 1-2 hours  
**Files to Create**: `docs/KNOWN_ISSUES.md`

- List all assumptions (batch shapes, model outputs must be real/complex, etc.)
- List what's not tested yet
- List TODOs
- Known numerical instabilities

---

## Phase 6: OPTIONAL IMPROVEMENTS

### 6.1 Convergence Diagnostics
- [ ] Compute R-hat (Gelman-Rubin)
- [ ] Report autocorrelation time
- [ ] Suggest run length needed for convergence

### 6.2 Visualization
- [ ] Plot energy convergence with error bands
- [ ] Plot acceptance rate over time
- [ ] Plot sample distribution vs. target

### 6.3 Benchmark Suite
- [ ] Known systems (harmonic oscillators, hydrogen atom)
- [ ] Compare against literature results
- [ ] Performance benchmarks (wall-time, memory)

---

## Implementation Order

1. **Week 1**: Phases 1.1-1.2 (Validation tests)
   - Goal: Know current code is numerically correct (or find bugs)
   
2. **Week 2**: Phase 2 (Numerical stability)
   - Goal: Clean up epsilon values, choose one kinetic method
   
3. **Week 3**: Phase 3 (Train refactoring)
   - Goal: Splitup train(), make it testable
   
4. **Week 4**: Phases 4-5 (Cleanup, docs, infrastructure)
   - Goal: Maintainable, tested codebase

---

## Success Criteria

- [ ] 50+ unit tests, all passing
- [ ] 10+ integration tests, all passing  
- [ ] All kinetic energy methods validated against each other
- [ ] Harmonic oscillator recovers exact ground state energy
- [ ] DeepSet proven permutation invariant
- [ ] Zero global mutable state
- [ ] All hardcoded epsilons centralized
- [ ] train() split into ≤5 top-level functions
- [ ] Config system uses type hints
- [ ] Zero FIXMEs or documented exceptions
- [ ] Reproducibility: same seed → identical results every time
- [ ] CI/CD pipeline runs tests on every commit

---

## Rollout Risk Mitigation

**Testing Strategy**:
- Write tests first, run on current code
- Refactor code to pass tests
- Tests act as regression protection

**Backwards Compatibility**:
- All changes should be transparent to users
- Energy results should not change (or change only due to fixed bugs)
- Experiment configs should still work

**Checkpoints**:
- Tag git commits at end of each phase
- If something breaks, we can revert to previous phase

---

## Estimated Timeline

| Phase | Hours | Weeks | Priority |
|-------|-------|-------|----------|
| 1. Validation | 12 | 1.5 | ⚠️ Critical |
| 2. Stability | 8 | 1 | ⚠️ Critical |
| 3. Train split | 8 | 1 | ⚠️ Important |
| 4. Cleanup | 10 | 1 | ✅ Good |
| 5. Docs | 5 | 0.5 | ✅ Good |
| 6. Optional | 10+ | 1+ | 📈 Future |
| **Total** | **53** | **6-7** | |

