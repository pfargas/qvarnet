# How to Split train() — Detailed Strategy

## The Problem You've Hit

The `train()` function is ~400 lines doing:
1. Setup (params, state, history)
2. Define probability function (depends on is_log_model)
3. Define sampling function (depends on MH kernel, vmap config)
4. Define training step (depends on loss function, QGT)
5. Loop with signal handling (depends on state, convergence checks)
6. Save/checkpointing (depends on state history)

**Why splitting is hard**: JAX JIT closures capture surrounding scope. If you extract a function, it needs to:
- Accept everything it depends on as arguments
- Be pure (no side effects)
- Work with JIT compilation

**Why you probably failed**: You likely tried:
```python
def separate_function():
    # Oops, this references variables from outer scope
    return loss(params)  # ❌ loss is not defined here
```

---

## The Right Approach: Separate into LAYERS

Don't try to modularize vertically (split by task). Modularize **horizontally by responsibility**:

```
Current: Giant monolithic function
    ↓
Layer 1: Configuration & Setup (pure functions)
Layer 2: Functional components (JIT-able functions) 
Layer 3: State machine (training loop driver)
Layer 4: I/O & History (checkpointing, saving)
```

---

## Concrete Implementation Plan

### Step 1: Extract Configuration Setup

**Create**: `src/qvarnet/config/training_setup.py`

```python
from dataclasses import dataclass
from typing import Callable, Dict, Any
import jax.numpy as jnp

@dataclass(frozen=True)
class SamplingConfig:
    """Immutable sampling configuration."""
    step_size: float
    chain_length: int
    thermalization_steps: int
    thinning_factor: int
    PBC: float
    is_log_prob: bool

@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration."""
    n_epochs: int
    init_positions: str  # "normal" or "zeros"
    is_update_step_size: bool
    min_step: float
    max_step: float
    target_acceptance: float = 0.5
    adaptation_rate: float = 0.1

def parse_sampler_params(sampler_args: Dict[str, Any]) -> SamplingConfig:
    """Convert dict config to typed dataclass."""
    return SamplingConfig(
        step_size=sampler_args.get("step_size", 1.0),
        chain_length=sampler_args.get("chain_length", 500),
        thermalization_steps=sampler_args.get("thermalization_steps", 50),
        thinning_factor=sampler_args.get("thinning_factor", 5),
        PBC=sampler_args.get("PBC", 40.0),
        is_log_prob=False,  # Set based on model type
    )

def parse_training_params(train_args: Dict[str, Any]) -> TrainingConfig:
    """Convert dict config to typed dataclass."""
    return TrainingConfig(
        n_epochs=train_args.get("num_epochs", 3000),
        init_positions=train_args.get("init_positions", "normal"),
        is_update_step_size=train_args.get("is_update_step_size", False),
        min_step=train_args.get("min_step", 1e-5),
        max_step=train_args.get("max_step", 5.0),
    )
```

**In train.py**: Replace lines 194-198 with:
```python
sampling_config = parse_sampler_params(sampler_params)
training_config = parse_training_params(train_args)
```

**Benefit**: 
- ✅ All defaults in one place
- ✅ Type-safe (IDE autocomplete)
- ✅ Testable: `test_parse_sampler_params()`
- ✅ Config validation in one place

---

### Step 2: Extract Probability Function Builder

**Current code** (train.py:171-183):
```python
if not is_log_model:
    def prob_fn(x, params):
        forward = model.apply(params, x).flatten()
        out = jnp.square(forward)
        return jnp.squeeze(out)
else:
    def prob_fn(x, params):
        forward = model.apply(params, x).flatten()
        out = 2 * forward
        return jnp.squeeze(out)
```

**Create**: `src/qvarnet/probability.py`

```python
from functools import partial
import jax
import jax.numpy as jnp

def build_prob_fn(model_apply, is_log_model: bool):
    """
    Factory function to create probability function matching model type.
    
    Args:
        model_apply: Flax model apply function
        is_log_model: If True, model outputs log(ψ). Else outputs ψ.
    
    Returns:
        Probability function: (x, params) -> ℝ
    """
    @partial(jax.jit, static_argnames=[])
    def prob_fn_direct(x, params):
        """ψ² for direct models."""
        forward = model_apply(params, x).flatten()
        psi_squared = jnp.square(forward)
        return jnp.squeeze(psi_squared)
    
    @partial(jax.jit, static_argnames=[])
    def prob_fn_log(x, params):
        """e^(2*log ψ) = ψ² for log models."""
        forward = model_apply(params, x).flatten()
        log_psi_squared = 2 * forward
        return jnp.squeeze(log_psi_squared)
    
    if is_log_model:
        return prob_fn_log
    else:
        return prob_fn_direct
```

**In train.py**: Replace lines 171-183 with:
```python
from .probability import build_prob_fn

prob_fn = build_prob_fn(model.apply, is_log_model=is_log_model)
```

**Benefit**:
- ✅ Probability logic centralized
- ✅ Easy to test: `test_prob_fn_matches_model_output()`
- ✅ Can add new model types without touching train()
- ✅ Can JIT compile both branches independently

---

### Step 3: Extract Sampling Step

**Current code** (train.py:211-231):
```python
def sample_and_process(key, params, init_pos, ...):
    # Complex JIT with many static args
    # Generates random numbers
    # Calls sampler_fn
    # Reshapes batches
```

**Create**: `src/qvarnet/sampling_step.py`

```python
from functools import partial
import jax
import jax.numpy as jnp
from jax import random

def create_sampler_fn(mh_chain, n_chains: int, DoF: int):
    """
    Create a vectorized sampler over multiple chains.
    
    This handles the vmap setup that's hard to understand.
    
    Args:
        mh_chain: Single-chain Metropolis-Hastings kernel
        n_chains: Number of parallel chains
        DoF: Degrees of freedom per sample
    
    Returns:
        sampler_fn: (random_values, params, init_pos, step_size, PBC, prob_fn, is_log_prob) 
                    -> (samples, acceptance_rates)
    """
    sampler_fn = jax.vmap(
        mh_chain,
        in_axes=(
            0,      # random_values: batch over chains
            None,   # prob_fn: same for all chains
            None,   # prob_params: same for all chains
            0,      # init_position: batch over chains
            None,   # step_size: same for all chains
            None,   # PBC: same for all chains
            None,   # is_log_prob: same for all chains
        ),
        out_axes=0,  # Output: batch over chains
    )
    return sampler_fn


@partial(jax.jit, static_argnames=[
    "n_chains", "DoF", "n_steps", "burn_in", "thinning", "PBC", "is_log_prob"
])
def sample_and_process(
    key,
    prob_fn,
    prob_params,
    init_positions,
    step_size,
    n_chains: int,
    DoF: int,
    n_steps: int,
    burn_in: int,
    thinning: int,
    PBC: float,
    is_log_prob: bool,
):
    """
    Generate one batch of samples from MCMC and process them.
    
    Args:
        key: JAX random key
        prob_fn: Probability function (x, params) -> ℝ
        prob_params: Model parameters
        init_positions: Starting positions, shape (n_chains, DoF)
        step_size: MH proposal step size
        n_chains, DoF, n_steps, burn_in, thinning: Config ints
        PBC: Periodic boundary condition size
        is_log_prob: If True, use log-space MH kernel
    
    Returns:
        samples: Processed batch, shape (n_chains * n_samples, DoF)
        last_positions: Final positions, shape (n_chains, DoF)
        acceptance_rates: Per-chain acceptance rates
    """
    # Generate random numbers for all chains
    rand_nums = random.uniform(key, (n_chains, n_steps, DoF + 1))
    
    # Run MH chains (vmap handles parallelization)
    sampler_fn = create_sampler_fn(mh_chain, n_chains, DoF)
    raw_batch, acceptance_rates = sampler_fn(
        rand_nums, prob_fn, prob_params, init_positions, 
        step_size, PBC, is_log_prob
    )
    
    # Post-process: drop burn-in, apply thinning
    processed_batch = raw_batch[:, burn_in::thinning, :]
    last_positions = raw_batch[:, -1, :]
    batch_flat = processed_batch.reshape(-1, DoF)
    
    return batch_flat, last_positions, acceptance_rates
```

**In train.py**: Replace lines 151-163 and 211-231 with:
```python
from .sampling_step import create_sampler_fn, sample_and_process

n_chains, DoF = shape
sampler_fn = create_sampler_fn(mh_chain, n_chains, DoF)

# In training loop:
batch, last_positions, acc_rates = sample_and_process(
    key=key,
    prob_fn=prob_fn,
    prob_params=state.params,
    init_positions=current_positions,
    step_size=step_size,
    n_chains=n_chains,
    DoF=DoF,
    n_steps=sampling_config.chain_length,
    burn_in=sampling_config.thermalization_steps,
    thinning=sampling_config.thinning_factor,
    PBC=sampling_config.PBC,
    is_log_prob=sampling_config.is_log_prob,
)
```

**Benefit**:
- ✅ vmap magic is explained with explicit in_axes dict (when you refactor further)
- ✅ Sampling is a pure function (testable)
- ✅ Can unit test: does it produce correct batch shapes?
- ✅ Can mock for training loop tests

---

### Step 4: Extract Training Step

**Current code** (train.py:87-111 for `train_step` but called as part of full_update):

**Create**: `src/qvarnet/training_step.py`

```python
from functools import partial
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .qgt import compute_natural_gradient, DEFAULT_QGT_CONFIG


@partial(jax.jit, static_argnames=["is_log_model", "use_qgt"])
def compute_step(
    state,
    batch,
    hamiltonian,
    is_log_model: bool,
    use_qgt: bool = False,
    qgt_config: dict = None,
):
    """
    Compute energy, gradients, and apply update.
    
    Args:
        state: VMCState (params, optimizer state, etc.)
        batch: Training batch of samples
        hamiltonian: Energy operator
        is_log_model: If model outputs log(ψ)
        use_qgt: If True, use natural gradient (QGT). Else vanilla gradient.
        qgt_config: Config dict for QGT solver
    
    Returns:
        new_state: Updated VMCState
        energy: Scalar energy estimate
        std: Standard error of energy estimate
    """
    if qgt_config is None:
        qgt_config = DEFAULT_QGT_CONFIG.to_dict()
    
    # Compute energy and gradients (from train.py:70-84, but extracted)
    E, local_energies, sigma_e = _energy_and_grads(
        hamiltonian, state.params, batch, state.apply_fn, is_log_model
    )
    
    grads = _compute_gradients(
        batch, hamiltonian, state.params, state.apply_fn, 
        E, local_energies, is_log_model
    )
    
    # Apply update
    if not use_qgt:
        new_state = state.apply_gradients(grads=grads)
    else:
        new_state = _apply_natural_gradient_update(
            state, grads, batch, qgt_config
        )
    
    return new_state, E, sigma_e


def _energy_and_grads(hamiltonian, params, batch, model_apply, is_log_model):
    """Helper: compute energy and local energies at samples."""
    local_energy_per_point = hamiltonian.local_energy(
        params, batch, model_apply, is_log_model=is_log_model
    ).reshape(-1, 1)
    
    E = jnp.mean(local_energy_per_point)
    sigma_e = jnp.std(local_energy_per_point)
    
    return E, local_energy_per_point, sigma_e


def _compute_gradients(batch, hamiltonian, params, model_apply, E, local_energies, is_log_model):
    """Helper: compute loss gradients w.r.t. parameters."""
    def log_psi(x):
        psi = model_apply(params, x)
        return jnp.log(jnp.abs(psi)).squeeze()
    
    if not is_log_model:
        loss = lambda p: 2 * jnp.mean(
            jax.lax.stop_gradient(local_energies - E)
            * jax.vmap(lambda x: log_psi(x))(batch).reshape(-1, 1)
        )
    else:
        loss = lambda p: 2 * jnp.mean(
            jax.lax.stop_gradient(local_energies - E)
            * model_apply(p, batch).reshape(-1, 1)
        )
    
    return jax.grad(loss)(params)


def _apply_natural_gradient_update(state, grads, batch, qgt_config):
    """Helper: update using QGT natural gradient instead of vanilla gradient."""
    natural_grad_flat, unravel_fn = compute_natural_gradient(
        state.params, batch, state.apply_fn, grads, qgt_config
    )
    learning_rate = qgt_config.get("learning_rate", 1e-3)
    new_params_flat = (
        ravel_pytree(state.params)[0] - learning_rate * natural_grad_flat
    )
    new_params = unravel_fn(new_params_flat)
    return state.replace(params=new_params)
```

**In train.py**: Replace lines 70-111 with:
```python
from .training_step import compute_step

new_state, energy, std = compute_step(
    state=state,
    batch=batch,
    hamiltonian=hamiltonian,
    is_log_model=is_log_model,
    use_qgt=use_qgt,
    qgt_config=qgt_config,
)
```

**Benefit**:
- ✅ Energy computation isolated and testable
- ✅ Can test that gradients match finite differences
- ✅ Can test QGT vs vanilla gradient separately
- ✅ Can mock for higher-level tests

---

### Step 5: Extract History & Checkpointing

**Create**: `src/qvarnet/training_state.py`

```python
from typing import List
from dataclasses import dataclass, field
from pathlib import Path
import jax
from .vmc_state import VMCState
from .utils import save_checkpoint, load_checkpoint

@dataclass
class TrainingHistory:
    """Track training progress."""
    energies: List[float] = field(default_factory=list)
    stds: List[float] = field(default_factory=list)
    acceptance_rates: List[float] = field(default_factory=list)
    step_sizes: List[float] = field(default_factory=list)
    
    def record(self, energy, std, acceptance_rate, step_size):
        """Record one epoch."""
        self.energies.append(float(energy))
        self.stds.append(float(std))
        self.acceptance_rates.append(float(acceptance_rate))
        self.step_sizes.append(float(step_size))
    
    def to_arrays(self):
        """Convert to JAX arrays for saving."""
        import jax.numpy as jnp
        return {
            'energies': jnp.array(self.energies),
            'stds': jnp.array(self.stds),
            'acceptance_rates': jnp.array(self.acceptance_rates),
            'step_sizes': jnp.array(self.step_sizes),
        }


class StateManager:
    """Manage VMC training state and checkpointing."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = Path(checkpoint_path)
        self.history = TrainingHistory()
    
    def load_checkpoint(self, state: VMCState) -> VMCState:
        """Load checkpoint if it exists."""
        return load_checkpoint(
            state, 
            path=str(self.checkpoint_path),
            filename="checkpoint.msgpack"
        )
    
    def save_checkpoint(self, state: VMCState):
        """Save current state."""
        save_checkpoint(
            state,
            path=str(self.checkpoint_path),
            filename="checkpoint.msgpack"
        )
    
    def record_epoch(self, energy, std, acceptance_rate, step_size):
        """Record one training epoch."""
        self.history.record(energy, std, acceptance_rate, step_size)
```

**In train.py**: Replace checkpoint and history management with:
```python
from .training_state import StateManager

state_manager = StateManager(checkpoint_path)
state = state_manager.load_checkpoint(state)

# In training loop:
state_manager.record_epoch(energy, sigma_e, acceptance_rate, step_size)

# After training:
state_manager.save_checkpoint(state)
energy_hist = state_manager.history.to_arrays()
```

**Benefit**:
- ✅ State management logic isolated
- ✅ Can unit test: does it save/load correctly?
- ✅ Can mock for training loop tests

---

### Step 6: Main Training Loop

**After extracting the above**, train.py becomes:

```python
def train(
    n_epochs,
    shape,
    model,
    optimizer,
    sampler_params,
    hamiltonian,
    rng_seed=0,
    checkpoint_path="./",
    save_checkpoints=False,
    is_log_model=False,
    **kwargs  # Catch remaining args
):
    """High-level training loop."""
    from jax import random
    
    # Setup
    key = random.PRNGKey(rng_seed)
    sampling_config = parse_sampler_params(sampler_params)
    training_config = parse_training_params(kwargs)
    
    # Initialize model and state
    params = model.init(key, jnp.ones(shape))
    state = VMCState.create(apply_fn=model.apply, params=params, tx=optimizer)
    
    # Load checkpoint if exists
    state_manager = StateManager(checkpoint_path)
    state = state_manager.load_checkpoint(state)
    
    # Build functions
    prob_fn = build_prob_fn(model.apply, is_log_model)
    current_positions = _init_positions(
        key, shape, training_config.init_positions
    )
    
    step_size = sampling_config.step_size
    n_chains, DoF = shape
    
    # Training loop
    try:
        for epoch in range(n_epochs):
            key, subkey = random.split(key)
            
            # Sample
            batch, current_positions, acc_rates = sample_and_process(
                key=subkey,
                prob_fn=prob_fn,
                prob_params=state.params,
                init_positions=current_positions,
                step_size=step_size,
                n_chains=n_chains,
                DoF=DoF,
                n_steps=sampling_config.chain_length,
                burn_in=sampling_config.thermalization_steps,
                thinning=sampling_config.thinning_factor,
                PBC=sampling_config.PBC,
                is_log_prob=sampling_config.is_log_prob,
            )
            
            # Train step
            state, energy, std = compute_step(
                state=state,
                batch=batch,
                hamiltonian=hamiltonian,
                is_log_model=is_log_model,
                use_qgt=False,  # TODO: make configurable
            )
            
            # Adapt step size
            acceptance_rate = jnp.mean(acc_rates)
            if training_config.is_update_step_size:
                step_size = update_step_size(
                    step_size, acc_rates,
                    training_config.min_step,
                    training_config.max_step,
                )
            
            # Record
            state_manager.record_epoch(energy, std, acceptance_rate, step_size)
            
            # Checkpoint
            if save_checkpoints and epoch % 100 == 0:
                state_manager.save_checkpoint(state)
            
            # Progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: E={energy:.4f} ± {std:.4f}, acc={acceptance_rate:.2f}")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    # Save results
    if save_checkpoints:
        state_manager.save_checkpoint(state)
    
    return state_manager.history.energies, state_manager


def _init_positions(key, shape, init_type):
    """Initialize walker positions."""
    if init_type == "normal":
        return jax.random.normal(key, shape) * 0.5
    elif init_type == "zeros":
        return jnp.zeros(shape)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
```

---

## Summary: What Each File Does

| File | Responsibility | Lines |
|------|---|---|
| `config/training_setup.py` | Parse & validate config | 40 |
| `probability.py` | Build prob functions | 30 |
| `sampling_step.py` | Generate MCMC samples | 60 |
| `training_step.py` | Compute energy & update | 70 |
| `training_state.py` | History & checkpointing | 50 |
| `train.py` | Main loop | ~80 |
| **Total** | | **330** |

vs. current monolithic train.py: **400 lines, 1 function**

---

## Why This Works

1. **No JIT closure issues**: Each function is self-contained with explicit args
2. **Testable**: Mock dependencies and unit test each layer
3. **Reusable**: Use `sample_and_process()` elsewhere if needed
4. **Readable**: `train()` loop is now the control flow (what it should be)
5. **Maintainable**: Change sampling? Edit `sampling_step.py` only

---

## Testing Strategy After Refactoring

```python
# tests/test_sampling_step.py
def test_sample_and_process_shapes():
    key = random.PRNGKey(0)
    prob_fn = lambda x, p: jnp.ones(x.shape[0])  # Mock
    params = jnp.array([1.0])
    
    batch, last_pos, acc = sample_and_process(
        key=key,
        prob_fn=prob_fn,
        prob_params=params,
        init_positions=jnp.zeros((4, 3)),  # 4 chains, 3 DoF
        step_size=0.5,
        n_chains=4,
        DoF=3,
        n_steps=100,
        burn_in=10,
        thinning=2,
        PBC=40.0,
        is_log_prob=False,
    )
    
    assert batch.shape == (4 * 45, 3)  # 4 chains * (100-10)/2 samples
    assert last_pos.shape == (4, 3)
    assert acc.shape == (4,)

# tests/test_training_step.py
def test_energy_decreases():
    state_before = ... # Create initial state
    batch = jnp.randn(100, 2)  # 100 samples, 2D
    
    state_after, E_after, std_after = compute_step(...)
    
    # Energy should be reasonable (not NaN, not infinite)
    assert jnp.isfinite(E_after)
    assert std_after > 0

# tests/test_train_integration.py
def test_full_training_loop(simple_hamiltonian, simple_model):
    history, manager = train(
        n_epochs=10,
        shape=(4, 2),
        model=simple_model,
        optimizer=optax.adam(1e-3),
        sampler_params={...},
        hamiltonian=simple_hamiltonian,
        is_log_model=False,
    )
    
    # Energy should generally decrease (or be noisy but not diverging)
    assert len(history) == 10
    assert all(jnp.isfinite(E) for E in history)
```

---

## Migration Path

**Phase 1**: Extract setup & probability (low risk)
```bash
1. Create config/training_setup.py
2. Create probability.py  
3. Update train() to use new modules
4. Run existing tests (if any) — should pass
```

**Phase 2**: Extract sampling (medium risk)
```bash
1. Create sampling_step.py
2. Update train() sampling logic
3. Add unit tests for sampling
4. Verify sampler still works
```

**Phase 3**: Extract training step (medium risk)
```bash
1. Create training_step.py
2. Update train() training logic
3. Add gradient tests
4. Verify energy convergence
```

**Phase 4**: Extract state management (low risk)
```bash
1. Create training_state.py
2. Update checkpoint/history logic
3. Verify loading/saving works
```

**Phase 5**: Simplify main loop (refactor)
```bash
1. Rewrite train() to be ~80 lines
2. Verify behavior unchanged
3. Add integration tests
```

---

## Key Points

- ✅ Each module is **independently testable**
- ✅ No JAX JIT closure issues (all dependencies explicit)
- ✅ train() becomes a **readable control flow**, not a giant blob
- ✅ Can **reuse sampling/training logic** elsewhere
- ✅ **Incremental refactoring**: do one phase at a time, test each
- ✅ **No breaking changes**: user-facing API stays the same

Good luck! This approach works because you're separating **concerns**, not **tasks**.
