# qvarnet tutorial

`qvarnet` is a JAX library for Variational Monte Carlo (VMC) with neural-network wavefunctions. This document covers everything you need to go from zero to a working VMC run, extend the library with custom physics, and understand what's happening under the hood.

---

## Table of contents

1. [Background: what VMC does](#1-background)
2. [Your first run in 15 lines](#2-your-first-run)
3. [train() — the full API](#3-the-train-function)
4. [TrainingConfig reference](#4-trainingconfig)
5. [SamplingConfig (sampler_params)](#5-samplingconfig)
6. [Models](#6-models)
7. [Hamiltonians](#7-hamiltonians)
8. [Coordinate modes](#8-coordinate-modes)
9. [Callbacks](#9-callbacks)
10. [Auxiliary losses](#10-auxiliary-losses)
11. [Checkpointing and resuming](#11-checkpointing)
12. [Analysing results](#12-analysing-results)
13. [Custom Hamiltonian](#13-custom-hamiltonian)
14. [Custom model](#14-custom-model)
15. [Stochastic reconfiguration (QGT)](#15-stochastic-reconfiguration)
16. [Hutchinson Laplacian estimator](#16-hutchinson)
17. [Common mistakes](#17-common-mistakes)

---

## 1. Background

VMC approximates the ground state of a quantum system by representing the wavefunction as a neural network ψ_θ(x) and minimising the variational energy

```
E[θ] = ⟨ψ_θ|Ĥ|ψ_θ⟩ / ⟨ψ_θ|ψ_θ⟩  ≥  E₀
```

`qvarnet` always works with **log|ψ_θ(x)|** — the model outputs the log of the amplitude, not ψ itself. This is numerically stable and simplifies the kinetic energy formula:

```
E_kin(x) = -½ [Δ log|ψ| + |∇ log|ψ||²]
```

The training loop:
1. Draw samples from |ψ_θ|² via Metropolis-Hastings MCMC.
2. Compute the local energy E_loc(x) = Ĥψ/ψ at each sample.
3. Estimate the gradient ∇_θ ⟨E⟩ and update θ with Adam (or natural gradient).
4. Repeat.

---

## 2. Your first run

```python
import optax
from qvarnet.train import train
from qvarnet.config.training_setup import TrainingConfig
from qvarnet.config.coord_mode import LabCoords
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.models.mlp import MLP

result = train(
    shape=(512, 3),                           # 512 MCMC chains, 3 degrees of freedom
    model=MLP(architecture=[3, 64, 64, 1]),   # input dim must match dof
    optimizer=optax.adam(1e-3),
    hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
    training_config=TrainingConfig(
        n_epochs=2000,
        checkpoint_path="./outputs/ho_3d",
    ),
    sampler_params={
        "step_size": 0.3,
        "chain_length": 200,
    },
    coord_mode=LabCoords(),
)

energies = [float(s.energy) for s in result.history]
best = result.best(n=1)[0]
print(f"Best energy: {float(best.energy):.6f}")
```

`shape = (n_chains, dof)` where `dof = n_particles * n_dim`.  The exact ground state of a 3D harmonic oscillator is 1.5 (ℏω/2 per mode).

---

## 3. The `train()` function

```python
def train(
    shape,                          # (n_chains, dof)  — MCMC shape
    model,                          # Flax nn.Module outputting log|ψ|
    optimizer,                      # any optax optimizer
    hamiltonian,                    # ContinuousHamiltonian subclass
    training_config: TrainingConfig,
    sampler_params,                 # dict or SamplingConfig
    coord_mode: CoordMode = None,   # default: LabCoords()
    model_name: str = None,         # for load_run() later
    model_args: dict = None,        # for load_run() later
    qgt_config = None,              # QGTConfig — only used when use_qgt=True
    auxiliary_losses: tuple = (),   # extra loss terms (cusp, BC, ...)
    callbacks: list = None,         # Callback instances
) -> TrainResult
```

`model_name` and `model_args` are optional but required if you want to call `load_run()` afterwards. Set both or neither.

`coord_mode` defaults to `LabCoords()`. Use `JacobiCoords` if you want the sampler to explore Jacobi coordinates (removes centre-of-mass drift) while the model sees lab coordinates.

---

## 4. TrainingConfig

```python
from qvarnet.config.training_setup import TrainingConfig, CuspConfig

cfg = TrainingConfig(
    n_epochs            = 5000,
    checkpoint_path     = "./outputs/my_run",
    save_checkpoints    = True,     # save every 50 steps
    rng_seed            = 42,
    init_positions      = "normal", # or "zeros"

    # Step-size adaptation (Metropolis acceptance target)
    is_update_step_size = True,
    target_acceptance   = 0.5,
    min_step            = 0.01,
    max_step            = 2.0,
    adaptation_rate     = 0.01,

    warm_walkers        = True,     # use new walker positions after each step

    # Natural gradient
    use_qgt             = False,    # set True to use stochastic reconfiguration

    # Cusp condition (optional, for singular potentials)
    cusp = None,                    # or CuspConfig(alpha=0.01, epsilon=1e-2, ...)
)
```

**All fields are frozen** — `TrainingConfig` is a frozen dataclass and lives in JAX's `static_argnames`. Changing any field during training triggers a JIT retrace.

### CuspConfig

Enforces the correct cusp condition ∂log|ψ|/∂r_ij → C_n / ε^(n/2) as two particles approach each other. Used for singular potentials (inverse-square, Coulomb, etc.).

```python
cusp = CuspConfig(
    alpha             = 0.01,  # penalty strength
    epsilon           = 1e-2,  # near-coalescence radius
    n                 = 2.0,   # power of the singularity
    C_n               = 1.0,   # target logarithmic derivative at contact
    n_configs_per_pair = 5,    # near-coalescence configs per particle pair
    rng_seed          = 42,
)
```

When `cusp` is set in `TrainingConfig`, a `CuspLoss` is automatically added to the auxiliary losses.

---

## 5. SamplingConfig (sampler_params)

Pass as a plain dict. All keys are optional — unset keys use the defaults below.

| Key | Default | Meaning |
|-----|---------|---------|
| `step_size` | 1.0 | Initial MH step size |
| `chain_length` | 200 | MCMC steps per epoch |
| `thermalization_steps` | 50 | Burn-in steps (must be < chain_length) |
| `thinning_factor` | 1 | Keep every Nth sample |
| `PBC` | 40.0 | Periodic box size (used only with PBC Hamiltonians) |

```python
sampler_params = {
    "step_size": 0.3,
    "chain_length": 300,
    "thermalization_steps": 30,
    "thinning_factor": 2,
}
```

**Step-size adaptation**: when `is_update_step_size=True` in `TrainingConfig`, the step size is adapted each epoch toward `target_acceptance` (default 0.5). The initial `step_size` still matters — set it in the right ballpark for your system.

---

## 6. Models

All models live in `qvarnet.models` and output log|ψ(x)|.

### MLP

```python
from qvarnet.models.mlp import MLP

model = MLP(
    architecture = [dof, 64, 64, 1],  # input → hidden → ... → 1
    hidden_activation = nn.tanh,       # default
)
```

`architecture[0]` must equal `dof` (or the `coord_mode` model input dimension).

### DeepSet

Permutation-invariant ansatz: φ network per particle → sum aggregation → F network.

```python
from qvarnet.models.deep_set import DeepSet

model = DeepSet(
    n_particles               = 5,
    n_dim                     = 1,
    phi_hidden_architecture   = [16, 16],
    F_hidden_architecture     = [32, 32],
    hidden_internal_dimension = 16,
)
```

Use for bosonic/distinguishable systems where ψ must be symmetric under particle exchange.

### ExponentialDeepSet

Like `DeepSet` but with a Gaussian or quartic exponential envelope, useful when the wavefunction decays rapidly at large distances.

```python
from qvarnet.models.deep_set import ExponentialDeepSet
```

### Input transforms (first-layer preprocessing)

Can be prepended to any model to provide translation-invariant or pairwise features:

```python
from qvarnet.models.layers import SubtractCM, AppendPairwiseDiffs
```

- `SubtractCM`: removes the centre of mass from all coordinates.
- `AppendPairwiseDiffs`: appends `x_i - x_j` for all pairs to the input.

### MODEL_REGISTRY

All models decorated with `@register_model("name")` can be looked up:

```python
from qvarnet.models.registry import MODEL_REGISTRY
model = MODEL_REGISTRY["mlp"].from_config({"architecture": [3, 32, 1]})
```

This is used internally by `load_run()`.

---

## 7. Hamiltonians

All Hamiltonians are Flax `@struct.dataclass` so they are JAX pytree-compatible.

### Built-in Hamiltonians

```python
from qvarnet.hamiltonian.continuous import (
    HarmonicOscillatorHamiltonian,   # V = ½ω² Σxᵢ²
    NN_OscillatorHamiltonian,         # trap + nearest-neighbour harmonic
    CalogeroSutherlandHamiltonian,    # inverse-square interactions on a line
)
```

```python
hamiltonian = HarmonicOscillatorHamiltonian(omega=1.0)
hamiltonian = CalogeroSutherlandHamiltonian(L=2.0, omega_trap=1.0)
```

### Laplacian method

`ContinuousHamiltonian` subclasses have a `laplacian_method` field:

| Value | Method | Cost | When to use |
|-------|--------|------|-------------|
| `"forward_ad"` | Forward-over-reverse AD | O(dof) JVPs | Default — recommended |
| `"central_difference"` | Finite differences | O(dof) model evals | No AD available |
| `"full_hessian"` | Full Hessian trace | O(dof²) | Debug only |

```python
hamiltonian = HarmonicOscillatorHamiltonian(
    omega=1.0,
    laplacian_method="forward_ad",   # default
)
```

### HAMILTONIAN_REGISTRY

```python
from qvarnet.hamiltonian.hamiltonian_registry import HAMILTONIAN_REGISTRY
hamiltonian = HAMILTONIAN_REGISTRY["harmonic-oscillator"](omega=1.0)
```

---

## 8. Coordinate modes

`coord_mode` controls the relationship between the space where MCMC samples live and what the model sees.

### LabCoords (default)

```python
from qvarnet.config.coord_mode import LabCoords
coord_mode = LabCoords()
```

Sampler and model both work in lab coordinates `(x₁, x₂, ..., xN)`. Model input shape = `(n_chains, N*d)`. Simple, correct for most cases.

### JacobiCoords

```python
from qvarnet.config.coord_mode import JacobiCoords
coord_mode = JacobiCoords(n_particles_physical=N, n_dim=d)
```

The MCMC sampler explores Jacobi coordinates (which decouple the centre-of-mass from relative motion). The model automatically receives lab coordinates via an inverse transform. Use this when:
- The wavefunction doesn't depend on the CM position (translationally invariant system)
- You want to avoid CM drift without using `SubtractCM`

With `JacobiCoords`, the model init shape is `(n_chains, N*d)` (lab), while the sampler shape is `(n_chains, (N-1)*d)`. The CM coordinate is frozen to zero in Jacobi space.

---

## 9. Callbacks

Callbacks run in Python after each training step, outside of JAX JIT. They can stop training early.

### Built-in callbacks

These are added automatically by `train()` based on `TrainingConfig`:

| Callback | Trigger | Behaviour |
|----------|---------|-----------|
| `NaNCallback` | Always | Stops and saves `nan_checkpoint.msgpack` if energy is NaN |
| `CheckpointCallback` | `save_checkpoints=True` | Saves `checkpoint.msgpack` every 50 steps |
| `ProgressCallback` | tqdm available | Updates tqdm bar with E and σ_E every 10 steps |

### Custom callbacks

Subclass `Callback` and implement `on_step_end`:

```python
from qvarnet.callbacks import Callback

class EarlyStopping(Callback):
    def __init__(self, threshold_std):
        self.threshold_std = threshold_std

    def on_step_end(self, step, state, metrics) -> bool:
        # Return True to stop training
        return metrics["std"] < self.threshold_std

    def on_train_end(self, state, history):
        print(f"Training ended at step {len(history)}")

result = train(
    ...,
    callbacks=[EarlyStopping(threshold_std=0.001)],
)
```

**`metrics` dict keys:**

| Key | Type | Description |
|-----|------|-------------|
| `"energy"` | float | ⟨E⟩ this epoch |
| `"std"` | float | σ(E_loc) this epoch |
| `"acceptance_rate"` | array | Per-chain MH acceptance (shape: n_chains) |
| `"step_size"` | float | Current MH step size |
| `"cm_mean"` | float | Mean centre-of-mass |
| `"cm_std"` | float | Std of centre-of-mass |

**`state`** is the full `VMCState` including params — use it for checkpointing or saving the best model.

### WandB logging example

```python
import wandb
from qvarnet.callbacks import Callback

class WandBCallback(Callback):
    def __init__(self, project):
        wandb.init(project=project)

    def on_step_end(self, step, state, metrics):
        wandb.log({"energy": metrics["energy"], "std": metrics["std"]}, step=step)
        return False

    def on_train_end(self, state, history):
        wandb.finish()
```

---

## 10. Auxiliary losses

Auxiliary losses add differentiable penalty terms on top of the VMC energy loss. They run inside JAX JIT (must be JAX-traceable — use `jnp` operations only, no Python control flow on array values).

### CuspLoss (built-in, auto-added via TrainingConfig)

```python
from qvarnet.config.training_setup import TrainingConfig, CuspConfig

cfg = TrainingConfig(
    ...,
    cusp=CuspConfig(alpha=0.01, epsilon=1e-2, n=2.0, C_n=1.0),
)
```

When `cusp` is set, `CuspLoss` is added automatically. You don't need to pass it to `auxiliary_losses`.

### Custom auxiliary losses

Subclass `AuxiliaryLoss`. The `__call__` method receives `(params, model_apply, batch)` and must return a JAX scalar.

```python
from qvarnet.losses import AuxiliaryLoss
import jax.numpy as jnp

class L2Regulariser(AuxiliaryLoss):
    """Penalise large parameter values: λ Σ θᵢ²"""
    def __init__(self, lam: float):
        self.lam = lam

    def __call__(self, params, model_apply, batch):
        import jax
        leaves = jax.tree_util.tree_leaves(params)
        return self.lam * sum(jnp.sum(p**2) for p in leaves)


class BoundaryPenalty(AuxiliaryLoss):
    """Enforce ψ(±L) ≈ 0 for hard-wall boundary conditions."""
    def __init__(self, L: float, alpha: float, n_pts: int = 20):
        import jax.numpy as jnp
        self.L = L
        self.alpha = alpha
        # Sample boundary points once at construction time (outside JIT)
        self.wall_pts = jnp.concatenate([
            jnp.full((n_pts, 1), -L),
            jnp.full((n_pts, 1),  L),
        ])

    def __call__(self, params, model_apply, batch):
        log_psi_wall = model_apply(params, self.wall_pts).squeeze()
        return self.alpha * jnp.mean(log_psi_wall**2)


result = train(
    ...,
    auxiliary_losses=(L2Regulariser(lam=1e-5), BoundaryPenalty(L=5.0, alpha=0.1)),
)
```

### Orthogonality penalty (excited states)

```python
from qvarnet.losses import AuxiliaryLoss
from qvarnet.utils.checkpoint import load_run
import jax.numpy as jnp

class OrthogonalityPenalty(AuxiliaryLoss):
    """Force ψ₁ ⊥ ψ₀ to target the first excited state."""
    def __init__(self, gs_params, gs_model_apply, beta=1.0):
        self.gs_params = gs_params
        self.gs_model_apply = gs_model_apply
        self.beta = beta

    def __call__(self, params, model_apply, batch):
        log_psi1 = model_apply(params, batch).squeeze()
        log_psi0 = self.gs_model_apply(self.gs_params, batch).squeeze()
        # ⟨ψ₁|ψ₀⟩ ≈ mean exp(log|ψ₀| - log|ψ₁|)  (unnormalised)
        overlap = jnp.mean(jnp.exp(log_psi0 - log_psi1))
        return self.beta * overlap**2

# Load ground state, then train excited state
gs = load_run("./outputs/ground_state/")
penalty = OrthogonalityPenalty(gs.params, gs.coord_mode.wrap_model_apply(gs.model.apply))

result = train(
    ...,
    auxiliary_losses=(penalty,),
)
```

---

## 11. Checkpointing

### Automatic checkpointing

```python
cfg = TrainingConfig(
    checkpoint_path  = "./outputs/my_run",
    save_checkpoints = True,   # saves checkpoint.msgpack every 50 steps
)
```

A NaN in the energy always saves `nan_checkpoint.msgpack` regardless of this setting.

### Resuming from a checkpoint

Just re-run `train()` with the same `checkpoint_path`. If `checkpoint.msgpack` exists, it is loaded automatically and training resumes from `state.step`.

```python
# First run: trains 0 → 1000 epochs
result1 = train(..., training_config=TrainingConfig(n_epochs=1000, checkpoint_path="./run1"))

# Second run: resumes from 1000, trains until 2000
result2 = train(..., training_config=TrainingConfig(n_epochs=2000, checkpoint_path="./run1"))
```

### Manual checkpoint callback

```python
from qvarnet.callbacks import CheckpointCallback

# Save every 100 steps instead of the default 50
result = train(
    ...,
    training_config=TrainingConfig(save_checkpoints=False, ...),  # disable auto
    callbacks=[CheckpointCallback("./outputs/run", save_every=100)],
)
```

### Save and reload a trained run

Pass `model_name` and `model_args` to `train()` to write `run_config.json` alongside the checkpoint. This enables `load_run()`:

```python
result = train(
    ...,
    model_name="mlp",
    model_args={"architecture": [3, 64, 64, 1]},
)

# Later, in a different session:
from qvarnet.utils.checkpoint import load_run

run = load_run("./outputs/my_run")
# run.model        — reconstructed Flax model
# run.params       — trained parameters
# run.training_config
# run.coord_mode
```

`load_run` only restores params (not optimizer state). It is for inference, not resuming training.

---

## 12. Analysing results

### TrainResult

```python
result = train(...)

result.history    # list of VMCState, one per epoch
result.cm_mean    # list of float — centre-of-mass mean per epoch
result.cm_std     # list of float — centre-of-mass std per epoch

# Unpack like a tuple (backward compat)
history, cm_mean, cm_std = result
```

### Energy history

```python
import numpy as np
import matplotlib.pyplot as plt

energies = np.array([float(s.energy) for s in result.history])
stds     = np.array([float(s.std)    for s in result.history])

plt.plot(energies, label="⟨E⟩")
plt.fill_between(range(len(energies)), energies - stds, energies + stds, alpha=0.2)
plt.axhline(y=exact_E0, linestyle="--", label="exact")
plt.xlabel("Epoch"); plt.ylabel("Energy"); plt.legend()
```

### Best model selection

```python
# Single best state by energy
best_state = result.best(n=1, metric="energy")[0]
print(float(best_state.energy))

# Top-5 by lowest variance
top5 = result.best(n=5, metric="std")
```

### VMCState fields

| Field | Description |
|-------|-------------|
| `state.params` | Model parameters (pytree) |
| `state.energy` | ⟨E⟩ this epoch |
| `state.std` | σ(E_loc) this epoch |
| `state.acceptance_rate` | Per-chain MH acceptance |
| `state.step_size` | MH step size this epoch |
| `state.grads` | ∇_θ E this epoch (pytree) |
| `state.step` | Training step counter |
| `state.cm_mean` | CM mean |
| `state.cm_std` | CM std |

### Computing an observable post-training

```python
from qvarnet.utils.checkpoint import load_run
from qvarnet.probability import build_prob_fn
from qvarnet.sampling_step import sample_and_process
import jax, jax.numpy as jnp

run = load_run("./outputs/my_run")
effective_apply = run.coord_mode.wrap_model_apply(run.model.apply)
prob_fn = build_prob_fn(effective_apply)

key = jax.random.PRNGKey(0)
batch, _, _ = sample_and_process(
    key=key, prob_fn=prob_fn, prob_params=run.params,
    init_positions=jnp.zeros(sample_shape),
    step_size=0.3, n_chains=4096, dof=dof,
    n_steps=500, burn_in=100, thinning=2, PBC=40.0,
)

# batch: (n_chains, dof) — samples from |ψ|²
density = jnp.histogram(batch[:, 0], bins=100, density=True)
```

---

## 13. Custom Hamiltonian

Three steps: subclass, implement `potential_energy`, register.

```python
from flax import struct
import jax.numpy as jnp
from qvarnet.hamiltonian.continuous import ContinuousHamiltonian
from qvarnet.hamiltonian.hamiltonian_registry import register_hamiltonian

@register_hamiltonian("my-hamiltonian")
@struct.dataclass
class MyHamiltonian(ContinuousHamiltonian):
    g: float = 1.0    # any scalar parameters go here as struct fields

    def potential_energy(self, samples):
        # samples: (batch, dof) — ALWAYS lab coordinates (guaranteed by base class)
        # return:  (batch,)     — one energy value per sample
        ...
```

`potential_energy` receives **lab coordinates** regardless of which `coord_mode` is active. The transform is handled by `ContinuousHamiltonian.local_energy`. Do not call `coord_mode` inside your subclass.

**Overriding the kinetic energy** (e.g. different mass units):

```python
@register_hamiltonian("cs-model")
@struct.dataclass
class CalogeroSutherland(ContinuousHamiltonian):
    L: float = 1.0

    def kinetic_local_energy(self, params, samples, model_apply):
        # CS convention uses ℏ²/m = 1 (factor of 2 relative to default ℏ²/2m = 1)
        from qvarnet.hamiltonian.kinetic import kinetic_log
        return 2 * kinetic_log(params, samples, model_apply, self._get_laplacian_fn())

    def potential_energy(self, samples):
        ...
```

**Using the registry**:

```python
from qvarnet.hamiltonian.hamiltonian_registry import HAMILTONIAN_REGISTRY
h = HAMILTONIAN_REGISTRY["my-hamiltonian"](g=2.0)
```

See `examples/custom_hamiltonian.py` for a complete runnable example (double-well potential).

---

## 14. Custom model

Four steps: subclass `BaseModel`, implement `__call__` and `from_config`, register.

```python
import flax.linen as nn
import jax.numpy as jnp
from qvarnet.models.base import BaseModel
from qvarnet.models.registry import register_model

@register_model("my-model")
class MyModel(BaseModel):
    hidden: int = 64

    @nn.compact
    def __call__(self, x):
        # x:      (batch, dof)
        # output: (batch, 1)  — log|ψ(x)|
        x = nn.Dense(self.hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)
```

**Conventions:**
- Input `x` has shape `(batch, dof)`. Always assume a batch dimension.
- Output must have shape `(batch, 1)` or `(batch,)` — a single log-amplitude per sample.
- The `.squeeze()` call in the kinetic energy formula handles both shapes.
- Do not apply an exponential or absolute value — output log|ψ| directly.

**Using with save/load:**

```python
result = train(
    ...,
    model_name="my-model",
    model_args={"hidden": 64},   # must match from_config() kwargs
)

run = load_run("./outputs/run")
# run.model is MyModel(hidden=64)
```

See `examples/custom_model.py` for a complete runnable example (Gaussian ansatz for the harmonic oscillator, converges to α = 0.5).

---

## 15. Stochastic reconfiguration (QGT)

Stochastic reconfiguration replaces the Adam gradient with the natural gradient:

```
θ_{t+1} = θ_t − η S⁻¹(θ_t) ∇_θ E(θ_t)
```

where S_{kl} = ⟨O_k O_l⟩ − ⟨O_k⟩⟨O_l⟩, O_k = ∂log|ψ|/∂θ_k. This often converges faster and avoids ill-conditioning of the energy landscape.

### Enabling it

```python
from qvarnet.qgt import QGTConfig
from qvarnet.config.training_setup import TrainingConfig

result = train(
    ...,
    training_config=TrainingConfig(
        ...,
        use_qgt=True,
    ),
    qgt_config=QGTConfig(
        solver        = "cholesky",   # "cholesky" (default), "direct", "gmres", "diagonal"
        learning_rate = 5e-3,         # step size for the natural gradient update
        regularization= 1e-4,         # ε added to diagonal of S (use ≥1e-4 with float32)
    ),
)
```

When `use_qgt=True`, the `optimizer` (Adam/SGD) is ignored for parameter updates — only `qgt_config.learning_rate` is used.

### Solver choice

| Solver | When to use |
|--------|-------------|
| `"cholesky"` | Default — fast and stable for SPD S |
| `"direct"` | LU decomposition — when Cholesky fails |
| `"gmres"` | Large n_params (iterative, no full S needed in principle) |
| `"diagonal"` | Very large models — cheap approximation |

### Float32 warning

JAX defaults to float32. The QGT matrix S is a sample covariance and can have floating-point noise on the order of 1e-5. Use `regularization ≥ 1e-4` to guarantee S is positive definite in float32. For float64 (`jax.config.update("jax_enable_x64", True)`), 1e-6 suffices.

### Memory warning

`qvarnet` materialises the full `(n_params, n_params)` QGT matrix. This is O(n_params²) memory. For networks with > ~5000 parameters, this becomes expensive. A matrix-free CG solver (not yet implemented) would avoid materialising S entirely by computing S·v on the fly.

---

## 16. Hutchinson

The default Laplacian (`forward_ad`) computes Δ log|ψ| exactly in O(dof) JVPs — one per spatial dimension. For systems with large dof this can dominate training time. The **Hutchinson trace estimator** replaces the exact trace with a stochastic approximation:

```
Tr(H_f) ≈ (1/m) Σᵢ zᵢᵀ H_f zᵢ  =  (1/m) Σᵢ zᵢ · ∇(zᵢ · ∇f)
```

where each zᵢ is a random probe vector and each term costs exactly one JVP of grad(f) — the same as one `forward_ad` iteration. For `n_terms << dof` the cost is O(n_terms) instead of O(dof), at the cost of estimator variance. The estimator is **unbiased** regardless of `n_terms`.

### Enabling Hutchinson

Set `laplacian_method="hutchinson"` on the Hamiltonian and configure via `hutchinson_n_terms` and `hutchinson_distribution`:

```python
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian

ham = HarmonicOscillatorHamiltonian(
    laplacian_method="hutchinson",
    hutchinson_n_terms=20,           # probe vectors per sample (default: 10)
    hutchinson_distribution="rademacher",  # "rademacher" or "gaussian" (default: "rademacher")
)
```

Then train as usual — no other changes needed:

```python
result = train(
    shape=(n_chains, dof),
    model=model,
    optimizer=optimizer,
    hamiltonian=ham,   # Hutchinson configured here
    training_config=TrainingConfig(n_epochs=2000, ...),
    sampler_params=...,
)
```

A fresh PRNGKey is split from the main key each epoch and threaded down to the Laplacian, so estimates are independent across steps.

### Probe distribution

| Distribution | Variance | Notes |
|---|---|---|
| `"rademacher"` | Optimal (minimum variance) | ±1 entries, default |
| `"gaussian"` | Slightly higher | Normal N(0,1) entries |

Rademacher (±1) is variance-optimal among unit-variance distributions and is the standard choice.

### When to use Hutchinson

| Regime | Recommendation |
|---|---|
| dof ≤ ~100 | `forward_ad` — exact, O(dof), no noise |
| dof ~ 100–1000, GPU | `hutchinson` with `n_terms ~ dof/5` — vmap over probes is GPU-efficient |
| dof > 1000 | `hutchinson` with `n_terms ~ 20–50` — large variance savings |

On GPU, Hutchinson's probes are vmapped and run in parallel. Even at equal FLOP to `forward_ad` (n_terms = dof), Hutchinson can be faster because vmap exploits SIMD parallelism while `forward_ad` uses a sequential `fori_loop`.

### Variance and bias

- The estimator is **unbiased**: `E[estimate] = Tr(H_f)` exactly.
- Variance decreases as 1/n_terms.
- Variance is **per sample** — the batch mean has lower variance than any individual sample.
- For smooth wavefunctions (not near cusps), n_terms=10–20 is typically sufficient.
- If the energy fluctuations (`sigma_e`) are large compared to a run with `forward_ad`, increase `n_terms`.

### Using the estimator directly

```python
from qvarnet.hamiltonian.laplacian import laplacian_hutchinson
import jax

key = jax.random.PRNGKey(0)

def log_psi(x):
    return model.apply(params, x[None]).squeeze()

# xs: (batch, dof)
lap = laplacian_hutchinson(log_psi, xs, key, n_terms=20, distribution="rademacher")
# lap: (batch,) — stochastic Laplacian estimate per sample
```

---

## 17. Common mistakes <a name="17-common-mistakes"></a>

### Wrong output shape from the model

The model must output `(batch, 1)`, not `(batch,)` in general (though `.squeeze()` handles both). If you see shape errors in the kinetic energy or probability computation, check `model(params, jnp.ones((4, dof))).shape`.

### thermalization_steps ≥ chain_length

`SamplingConfig` validates this at construction time:

```python
# BAD
sampler_params = {"chain_length": 20, "thermalization_steps": 50}
# Raises: ValueError: thermalization_steps (50) must be < chain_length (20)

# GOOD
sampler_params = {"chain_length": 200, "thermalization_steps": 20}
```

### Python raise inside jax.jit

Shape assertions inside JIT are traced away — they run at trace time, not at runtime. All shape/validity checks in qvarnet happen at the Python level (in `train()` before the JIT boundary). Don't add `if x.shape != ...: raise` inside JIT-compiled functions.

### Forgetting model_name/model_args when you need load_run

If you don't pass `model_name` and `model_args` to `train()`, no `run_config.json` is written. `load_run()` will fail later. Always pass both if you plan to reload the run.

### Using a dict for qgt_config standalone

```python
# This works (called via train() which handles the JIT boundary):
train(..., qgt_config={"solver": "cholesky"})   # auto-converted to QGTConfig

# This may fail standalone (dict with string values inside JIT):
compute_step(..., use_qgt=True, qgt_config={"solver": "cholesky"})
# Use QGTConfig directly for standalone use:
compute_step(..., use_qgt=True, qgt_config=QGTConfig(solver="cholesky"))
```

### step_size too large or too small

If `acceptance_rate → 0`: step size too large, walkers never move. Reduce `step_size`.  
If `acceptance_rate → 1`: step size too small, walkers are correlated. Increase `step_size`.  
Target: 0.4–0.6 acceptance. Enable `is_update_step_size=True` to adapt automatically.

### Energy blows up or goes NaN

1. Lower the learning rate.
2. Check that the model outputs log|ψ| (a large negative number for unlikely configurations), not raw ψ or |ψ|².
3. For singular potentials (1/r²), enable the cusp condition.
4. Save `nan_checkpoint.msgpack` is created automatically — inspect which step it failed at.

### JacobiCoords shape

With `JacobiCoords(n_particles_physical=N, n_dim=d)`:
- `shape = (n_chains, (N-1)*d)` — the sampler has N-1 Jacobi coordinates
- The model init shape is `(n_chains, N*d)` — lab coordinates

Both are set automatically. Just make sure `model.architecture[0] == N*d`.
