# QVarNet — Code Design Document

## 1. Purpose

Variational Monte Carlo (VMC): find the ground state energy of a quantum system by
optimising the parameters of a neural network wavefunction ψ_θ(x) to minimise

    ⟨E⟩ = ⟨ψ|Ĥ|ψ⟩ / ⟨ψ|ψ⟩  ≥  E₀

Samples are drawn from |ψ_θ|² via Metropolis-Hastings (MH). Gradients are
computed using the VMC estimator and backpropagated through the network.

---

## 2. Module map

```
src/qvarnet/
│
├── train.py                 ← entry point: train()
├── training_step.py         ← compute_step(), energy_and_grads()
├── sampling_step.py         ← sample_and_process()
├── vmc_state.py             ← VMCState (Flax TrainState extension)
├── probability.py           ← build_prob_fn() → prob_fn used by sampler
├── qgt.py                   ← natural gradient (QGT / stochastic reconfiguration)
│
├── config/
│   ├── coord_mode.py        ← LabCoords, JacobiCoords
│   └── training_setup.py   ← TrainingConfig, SamplingConfig
│
├── samplers/
│   └── sampler.py           ← mh_kernel, mh_kernel_log, mh_chain
│
├── hamiltonian/
│   ├── base.py              ← BaseHamiltonian (abstract)
│   ├── continuous.py        ← ContinuousHamiltonian + specific systems
│   ├── kinetic.py           ← kinetic energy functions
│   ├── laplacian.py         ← Laplacian implementations
│   └── custom_definition.py ← define_hamiltonian() factory
│
├── models/
│   ├── base.py              ← BaseModel (abstract)
│   ├── registry.py          ← @register_model decorator + MODEL_REGISTRY
│   ├── mlp.py               ← MLP
│   ├── exponential.py       ← exponential-envelope models
│   ├── deep_set.py          ← DeepSet, ExponentialDeepSet
│   ├── mlp_fermions.py      ← fermionic wavefunctions
│   └── layers/
│       ├── custom_dense.py  ← CustomDense (supports beta scaling)
│       └── input_transforms.py ← SubtractCM, AppendPairwiseDiffs
│
└── utils/
    ├── jacobi.py            ← jacobi_transformation(), from_lab_to_jacobi(), ...
    ├── coord_transforms.py  ← build_effective_apply(), init_shape_for_model()
    ├── checkpoint.py        ← save/load msgpack checkpoints
    └── ...
```

---

## 3. Top-level training loop

```
User script
│
│  from qvarnet import train, TrainingConfig, SamplingConfig, LabCoords
│
│  cfg  = TrainingConfig(n_epochs=5000, is_log_model=True, ...)
│  samp = SamplingConfig(step_size=0.5, chain_length=500, ...)
│  result = train(shape, model, optimizer, hamiltonian, cfg, samp)
│
▼
train()                                          train.py
│
├─ build_effective_apply(model.apply, coord_mode)
│   └─ wraps model.apply with coordinate pre-processing (see §6)
│
├─ build_prob_fn(effective_apply, is_log_model)
│   └─ returns prob_fn : (x, params) → |ψ(x)|² or log|ψ(x)|²
│
├─ model.init() → params (pytree)
├─ VMCState.create(apply_fn, params, optimizer) → state
├─ load_checkpoint(state, ...) → state         (if checkpoint exists)
│
└─ for step in range(n_epochs):
       │
       ▼
   full_update(state, key, current_pos,          ← @jax.jit
               prob_fn, step_size,                 static: prob_fn, hamiltonian,
               hamiltonian, sampling_config,               sampling_config,
               training_config)                            training_config
       │
       ├─ sample_and_process(...)  →  (batch, new_pos, acceptance_rate)
       │
       ├─ [optional] update step_size via acceptance rate
       │
       ├─ compute_step(state, batch, hamiltonian, ...)
       │   └─ → (new_state, E, σ_E, grads)
       │
       └─ returns (new_state, key, new_pos, E, σ_E, acc_rate,
                   step_size, grads, cm_mean, cm_std)
       │
       ▼
   state_history.append(state.replace(energy=E, std=σ_E, ...))

▼
TrainResult(history, cm_mean, cm_std)
```

---

## 4. Sampling pipeline

Goal: draw samples from |ψ_θ(x)|² using Metropolis-Hastings.

```
sample_and_process(key, prob_fn, prob_params,     sampling_step.py
                   init_positions, step_size,
                   n_chains, DoF, n_steps,
                   burn_in, thinning, PBC,
                   is_log_prob)
│
├─ draw rand_nums : (n_chains, n_steps, DoF+1)
│   ├─ [:, :, :DoF]  ← Gaussian  (proposal displacements)
│   └─ [:, :, DoF]   ← Uniform   (accept/reject threshold)
│
├─ vmap(mh_chain, in_axes=(0,None,None,None,0,None,None))
│   │   maps over chains dimension
│   │
│   └─ mh_chain(rand_values, PBC, prob_fn, prob_params,    sampler.py
│               init_position, step_size, is_log_prob)
│       │
│       └─ jax.lax.scan over n_steps:
│               │
│               ├─ is_log_prob=False → mh_kernel
│               │     proposal:    x' = x + step_size * ξ,  ξ ~ N(0,1)
│               │     accept prob: A = min(1, P(x')/P(x))
│               │
│               └─ is_log_prob=True  → mh_kernel_log
│                     proposal:    x' = x + step_size * ξ,  ξ ~ N(0,1)
│                     accept prob: log A = min(0, log P(x') - log P(x))
│
│       returns: positions (n_steps, DoF), acceptance_rate scalar
│
│   → raw_batch : (n_chains, n_steps, DoF)
│   → acceptance_rates : (n_chains,)
│
├─ thermalization: raw_batch[:, burn_in:, :]
├─ thinning:       raw_batch[:, burn_in::thinning, :]
│   → processed : (n_chains, n_effective, DoF)
│
├─ last_positions = raw_batch[:, -1, :]   → (n_chains, DoF)
│
└─ batch_flat = processed.reshape(-1, DoF)
   → (n_chains * n_effective, DoF)      ← this is the training batch
```

**Array shapes summary:**

| Variable | Shape |
|---|---|
| `init_positions` / `new_pos` | `(n_chains, DoF)` |
| `rand_nums` | `(n_chains, n_steps, DoF+1)` |
| `raw_batch` | `(n_chains, n_steps, DoF)` |
| `batch` (training input) | `(n_chains * n_effective, DoF)` |
| `acceptance_rates` | `(n_chains,)` |

---

## 5. Training step

Goal: given a batch of samples, compute energy and update parameters.

```
compute_step(state, batch, hamiltonian,           training_step.py
             is_log_model, use_qgt, qgt_config)
│
└─ energy_and_grads(hamiltonian, params, batch,
                    model_apply, is_log_model)
    │
    ├─ energy_fn(...)
    │   └─ compute_local_energy(hamiltonian, params, batch,
    │                           model_apply, is_log_model)
    │       │
    │       └─ hamiltonian.local_energy(params, batch,      continuous.py
    │                                   model_apply,
    │                                   is_log_model)
    │           │
    │           ├─ kinetic_energy(params, batch, model_apply)
    │           │   └─ Δlog|ψ| + |∇log|ψ||²   (via AD or central diff)
    │           │
    │           └─ potential_energy(batch)
    │               └─ system-specific (harmonic, NN-interaction, ...)
    │
    │       → E_loc : (batch,)
    │
    │   → E = mean(E_loc),  σ_E = std(E_loc)
    │
    ├─ def _log_psi(p):
    │       out = model_apply(p, batch).squeeze()       # (batch,)
    │       return out if is_log_model                  # already log|ψ|
    │              else log|out|                        # direct model
    │
    ├─ def loss(p):
    │       return 2 * mean( stop_grad(E_loc - E) * _log_psi(p) )
    │
    ├─ grads = jax.grad(loss)(params)
    │
    └─ (E, σ_E, grads)

        ├─ use_qgt=False → state.apply_gradients(grads)   (Adam / SGD)
        └─ use_qgt=True  → natural gradient via QGT        qgt.py
                           θ ← θ - η S⁻¹ ∇E
```

---

## 6. Coordinate transforms

The sampler always works in one fixed coordinate space. The model may need a
different representation. `build_effective_apply` wraps `model.apply` so the
rest of the code sees a single uniform interface.

```
coord_mode          sampler space      model receives
──────────────────────────────────────────────────────
LabCoords           x_lab (N*d)        x_lab           (identity)

JacobiCoords        x_jac (N*d)        x_lab ((N+1)*d)
                                        ↑
                                   inverse Jacobi:
                                   pad x_jac with zero CM coord
                                   → apply from_jacobi_to_lab()

                    n_particles_physical = N + 1
                    model init shape: (n_chains, (N+1)*d)
                    sampler shape:    (n_chains, N*d)
```

Input transforms (inside the model, as a first layer):

```
raw coords                    model layer             output
──────────────────────────────────────────────────────────────
(batch, N*d)  →  SubtractCM(n_particles, n_dim)  →  (batch, N*d)
(batch, N*d)  →  AppendPairwiseDiffs(N, d)        →  (batch, N*d + n_pairs*d)
(batch, N*d)  →  BackflowTransform(...)           →  (batch, N*d)   ← future
```

These are `nn.Module` layers stacked before the main network. They are model
architecture decisions, not training-loop concerns.

---

## 7. Config objects

```
TrainingConfig (frozen dataclass)            training_setup.py
├── n_epochs         : int
├── rng_seed         : int       = 0
├── init_positions   : str       = "normal"
├── warm_walkers     : bool      = False
├── is_update_step_size : bool   = False
├── min_step         : float     = 1e-5
├── max_step         : float     = 5.0
├── is_log_model     : bool      = False    ← model outputs log|ψ| or |ψ|
├── use_qgt          : bool      = False
├── checkpoint_path  : str       = "./"
├── save_checkpoints : bool      = False
├── target_acceptance: float     = 0.5
└── adaptation_rate  : float     = 0.1

SamplingConfig (frozen dataclass)
├── step_size             : float
├── chain_length          : int
├── thermalization_steps  : int
├── thinning_factor       : int
├── PBC                   : float
└── is_log_prob           : bool   ← mirrors is_log_model

CoordMode (frozen dataclass hierarchy)       coord_mode.py
├── LabCoords()
└── JacobiCoords(n_particles_physical, n_dim=1)
```

Both `TrainingConfig` and `SamplingConfig` are frozen and hashable, so they can
be passed as `static_argnames` to `full_update`. JAX retraces only when they
change.

---

## 8. Model architecture

```
BaseModel (abstract)                                 models/base.py
│   __call__(x) → scalar or (batch,)
│   from_config(cls, model_args) → Model
│   get_input_shape(cls, model_args, batch_size) → tuple
│
├── MLP                                              models/mlp.py
│   architecture: [in, h1, h2, ..., 1]
│   layers: CustomDense (supports beta scaling)
│
├── ExponentialMLPwithPenalty                        models/exponential.py
│   MLP output + exponential envelope exp(-α Σxⁿ)
│   Variants: log-space, Gaussian (x²) vs quartic (x⁴) envelope
│
├── DeepSet / ExponentialDeepSet                     models/deep_set.py
│   Permutation-invariant architecture:
│   φ: (n_dim,) → (internal_dim,)   per particle
│   F: (internal_dim,) → (1,)       after aggregation Σ_i φ(xᵢ)
│
└── FermionicMLP / FermionicMLP2ferms               models/mlp_fermions.py
    Slater-determinant structure for antisymmetric wavefunctions

First-layer input transforms (optional, any model):
    SubtractCM(n_particles, n_dim)      → translational invariance
    AppendPairwiseDiffs(n_particles, d) → relative coordinate features
```

---

## 9. Hamiltonian hierarchy

```
BaseHamiltonian (abstract)                           hamiltonian/base.py
│   local_energy(params, samples, model_apply,
│                is_log_model) → (batch,)
│
└── ContinuousHamiltonian                            hamiltonian/continuous.py
    │   kinetic_local_energy(params, samples, model_apply)
    │   potential_energy(samples) → (batch,)     ← abstract, system-specific
    │
    ├── HarmonicOscillatorHamiltonian
    │   V(x) = ω²/2 Σ xᵢ²
    │
    ├── NN_OscillatorHamiltonian
    │   V(x) = ω²/2 Σ xᵢ²  +  g Σ_{<ij>} (xᵢ - xⱼ)²
    │
    └── CalogeroSutherlandHamiltonian
        V(x) = λ(λ-1)/2 Σ_{i<j} 1/(xᵢ-xⱼ)²  +  trap

Kinetic energy computation:                          hamiltonian/kinetic.py
                                                     hamiltonian/laplacian.py
    is_log_model=True (recommended):
        E_kin = Δ log|ψ| + |∇ log|ψ||²
        Laplacian via:
            laplacian_autodiff_new    ← forward-mode AD (efficient, O(DoF))
            laplacian_central_difference_scan  ← finite differences (no AD)

    is_log_model=False:
        E_kin = Δψ / ψ
        via: kinetic_term (full Hessian, expensive O(DoF²))
```

---

## 10. VMCState

`VMCState` extends Flax's `TrainState` with VMC diagnostics. One instance
is stored per training step in `result.history`.

```
VMCState                                             vmc_state.py
├── step          : int          ← auto-incremented by apply_gradients()
├── apply_fn      : Callable     ← effective_apply (with coord wrapping)
├── params        : PyTree       ← neural network weights
├── tx            : GradientTransformation  ← optimizer (Adam, SGD, ...)
├── opt_state     : PyTree       ← optimizer internal state
│
├── energy        : float        ← ⟨E⟩ at this step
├── std           : float        ← σ_E at this step
├── acceptance_rate: ndarray     ← per-chain MH acceptance rate
├── step_size     : float        ← current MH step size
├── grads         : PyTree       ← ∇_θ L at this step
├── cm_mean       : float        ← mean CM position across chains
└── cm_std        : float        ← std  CM position across chains
```

Access pattern in user scripts:

```python
result = train(shape, model, optimizer, hamiltonian, cfg, samp)

energies = [s.energy for s in result.history]
acc_rates = [s.acceptance_rate.mean() for s in result.history]

# backward-compatible unpack:
history, cm_mean, cm_std = result
```
