# THE ULTIMATE GUIDE — `calogero_sutherland.ipynb` × qvarnet VMC

A complete, line-referenced walkthrough of everything the notebook touches: the physics,
every dataclass, every method, the exact per-epoch flow, the snapshot/selection machinery,
and the frozen-parameter evaluation. Structure mirrors the notebook sections. All file
references are relative to `qvarnet/src/qvarnet/` unless stated otherwise.

---

## 0. Executive summary — direct answers to the three questions

**Q1 — "the jastrow kind uses 2 parameters. How? What are they? Can I trust the count?"**
Yes, and yes. The `'jastrow'` ansatz is `LogWavefunction(NoNetwork + GaussianEnvelope + LogJastrow)`.
`NoNetwork` has **zero** parameters (it returns a constant, `self.param` is never called), the
envelope has one scalar $\alpha$, the Jastrow has one scalar $\lambda$. Verified by direct init:

```
{'params': {'envelope': {'alpha': ()}, 'jastrow': {'lambda': ()}}}   → 2 parameters
```

The count printed in the summary is `ravel_pytree(final_params)[0].size`
(`vmc/train_result.py:136-137`) — it flattens *every* array leaf of the actual trained
pytree and counts elements. It cannot over- or under-count. The 899 of `mlp_jastrow` also
checks out: $5{\cdot}128 + 128$ (Dense 5→128) $+\,128 + 1$ (Dense 128→1) $+\,1$ ($\alpha$) $+\,1$ ($\lambda$) $= 899$.
See §3.3 for what the two parameters *mean* and the exact values they should converge to.

**Q2 — "`sample_factor` should be 1? What is `**ctx_i`?"**
`sample_factor` is a *measurement-budget knob*, not a correctness switch: the evaluation runs
`ceil(sample_factor × n_training_epochs)` sampling epochs at frozen parameters
(`vmc/evaluate.py:197`). `1.0` is only the default ("as many sampling epochs as you trained
for"); `0.2` with 5 000 training epochs means 1 000 eval epochs → ~2×10⁶ samples, error bar
∝ $1/\sqrt{\texttt{sample\_factor}}$. Any positive value is statistically valid; only the error
bar changes. `**ctx_i` is plain Python dict-unpacking: `ctx` is
`dict(model=…, hamiltonian=…, shape=…, sampling_config=…)` built at the end of `run_cs`, and
`evaluate_result(res_i, **ctx_i, sample_factor=0.2)` is identical to spelling out those four
keyword arguments. Full anatomy in §6.

**Q3 — "selecting the 3 best states by std — why do I get a σ = 3.56 state when the last
epoch is already better?"**
Because you are comparing **two different estimators of σ**, and there is also a real
**off-by-one bug** in the snapshot callback:

1. The snapshot policy sees *every* epoch, including the last. If the last epoch's
   *training-time* σ had genuinely been lower than every kept snapshot's metric, it **would
   have been kept** (`callbacks/snapshot.py:58`). So when a comparison looks contradictory,
   the two numbers are not the same quantity.
2. The training σ_E of one epoch is a single-batch estimate (2 048–4 096 samples) of a
   **heavy-tailed** variable (E_loc diverges like $1/x_{ij}^2$ near particle coincidences when
   the cusp isn't exact). Taking the min over thousands of such noisy estimates selects
   downward fluctuations (selection bias). The σ printed by `evaluate` is re-measured at
   frozen parameters over ~10⁶ samples — that 3.56 is the *honest* value; the training-time
   metric that won the selection was optimistically low by luck.
3. **Bug (off-by-one):** the snapshot stores the parameters *after* the gradient update of
   the epoch whose metric it records — metric of $\theta_t$, parameters $\theta_{t+1}$
   (§5.3). Near convergence this is mild; early in training it can pair a good metric with
   substantially different parameters.
4. σ_E is the **per-sample spread** of the local energy, not an error bar. It does *not*
   shrink with more sampling (only an exact eigenstate has σ = 0). Never compare it with
   `error` / `error_of_mean`, which do shrink like $1/\sqrt{M}$.

All bugs and pitfalls found are collected in §7.

---

## 1. The physics and its conventions (Section 0 of the notebook)

### 1.1 The Hamiltonian

The code convention sets $\hbar^2/m = 1$ (kinetic $-\partial^2$, **not** $-\tfrac12\partial^2$)
and $\omega = 1$:

$$
H \;=\; -\sum_{i=1}^{N} \frac{\partial^2}{\partial x_i^2}
\;+\; \sum_{i=1}^{N} x_i^2
\;+\; 2L(L-1)\sum_{i<j}\frac{1}{(x_i - x_j)^2}.
$$

This is exactly $2\times$ the "textbook" Calogero Hamiltonian
$H' = -\tfrac12\sum\partial_i^2 + \tfrac12\sum x_i^2 + \lambda(\lambda-1)\sum_{i<j}x_{ij}^{-2}$
with $\lambda = L$. Both factors of 2 are implemented explicitly in
`hamiltonian/continuous.py`:

- **kinetic:** `kinetic_local_energy` returns `2 * self._kinetic(...)` (line 170), where
  `_kinetic` computes the standard $-\tfrac12(\ldots)$ operator (§4.4);
- **interaction:** `potential_energy` returns `2 * interaction + trap` (line 198), because
  the pairwise sum is built with $L(L-1)/x_{ij}^2$ and the convention wants $g = 2L(L-1)$.

The trap is `omega_trap**2 * Σ x²` with `omega_trap = 1.0` — consistent with $+\sum x_i^2$.

### 1.2 Exact ground state and energy

$$
\psi_0(x) = C \prod_{i<j} |x_i - x_j|^{L} \, e^{-\frac12\sum_i x_i^2},
\qquad
\log|\psi_0| = L\sum_{i<j}\log|x_i-x_j| \;-\; \tfrac12\sum_i x_i^2 .
$$

Textbook energy $E_0' = \tfrac{N}{2} + \tfrac{\lambda N(N-1)}{2}$; doubling for the code
convention:

$$
E_0 = N\bigl(1 + L(N-1)\bigr), \qquad E_0/N = 1 + L(N-1).
$$

For $N=5,\ L=0.8$: $E_0 = 21$, $E_0/N = 4.2$. ✓ (matches the notebook printout).

### 1.3 The coupling regimes

$g = 2L(L-1)$: repulsive for $L>1$, attractive for $0<L<1$ (the notebook's $L=0.8$ gives
$g=-0.32$), free at $L\in\{0,1\}$. For $L<1$ the wavefunction still *vanishes* at
coincidence ($|x_{ij}|^{0.8}$) but with a weaker cusp; walkers approach coincidences more
often and $E_\text{loc}$ has heavier tails — this is why $L<1$ is flagged as "numerically
harder" and why σ estimates fluctuate so much (relevant for Q3, §5.4).

### 1.4 The `EPS` softening

`CalogeroSutherlandHamiltonian(L, epsilon=1e-12)` replaces
$1/x_{ij}^2 \to 1/(x_{ij}^2 + \varepsilon)$ (`continuous.py:196`). This regularises the
potential *only*; the ansatz's own regularisers (`+1e-10` inside `LogJastrow`'s log,
`+1e-6` in the analytic model) are independent. Consequence: even the "exact" ansatz with
$\lambda = L$ is **not** an exact eigenstate of the softened Hamiltonian — σ_E is small but
strictly nonzero, concentrated at small $x_{ij}$.

---

## 2. The composable ansatz (Section 1 of the notebook)

### 2.1 `LogWavefunction` (`models/compose.py:9`)

Every qvarnet model outputs $\log|\psi(x)|$ (mandatory — the kinetic operator and the
sampler both assume log-space). `LogWavefunction` is a sum of up to three log-space terms:

$$
\log|\psi_\theta(x)| \;=\;
\underbrace{\text{network}(\text{transform}(x))}_{\text{flexible correction}}
\;+\;
\underbrace{\text{envelope}(x)}_{\text{confinement}}
\;+\;
\underbrace{\text{jastrow}(x)}_{\text{pair correlations / cusp}}
$$

Mechanics (`compose.py:49-78`):
1. `x_enc = transform(x)` — here `NoBoundary()`, the identity (`boundaries.py:36-61`).
   (`PeriodicBoundary` would map each coordinate to interleaved $(\sin,\cos)$ features.)
2. If `n_particles` is set, reshape flat `(..., N·d)` → `(..., N, d)` for set networks
   (DeepSet). The notebook leaves it `None` → the MLP sees the flat vector.
3. `log_psi = network(x_for_net)`, expected shape `(..., 1)`.
4. Envelope and Jastrow are added **on the raw pre-transform coordinates** — deliberate,
   so their physics lives in real space even under a periodic encoding.
5. Returns `(..., 1)`.

⚠ The "each component outputs `(..., 1)`" contract is *not validated*. A network returning
`(..., N)` would silently broadcast $\log|\psi|$ into a vector — this is exactly why the
notebook's `NoNetwork` returns `jnp.zeros((*x.shape[:-1], 1))` rather than e.g. `0.0` or `x`
(its docstring documents this trap).

### 2.2 The components used by the notebook

**`GaussianEnvelope`** (`models/envelopes.py:21-32`) — one learnable scalar `alpha`,
initialised to `init = 0.1`:

$$
\text{envelope}(x) = -\alpha^2 \sum_i x_i^2 .
$$

The $\alpha^2$ parametrisation guarantees a *negative-definite* quadratic form for any real
$\alpha$, i.e. normalizability by construction (the sign of $\alpha$ is a gauge — $\pm\alpha$
are equivalent, so the landscape has a reflection symmetry with a saddle at $\alpha=0$).

**`LogJastrow`** (`models/jastrow.py:5-46`) — one learnable scalar `lambda`, initialised to
`lambda_init` (notebook: 1.2):

$$
\text{jastrow}(x) = \lambda \sum_{i<j} \log\bigl(|x_i - x_j| + 10^{-10}\bigr)
\quad\text{(open boundary, } L_{\text{box}}=\texttt{None}\text{)} .
$$

Implementation detail: it builds the full antisymmetric difference matrix
`dx[..., i, j] = x_i − x_j`, takes `|dx|`, then applies a **static** upper-triangle boolean
mask `jnp.triu(ones((n_particles, n_particles)), k=1)` and sums. The mask size is fixed by
the constructor argument `n_particles` — see the global-`N` trap in §7.2. The `1e-10`
softens $\log 0 = -\infty$ at exact coincidence (measure-zero for the sampler, but reachable
in float arithmetic).

**`MLP`** (`models/mlp.py`, used by `mlp_jastrow`/`mlp` kinds) — `hidden=[128]`,
`output_dim=1`, tanh activations, lazy input width (Flax infers `5` on first call). Note it
acts on the flat coordinate vector, so it is **not** permutation-invariant: the composed
`mlp_jastrow` ansatz is not exactly bosonic-symmetric (envelope and Jastrow are; the MLP
correction is not). Part of the variational bias discussed in §6.6.

**`CalogeroSutherlandAnalyticModel`** (`models/analytic.py:12-45`, `kind='analytic'`) — the
exact functional form with a single learnable $\lambda$ (`lam`):

$$
\log|\psi| = \lambda\sum_{i<j}\log\bigl(|x_i-x_j| + 10^{-6}\bigr) - \tfrac12\sum_i x_i^2 ,
$$

$\omega$ hard-frozen at 1 (the `omega` parameter is commented out). One parameter total.

### 2.3 Parameter inventory per `kind` — the Q1 answer in full

| kind | network | envelope | jastrow | total params | parameter names |
|---|---|---|---|---|---|
| `analytic` | — | — | — | **1** | `lam` |
| `jastrow` | `NoNetwork` (0) | 1 | 1 | **2** | `envelope/alpha`, `jastrow/lambda` |
| `mlp` | MLP (897) | 1 | — | **898** | `network/CustomDense_{0,1}/{kernel,bias}`, `envelope/alpha` |
| `mlp_jastrow` | MLP (897) | 1 | 1 | **899** | all of the above + `jastrow/lambda` |

MLP with input 5, `hidden=[128]`: kernel $5{\times}128=640$, bias $128$, kernel
$128{\times}1=128$, bias $1$ → $897$.

**What the two `jastrow`-kind parameters should converge to.** Comparing the ansatz
$\exp(\lambda\sum\log|x_{ij}| - \alpha^2\sum x_i^2)$ with $\psi_0$:

$$
\lambda^\star = L = 0.8, \qquad (\alpha^\star)^2 = \tfrac12 \;\Rightarrow\; \alpha^\star = \pm\tfrac{1}{\sqrt2} \approx \pm 0.7071 .
$$

Inspect them after training with
`result.best_params()['params']['jastrow']['lambda']` and `...['envelope']['alpha']`.
Note the notebook initialises $\lambda = 1.2$ (deliberately off, "should drift toward L")
and $\alpha = 0.1 \Rightarrow \alpha^2 = 0.01$ — a state **50× too broad** at init. The first
hundreds of epochs are spent tightening the envelope, which is why early energies (and σ)
are enormous.

**Can you trust the count?** Yes. It is computed on the *final trained pytree*, not from the
model definition: `ravel_pytree(self.final_params)[0].size` concatenates every leaf array of
`{'params': {...}}` into one flat vector. Scalars `()` count as 1. There is no double
counting (optimizer state lives in `VMCState.opt_state`, not in `params`) and nothing is
missed (Flax stores every `self.param` there; `NoNetwork` never calls `self.param`, hence
contributes nothing — it doesn't even get a subtree).

---

## 3. The Hamiltonian implementation (Section 1, continued)

### 3.1 Class hierarchy and JAX plumbing

`CalogeroSutherlandHamiltonian` (`hamiltonian/continuous.py:152`) ⊂ `ContinuousHamiltonian`
⊂ `BaseHamiltonian`, all `@struct.dataclass` (Flax struct). Every field is declared
`pytree_node=False` → the Hamiltonian is a **static** JIT argument; changing `L`, `epsilon`
or `laplacian_method` retriggers compilation (consistent with `static_argnames` in
`train.py:233-241`).

Fields: `L=1.0`, `epsilon=0.0`, `omega_trap=1.0`, `pairwise_impl="vectorized"`, plus the
inherited `laplacian_method="forward_ad"`, `coord_mode` (injected by `train()`),
`hutchinson_*`, `folx_sparsity_threshold`, `particles` (masses; `None` = equal).

### 3.2 Local energy

$$
E_\text{loc}(x) = \frac{H\psi(x)}{\psi(x)} = T_\text{loc}(x) + V(x)
$$

`local_energy` (`continuous.py:90-94`): computes kinetic on **sampler coordinates**, converts
samples to lab coordinates via `coord_mode.samples_to_lab` (identity for `LabCoords`), then
adds `potential_energy(lab_samples)`. Subclasses never see `coord_mode`.

### 3.3 Kinetic term — the log-domain identity

For any $\psi > 0$, writing $u = \log|\psi|$:

$$
-\frac{\nabla^2 \psi}{\psi} = -\Bigl(\nabla^2 u + |\nabla u|^2\Bigr),
$$

so the generic operator (`hamiltonian/kinetic.py`) is

$$
T_\text{loc} = -\tfrac12 \sum_k w_k\Bigl(\partial_k^2 \log|\psi| + (\partial_k \log|\psi|)^2\Bigr),
\qquad w_k = 1/m_k \ (\equiv 1\text{ here}),
$$

and CS multiplies by 2 (`kinetic_local_energy`, `continuous.py:167-170`) to realise
$-\sum_k \partial_k^2$. Two equivalent evaluation branches:

- **AD branch** (default `'forward_ad'`): $\nabla u$ via `vmap(grad)`, $\nabla^2 u$ via
  forward-over-reverse AD, $O(\text{DoF})$ JVPs — exact.
- **folx branch** (`'folx'`): the forward-Laplacian (LapNet) algorithm — value, gradient and
  Laplacian in *one* fused forward pass; exact, faster at large DoF. The notebook's "Notes"
  cell records they were validated identical to every printed digit.
- Also available: `'hutchinson'` (stochastic trace, uses the `key` threaded through
  `full_update`/`eval_step`), `'central_difference'`, `'full_hessian'` (debug).

### 3.4 Potential term

$$
V(x) = \underbrace{\omega_\text{trap}^2 \sum_i x_i^2}_{\text{trap}}
\;+\; 2\sum_{i<j} \frac{L(L-1)}{(x_i-x_j)^2 + \varepsilon} .
$$

Two implementations selected by `pairwise_impl`:
`"vectorized"` — `triu_indices` gathers all $N(N-1)/2$ pairs at once, $O(B N^2)$ memory,
best on GPU for $N \lesssim 100$; `"scan"` — a `fori_loop` over rows, $O(BN)$ memory, for
large $N$. Identical math.

---

## 4. `train()` — the exact flow (Section 2 of the notebook)

Entry: `qvarnet.vmc.train.train` (`vmc/train.py:63`). The notebook's `run_cs` calls it with
`shape=(n_chains, N)`, `optimizer=optax.adam(1e-3)`, `coord_mode=LabCoords()`, the sampler
dict, and

```python
TrainingConfig(n_epochs, rng_seed=seed, warm_walkers=True, is_update_step_size=True,
               min_step=1e-5, max_step=5.0, checkpoint_path='./checkpoints/cs')
```

### 4.1 The two frozen config dataclasses

`SamplingConfig` and `TrainingConfig` (`config/training_setup.py`) are frozen `dataclass`es
passed as `static_argnames` to `jax.jit` — every distinct field combination compiles a new
graph. `parse_sampler_params` converts the notebook's dict, applying defaults and
validation (`thermalization_steps < chain_length`, positive step, etc.). The notebook dict
resolves to:

```
step_size=0.5, chain_length=21, thermalization_steps=20, thinning_factor=1,
block_size=0 (no memory blocking), box_L=None (open boundary), sampler="mh"
```

### 4.2 Setup phase, step by step (`train.py:89-231`)

1. `coord_mode = LabCoords()`; `hamiltonian = hamiltonian.replace(coord_mode=coord_mode)`.
2. `key = PRNGKey(rng_seed)`.
3. **Init:** `params = model.init(key, jnp.ones(init_shape))` with
   `init_shape = coord_mode.model_input_shape(shape) = shape`. (Lazy Dense layers fix their
   widths here.)
4. `VMCState.create(apply_fn, params, tx=optimizer)` — a Flax TrainState (params + Adam
   moments + step counter).
5. **Silent resume:** `load_checkpoint(state, path=checkpoint_path,
   filename="checkpoint.msgpack")` — if `<checkpoint_path>/checkpoints/checkpoint.msgpack`
   exists it is loaded *without any message* and the epoch loop starts at the restored
   `state.step`. See pitfall §7.6. (`save_checkpoints` defaults to `False`, so this file is
   only written when you opt in; currently the dir holds only `final_state.msgpack`, which
   is *not* auto-loaded.)
6. `prob_fn = build_prob_fn(effective_apply)` (`vmc/probability.py`):
   $\texttt{prob\_fn}(x,\theta) = 2\log|\psi_\theta(x)| = \log|\psi|^2$ — the log-density the
   MH kernel targets.
7. PBC sanity warnings (not applicable here: open ansatz + unwrapped sampler).
8. **Walker init:** `init_positions="normal"` → `current_positions ~ 0.5·N(0, 1)`, shape
   `(n_chains, dof)`.
9. `step_size = 0.5` (a *traced* value from here on — adapting it does not retrace).
10. Cusp auxiliary loss: skipped (`training_config.cusp is None`).
11. **Callbacks assembled** in order: `NaNCallback` (stop + emergency checkpoint on NaN
    energy), auto-added `SnapshotCallback(policy="best_k", k=3, metric="std")` (because
    `select="std"`, `k_best=3` defaults and none was user-supplied — §5), and
    `RunOutputCallback(n=1, path=checkpoint_path)` (writes `history.csv` +
    `final_state.msgpack` at train end). `CheckpointCallback` only if
    `save_checkpoints=True`. `ProgressCallback` drives the tqdm postfix.
12. SIGINT handler installed: Ctrl-C sets a flag, the loop finishes the current epoch and
    exits cleanly (that's the "Signal received, stopping after current step." you saw).

### 4.3 One epoch = one `full_update` call (JIT-compiled, `train.py:242-332`)

Per epoch, with the notebook numbers ($C$ = 2 048 chains, dof = 5):

**(a) Sampling** — `sample_and_process` (`samplers/step.py:122`):
- Draws all randomness up front: `(C, 21, 6)` — 5 Gaussian proposal numbers + 1 uniform
  accept draw per step per chain.
- Runs 21 Metropolis-Hastings steps per chain, vmapped over chains
  (`samplers/kernel.py:mh_chain` → `lax.scan` of `mh_kernel_log`). One step:

$$
x' = x + s\,\eta,\;\; \eta\sim\mathcal N(0,\mathbb 1);\qquad
A(x\to x') = \min\!\bigl(1,\, e^{\,2\log|\psi(x')| - 2\log|\psi(x)|}\bigr);\qquad
\text{accept if } \log u < \log A .
$$

  The proposal is symmetric, so no Hastings correction is needed; the log-space form never
  under/overflows. The current log-prob is threaded through the scan — **one** network
  evaluation per step, not two.
- Post-processing (`step.py:195-197`): `raw_batch[:, burn_in::thinning]` with
  `burn_in=20, thinning=1` keeps `range(20, 21)` → **exactly 1 configuration per chain per
  epoch**. So `n_eff = 1` and the batch is `(2048, 5)`. `last_positions = raw_batch[:, -1]`
  (= the kept sample here) and per-chain `acceptance_rates` are returned.
- Design intent: with `warm_walkers=True` the 20 "thermalization" steps per epoch are really
  **decorrelation** steps — walkers persist across epochs (step (d)), and each epoch's
  single kept sample sits 21 MH steps after the previous epoch's.

**(b) Centre-of-mass diagnostics:** `cm = mean_i x_i` per chain; its mean/std over chains
are logged (`cm_mean`, `cm_std`) — a drift detector (the CS trap should keep
$\langle X_\text{CM}\rangle \approx 0$).

**(c) Step-size adaptation** (because `is_update_step_size=True`, `train.py:54-59`):

$$
s \leftarrow \operatorname{clip}\Bigl(s\,\bigl[1 + r\,(\overline{\text{acc}} - a^\ast)\bigr],\; s_\text{min}, s_\text{max}\Bigr),
\qquad a^\ast = 0.5,\; r = 0.1,\; s\in[10^{-5}, 5].
$$

A stochastic-approximation controller pinning acceptance at 50 % (you can see it working:
`acceptance (tail): 0.499` in the outputs). Strictly speaking, adapting $s$ from chain
history violates detailed balance ("diminishing adaptation" is the usual justification —
the factor →1 as acceptance settles); during *training* this is irrelevant (the target
distribution moves anyway), and `evaluate` uses a **fixed** step, which is exactly right.

**(d) Warm walkers:** `new_pos` (each chain's last position) becomes next epoch's start.
(`warm_walkers=False` would reset to the same initial positions every epoch — sampler
debugging only.)

**(e) Energy, gradient, update** — `compute_step` (`vmc/training_step.py:77`):

$$
E_\text{loc}(x_b) = T_\text{loc}(x_b) + V(x_b), \qquad
\bar E = \frac1M\sum_b E_\text{loc}(x_b), \qquad
\sigma_E = \operatorname{std}_b\bigl(E_\text{loc}\bigr),\; M = 2048 .
$$

The gradient uses the standard VMC estimator. From
$E(\theta) = \langle E_\text{loc}\rangle_{|\psi_\theta|^2}$:

$$
\nabla_\theta E = 2\,\Bigl\langle \bigl(E_\text{loc} - \bar E\bigr)\, \nabla_\theta \log|\psi_\theta| \Bigr\rangle,
$$

implemented as the surrogate loss (`training_step.py:64-70`)

$$
\mathcal L(\theta) = \frac{2}{M}\sum_b \operatorname{sg}\!\bigl[E_\text{loc}(x_b) - \bar E\bigr]\,\log|\psi_\theta(x_b)|
\;+\; \sum_\text{aux}\mathcal L_\text{aux},
$$

whose `jax.grad` equals the estimator ($\operatorname{sg}$ = `stop_gradient`; the baseline
$\bar E$ subtraction gives the minimal-variance control variate at zero bias, since
$\langle\nabla\log|\psi|\rangle$-weighted constants vanish in expectation). Then
`state.apply_gradients(grads)` performs the Adam update
$\theta_{t+1} = \theta_t - \text{Adam}(\nabla_\theta E)$. (`use_qgt=True` would instead
precondition with the Quantum Geometric Tensor $S^{-1}\nabla E$ — stochastic
reconfiguration — not used here.)

Also returned: `E_chain` = per-chain mean E_loc (feeds split-$\hat R$/Geweke diagnostics in
`result.diagnose()`), and the naive error of the mean

$$
\texttt{error\_of\_mean} = \frac{\sigma_E}{\sqrt{M}} \quad (\text{ignores autocorrelation}).
$$

### 4.4 The host-side loop body (`train.py:375-464`)

Per epoch, outside JIT: gradient norm and relative parameter change
$\|\theta_{t+1}-\theta_t\|/\|\theta_t\|$ are computed for logging; **one** `jax.device_get`
syncs all scalars; a `metrics` dict of 12 fields (`step, energy, std, error_of_mean,
E_chain, acceptance_rate, step_size, grad_norm, theta_ratio, cm_mean, cm_std, wall_time`) is
appended to `MetricsHistory` (a schema-free struct-of-dicts holding **no parameters** —
the fix for the old per-epoch memory bomb); then every callback's
`on_step_end(step, state, metrics)` runs — note `state` here is **already the updated**
$\theta_{t+1}$ while `metrics` describe the batch generated from $\theta_t$. This ordering
is the source of the snapshot off-by-one (§5.3).

### 4.5 End of run

`finally:` restores the SIGINT handler and fires `on_train_end` on all callbacks
(→ `history.csv`, `final_state.msgpack`). One final `device_get` of the last params →
`TrainResult(history, final_params, snapshots)`; `result.summary()` auto-prints.

### 4.6 Reading the summary block

```
epochs ran       : 5000   (0m 20.6s wall, 899 parameters)
final epoch      : E = 21.332584 ± 5.72e-02   σ_E = 2.5891
best epoch (  590) : E = 15.906246 ± 1.27e+01   σ_E = 573.1189
acceptance (tail): 0.500
best snapshot    : epoch 4991 (select metric = 2.17146; 3 kept ...)
```

- **final epoch** — last history record: sampled from $\theta_{4999}$; note
  `result.final_params` is $\theta_{5000}$, which was *never measured*.
- **best epoch** — `argmin` of the **energy** column (`train_result.py:132`), *not* the
  snapshot metric. `E = 15.9 ± 12.7` is *below* the exact 21 — impossible for a true
  expectation value (variational bound); it's a pure MC fluctuation of a wild early epoch
  (σ_E = 573!). This line is a diagnostic curiosity, not a model choice — and it is a
  perfect illustration of why min-over-noisy-history selection is dangerous (§5.4).
- **best snapshot** — the actual selection result: min of the **std** metric among kept
  snapshots (different criterion, different epoch — confusing but intentional).
- **acceptance (tail)** — mean acceptance over the last half of training.
- **899 parameters** — §2.3.

There is deliberately **no mean over the training history**: those samples were drawn from
a *moving* $|\psi_{\theta_t}|^2$; their average estimates nothing physical. Hence `evaluate`.

---

## 5. Snapshot selection — dataclasses, methods, and the Q3 deep dive

### 5.1 `SnapshotCallback` (`callbacks/snapshot.py`)

Auto-added by `train()` as `SnapshotCallback(policy="best_k", k=3, metric="std")` (from the
`select="std"`, `k_best=3` defaults of `train()` — override with
`train(..., select="e_plus_sigma")`, `select="energy"`, any metrics key, or a callable
`(metrics_dict) -> float`, lower = better; `k_best=0` disables).

Policies: `"none"`, `"every_n"`, `"all"`, `"best_k"`. The `best_k` update per epoch:

```python
value = metric(metrics)                      # e.g. float(metrics["std"])
if len(snapshots) < k or value < snapshots[-1]["metric"]:
    snapshots.append({"step": step, "metric": value,
                      "params": jax.device_get(state.params)})
    snapshots.sort(key=lambda s: s["metric"]); del snapshots[k:]
```

A rolling top-k: correct as a selection algorithm, sees every epoch including the last,
`device_get` keeps VRAM clean.

### 5.2 `TrainResult` accessors (`vmc/train_result.py`)

- `best_params()` — params of the single lowest-metric snapshot (falls back to
  `final_params` if none kept).
- `best_k(n)` — ranked list of `{"step", "metric", "params"}` dicts, best first.
- `best_k_params(n)`, `best_steps(n)` — projections of the same.
- `best(n, metric=[...])` — ranks the *history records* (no params) by `"energy"`/`"std"`/
  callables; unrelated to snapshots.
- Selection-metric factories: `e_plus_sigma_metric(α)` → $\bar E + \alpha\sigma_E$, and
  `v_score_metric(N, E_\infty)` → $N\sigma_E^2/(\bar E - E_\infty)^2$ (arXiv:2302.04919).

### 5.3 🐞 BUG — off-by-one between snapshot metric and snapshot params

Trace the epoch (train.py):

```
full_update:  batch ~ |ψ_{θ_t}|²  →  Ē_t, σ_t from that batch  →  Adam  →  θ_{t+1}
train loop:   state = new_state                     # train.py:413  (state is now θ_{t+1})
              metrics = {..σ_t..}                   # describes θ_t
              cb.on_step_end(step, state, metrics)  # train.py:463
SnapshotCallback: params = device_get(state.params) # snapshot.py:53/60  → stores θ_{t+1} (!)
```

So the snapshot labelled *"epoch 4991, σ = 2.17"* actually holds **θ₄₉₉₂** — the parameters
one Adam step *after* the ones that produced that σ. Near convergence
($\|\Delta\theta\|/\|\theta\| \sim 10^{-4}$) the damage is small; early in training (large
gradients, σ in the hundreds) the stored params can be a genuinely different state than the
metric advertises. The same ordering affects `NaNCallback` (the "emergency" checkpoint holds
the post-NaN-update params).

**Fix:** capture the pre-update state — e.g. call `on_step_end(step, state, metrics)`
*before* `state = new_state` (renaming for clarity), or pass both and let each callback pick.
If you keep the current ordering intentionally ("params to *continue* from"), the metric
label is still wrong and should be shifted.

### 5.4 Why σ-selection returns a "3.56" that looks worse than the last epoch

Layered mechanics, in decreasing importance:

1. **Different quantities.** The snapshot metric is the *training-time* single-batch σ_E
   (M ≈ 2–4 k samples, moving distribution, adapted step). The 3.56 printed by
   `evaluate`/`EvalResult.sigma` is σ_E re-measured at frozen params over ~10⁶ samples. The
   training-time "final epoch σ" you compared against is *also* a noisy single-batch draw.
   None of the three are the same estimator.
2. **Heavy tails ⇒ σ estimates are terrible.** With an imperfect cusp,
   $E_\text{loc} \sim [\lambda(\lambda-1) - L(L-1)]/x_{ij}^2$ near coincidences: the E_loc
   distribution has power-law tails, so the *sample variance* has enormous (possibly
   infinite) variance. Single-batch σ_E swings wildly epoch-to-epoch (the `jastrow`-kind run:
   4.5 → 28 between neighbouring epochs at essentially converged params).
3. **Selection bias.** $\min_t \hat\sigma_t$ over 5 000 noisy draws is biased *low* — the
   winner won partly by luck. Honest re-measurement (that's what `evaluate` is) regresses
   it back up. Expect `EvalResult.sigma > snapshot["metric"]`, systematically.
4. **The off-by-one** (§5.3): the params you evaluate are not the ones that generated the
   winning metric.
5. **σ is not an error bar.** σ_E = 3.56 with $10^6$ samples still gives an energy error of
   order $3.56 \cdot \sqrt{\tau_\text{int}}/10^3 \sim 10^{-3}$. A "3.56-σ state" can be an
   excellent measurement. Only $\sigma \to 0$ as the *ansatz* → eigenstate (zero-variance
   principle); it never decreases with more sampling.
6. **Consistency check that resolves the paradox:** if the last epoch's *training* σ had
   truly been smaller than all three kept metrics, the callback would have kept it — it
   inspects every epoch. Whenever the comparison looks violated, one of items 1–4 is in play.

**Practical recommendations.** For final-answer selection prefer `select="e_plus_sigma"`
(penalises both energy and spread, more robust than either alone) or a V-score; make the
per-epoch metric less noisy before trusting a min over it (e.g. a callable computing a
tail-robust spread, or an EMA-smoothed σ); and *always* re-rank candidates by a frozen
`evaluate` over the `best_k()` list rather than trusting training-time metrics:

```python
for i, snap in enumerate(result.best_k()):
    ev = evaluate_result(result, **ctx, snapshot_index=i, sample_factor=0.1)
    print(snap["step"], snap["metric"], ev)
```

---

## 6. Frozen-parameter evaluation (Section 4 of the notebook)

### 6.1 Why it exists

Training energies average over a moving distribution — not a physical estimator. The paper
number is: freeze $\theta$, sample $|\psi_\theta|^2$ with plain MH (no gradients, fixed step),
and report a block-averaged mean whose error bar honestly includes autocorrelation.

### 6.2 `evaluate_result(result, **ctx, sample_factor=0.2)` — exact steps (`vmc/evaluate.py:177-201`)

1. **Parameter choice:** `ranked = result.best_k()`;
   `params = ranked[snapshot_index]["params"]` (default `snapshot_index=0` → best by the
   training `select` metric — subject to §5 caveats!). Falls back to `final_params` if no
   snapshots. To measure the *final* state instead: pass the params explicitly to
   `evaluate(...)`, or run with `k_best=0`.
2. **Budget:** `n_epochs = max(1, ceil(sample_factor · len(result.history)))`. Note
   `len(history)` = epochs *actually ran* (189 if you Ctrl-C'd), not the configured number.
3. **Step size:** unless overridden, `kwargs["step_size"] = history.get("step_size")[-1]` —
   the last *adapted* training step, so the ≈50 % acceptance carries over. This is why the
   controller can be off during eval and everything stays exact.
4. Delegates to `evaluate(model, params, hamiltonian, shape, sampling_config, n_epochs=…, …)`.

`**ctx_i` (Q2): `run_cs` returns `ctx = dict(model=model, hamiltonian=hamiltonian,
shape=shape, sampling_config=parse_sampler_params(sampler_params))`; the `**` operator
splats it into keyword arguments — nothing more. It exists so the eval cell can't
accidentally measure with a different model/Hamiltonian/shape than it trained with. (Note
`hamiltonian` in `ctx` is the *pre*-`replace(coord_mode=…)` object; `evaluate` re-applies
`replace(coord_mode=…)` itself, so this is consistent.)

### 6.3 `evaluate(...)` internals (`evaluate.py:80-174`)

1. Rejects `sampler="pt"` (MH only for now). `coord_mode` defaults to `LabCoords()`.
2. **Fresh walkers:** `PRNGKey(rng_seed=0)`, positions `~ 0.5·N(0,1)` — it does *not* reuse
   the trained walker cloud, hence the burn-in below. (Also note eval's seed 0 coincides
   with the training default; harmless here since params are frozen, but for multi-seed
   studies pass distinct `rng_seed`s.)
3. `burn_in_epochs = max(1, n_epochs // 10)` unless given — for the notebook call: 1 000 eval
   epochs + 100 burn-in epochs, each epoch = 21 MH sweeps of 2 048 chains.
4. Per epoch, the same JIT'd `sample_and_process` as training (same
   `chain_length=21 / thermalization=20 / thinning=1` → 1 kept config per chain) followed by
   `hamiltonian.local_energy` — **no gradient, no update, fixed step**. Records the epoch
   mean $\bar E_e$, epoch std, mean acceptance; discards the first `burn_in` epochs.
5. Aggregation:

$$
E = \frac{1}{K}\sum_{e=1}^{K} \bar E_e , \qquad
\sigma = \frac{1}{K}\sum_e \operatorname{std}_b(E_\text{loc})_e , \qquad
n_\text{samples} = K \cdot C \cdot n_\text{eff} \;(= 1000 \cdot 2048 \cdot 1),
$$

$$
\texttt{error\_naive} = \frac{\sigma}{\sqrt{n_\text{samples}}}, \qquad
\texttt{error} = \operatorname{std}\bigl(\{\text{block means}\}\bigr)\big/\sqrt{n_\text{blocks}} .
$$

**Block averaging** (`block_error`, `evaluate.py:65-77`): split the per-epoch series
$\{\bar E_e\}_{e=1}^{K}$ into `n_blocks = 20` contiguous blocks of $K/20$ epochs (trim the
remainder; shrink `n_blocks` if $K < 2\,n_\text{blocks}$), average each block, and take
std(block means, ddof=1)/√20. If blocks are longer than the autocorrelation time
$\tau_\text{int}$, this recovers the true error $\sigma_{\bar E}\sqrt{2\tau_\text{int}}$;
for an uncorrelated series it reproduces the naive estimate. **The `error/error_naive`
ratio printed by `print(ev)` is your autocorrelation alarm** — ≈1 means decorrelated epochs
(expected here, since consecutive kept samples are 21 MH sweeps apart), ≫1 means the blocks
are too short or chains are sticky.

### 6.4 `EvalResult` fields, precisely

| field | meaning | shrinks with more sampling? |
|---|---|---|
| `energy` | mean of per-epoch means = grand mean | — |
| `error` | **the** error bar (block-averaged) | yes, ∝ $1/\sqrt{K}$ |
| `error_naive` | $\sigma/\sqrt{n_\text{samples}}$, ignores autocorrelation | yes |
| `sigma` | per-sample spread of $E_\text{loc}$ (ansatz quality!) | **no** |
| `acceptance` | mean MH acceptance | — |
| `n_epochs, n_samples, n_blocks` | bookkeeping | — |
| `energies` | the raw per-epoch series (replot, re-block) | — |

The user-facing quality figure of an *ansatz* is `sigma` (zero-variance principle); the
quality figure of a *measurement* is `error`.

### 6.5 So — should `sample_factor` be 1?

No constraint whatsoever. It only sets the measurement budget relative to training length
(a convenient unit because you already know what one epoch costs). Guidance:

- Target error bar $\epsilon$: you need $K \approx (\sigma^2\, 2\tau_\text{int})/(C\,\epsilon^2)$
  epochs — measure once with a small factor, read off `error`, and scale:
  `error ∝ 1/√(sample_factor)`.
- The notebook's `0.2` (1 000 epochs) already delivered ±0.0004 per particle. `1.0` would
  cost 5× the time for √5 ≈ 2.2× smaller error — usually pointless when the *variational
  bias* (0.058/particle there, i.e. ~144 error bars!) dominates. Improve the ansatz/training
  before burning eval samples.
- Efficiency note: the eval inherits `thermalization_steps=20`, so 20 of every 21 MH sweeps
  are discarded *within each epoch* even though params are frozen and walkers stay warm
  across epochs. That decorrelates epochs nicely (block ratio ≈ 1) but costs 21 sweeps per
  kept sample. Passing an eval-specific `sampling_config` (e.g. `chain_length=5,
  thermalization_steps=0, thinning_factor=5`, or simply `thermalization_steps=4,
  chain_length=5`) can give ~4× more samples per second; the block error then honestly
  absorbs whatever autocorrelation appears. Only the burn-in epochs truly need long
  thermalization.

### 6.6 Why 4.2578 ± 0.0004 vs exact 4.2000 — the bias is real

The error bar is honest; the 0.058/particle gap is **variational bias**, not statistics:
(i) the MLP correction is not permutation-symmetric and must *learn* the symmetry;
(ii) λ was initialised at 1.2 and Adam(1e-3) moves a near-flat direction slowly;
(iii) the evaluated params were σ-selected (§5.4) rather than energy-selected;
(iv) the regularisers (`1e-10` in the Jastrow log, `ε=1e-12` in the potential) preclude an
exact eigenstate. The `analytic` kind is the control experiment: it converges to $E_0$ and
isolates optimisation issues from expressivity issues.

---

## 7. All bugs and pitfalls found (ranked)

### 7.1 🐞 Snapshot off-by-one (library bug — `vmc/train.py:413,463` + `callbacks/snapshot.py:53,60`)
Snapshots store $\theta_{t+1}$ under the metric of $\theta_t$ (full analysis §5.3). Affects
`best_params()`, `best_k()`, `evaluate_result`, and `NaNCallback`'s emergency checkpoint.
Fix: hand the pre-update state to `on_step_end`.

### 7.2 🐞 `make_model` closes over the **global** `N` (notebook bug — cell "1. Ansatz")
`run_cs(L, N=..., ...)` takes `N` as a parameter but calls `make_model(kind, lambda_init)`,
and `make_model` builds `LogJastrow(n_particles=N)` from the **module-level** `N`. The
N-scan cells only work because `for N in N_scan:` *rebinds the global* before each call.
Calling `run_cs(N=10)` directly (or from a function/import) crashes inside JIT with
`Incompatible shapes for broadcasting: shapes=[(5, 5), (batch, 10, 10), ()]` (the static
triu mask vs. the actual pair matrix — verified). The MLP/envelope don't save you: they are
lazily shaped and would silently adapt; only the Jastrow mask is fixed. **Fix:** give
`make_model` an explicit `n_particles` argument and pass it from `run_cs`.

### 7.3 🐞 Cell 13 unpacking (notebook bug — the big N-scan cell)
`_, Ev, errv = run_cs(...)` → `ValueError: not enough values to unpack (expected 3, got 2)`
(already visible in the stored output). `run_cs` returns `(result, ctx)`. The cell predates
the `evaluate_result` refactor; it must mirror cell 11:

```python
res_i, ctx_i = run_cs(L=L, N=N, n_epochs=..., n_chains=2048)
ev_i = evaluate_result(res_i, **ctx_i, sample_factor=0.2)
rows.append((N, ev_i.energy / N, ev_i.error / N, 1 + L * (N - 1)))
```

Painful detail: the crash comes *after* the 50 000-epoch training — the exception threw away
a finished run. Also note this cell's `for N in N_scan` **permanently rebinds the global
`N`** (to 100, or wherever it stopped), corrupting any later cell that relies on it.

### 7.4 ⚠ The promised "Evaluate" cell / section 3 is missing
The Section-2 markdown says "The physical measurement happens in the *Evaluate* cell below"
and the sections jump 2 → 4. The single-run training in cell `f8812234` is never evaluated;
the only `evaluate_result` call hides inside the N-scan. Add a dedicated section-3 cell
(`ev = evaluate_result(result, **ctx, sample_factor=0.2); print(ev)`) — it is also where
the `error/error_naive` ratio and `ev.energies` trace deserve a look.

### 7.5 ⚠ Stale/misleading stored outputs
Cell 7 shows a `NameError: nn is not defined` from a session where imports hadn't run
(cell 3 does `import flax.linen as nn` — execution order artifact, not a code bug). The
stored `run_cs` outputs print `laplacian=folx` while the current source defaults to
`'forward_ad'`. Re-run top-to-bottom before trusting any stored numbers.

### 7.6 ⚠ Shared `checkpoint_path='./checkpoints/cs'` across every run
Three separate consequences: (i) `RunOutputCallback` **overwrites** `history.csv` and
`final_state.msgpack` on every `run_cs` call — the N-scan keeps only the last run's
artifacts; (ii) if you ever set `save_checkpoints=True`, `checkpoint.msgpack` appears and
every subsequent `train()` at this path **silently resumes** from it (`train.py:117-119`) —
across *different* N this fails at deserialisation; at the same N with `state.step ≥
n_epochs` the epoch loop `range(init_steps, n_epochs)` is empty and you get an instant
"0 epochs" run; (iii) an old `nan_checkpoint.msgpack` lingers as a confusing artifact. Fix:
per-run directories, e.g. `checkpoint_path=f'./checkpoints/cs_N{N}_L{L}_{kind}'`.

### 7.7 ⚠ Selection metric `"std"` on heavy-tailed single-batch estimates
Not a code bug but a statistics trap (full story §5.4): min-over-epochs of a noisy σ is
biased and can crown a lucky epoch. The same trap makes the summary's "best epoch"
(argmin energy) show impossible sub-exact energies (E = 15.9 < E₀ = 21 at σ_E = 573).
Prefer `select="e_plus_sigma"`, and re-rank `best_k()` with frozen evaluation.

### 7.8 ℹ Minor observations
- `NoNetwork`'s docstring/shape contract is correct and necessary — `LogWavefunction` never
  validates the `(..., 1)` contract, so keep returning `zeros((*batch, 1))`.
- `EvalResult.sigma` is a mean of per-epoch stds, not the pooled std — fine as a quality
  indicator, slightly below the pooled value (Jensen).
- `evaluate`'s `n_samples` (and hence `error_naive`) counts kept samples only; with
  `n_eff=1` this is exact.
- The `TrainResult.summary` "best epoch" (by energy) and "best snapshot" (by `select`)
  use different criteria by design — easy to misread as inconsistency.
- `E_exact` uses $E_0 = N(1+L(N-1))$ — verified consistent with the factor-2 convention
  audit (§1.2). The CLAUDE.md one-liner "`E_kin = Δlog|ψ| + |∇log|ψ||²`" is a sign/factor
  shorthand; the code implements $-\tfrac12\sum_k(\partial_k^2\log|\psi| +
  (\partial_k\log|\psi|)^2)$ times the CS factor 2.

---

## 8. Quick-reference: the whole pipeline on one page

```
run_cs(L, N, kind, n_epochs, n_chains, lr, lambda_init, seed, laplacian_method)
│
├─ E0 = N(1 + L(N-1))                                   # exact target
├─ model = make_model(kind)                             # §2  (⚠ global-N trap §7.2)
├─ hamiltonian = CalogeroSutherlandHamiltonian(L, ε)    # §3
├─ TrainingConfig / sampler dict                        # §4.1
├─ result = train(...)                                  # §4
│    setup: init params → VMCState → (silent ckpt resume ⚠§7.6) → prob_fn = 2·log|ψ|
│    per epoch (JIT full_update):
│      1. MH: 21 steps/chain, keep step 21 → batch (n_chains, dof)      # §4.3a
│      2. adapt step: s ← clip(s(1 + 0.1(acc − 0.5)))                   # §4.3c
│      3. E_loc = 2·T_log + V;  Ē, σ_E;  ∇E = 2⟨(E_loc−Ē)∇log|ψ|⟩; Adam # §4.3e
│      4. log 12 metrics; callbacks (snapshot best-3 by σ ⚠ off-by-one §5.3)
│    end: history.csv, final_state.msgpack, TrainResult, summary()
│
└─ ctx = dict(model, hamiltonian, shape, sampling_config)

evaluate_result(result, **ctx, sample_factor=f)          # §6
│    params  = result.best_k()[snapshot_index]["params"] # σ-selected (§5.4 caveats)
│    n_ep    = ceil(f · len(result.history));  burn-in = n_ep // 10
│    step    = last adapted training step (acceptance carries over)
│    per epoch: same MH sampling, frozen params, no gradient → Ē_e
│    E = mean(Ē_e);  error = block_error(Ē_e, 20);  sigma = mean(σ_e)
└─ print(ev)  →  "E ± error (naive ±…, ratio r) σ_E acc [samples/epochs/blocks]"
     use: error → error bar;  sigma → ansatz quality;  ratio ≫ 1 → autocorrelation alarm
```

---

*Generated 2026-07-02 from a full source review of `qvarnet/src/qvarnet` at branch
`clean-slate` (HEAD ba308ee) and `calogero-sutherland/calogero_sutherland.ipynb`. Parameter
counts and the N-mismatch crash were verified by executing the model constructors.*
