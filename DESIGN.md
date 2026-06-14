# QVarNet — Code Design Document

*Regenerated 2026-06-14 after the engine restructure (roadmap steps 1–7). The canonical
forward-looking plan lives in `../comments/` (`IMPORTANT_NEXT_STEPS.md`); this document
describes the code as it is now.*

## 1. Purpose

Variational Monte Carlo (VMC): minimise ⟨E⟩ = ⟨ψ|Ĥ|ψ⟩/⟨ψ|ψ⟩ ≥ E₀ over the parameters of a
neural-network wavefunction. All models output **log|ψ(x)|**. Samples are drawn from |ψ_θ|²
by Metropolis-Hastings; the gradient is the score-function estimator
∇L = 2⟨(E_loc−⟨E⟩)∇log|ψ|⟩, optionally preconditioned by the quantum geometric tensor (SR).

The engine is factored so the *method* (VMC, …) is separable from the *space* (continuum,
discrete): the same training step runs continuum HO and discrete TFIM with only the sampler and
Hamiltonian swapped (see §3, §7).

---

## 2. Module map

```
src/qvarnet/
├── vmc/                    ← method package (ground-state VMC)
│   ├── train.py            entry point: train()
│   ├── training_step.py    compute_step(), energy_and_grads()  (per-chain E, SR/minSR)
│   ├── vmc_state.py        VMCState — slim Flax TrainState (params/tx/opt_state/step)
│   ├── metrics_history.py  MetricsHistory (struct-of-arrays per-epoch; NO params)
│   ├── train_result.py     TrainResult (+ diagnose(), best(), metric factories)
│   ├── probability.py      build_prob_fn() → 2·log|ψ|
│   ├── full_sum.py         deterministic exact-sum VMC (discrete testbed)
│   ├── discrete_train.py   train_discrete() — single-spin-flip MCMC VMC
│   ├── multi_seed.py       multi_seed_run() + seed-safe R̂
│   └── masking.py          update-masking / pruning optax wrappers
├── geometry/               ← shared QGT (SR preconditioner + future TDVP metric)
│   └── qgt.py              compute_qgt, compute_natural_gradient[_minsr], solvers
├── diagnostics/            ← the three-referee suite + instrumentation (host/numpy)
│   ├── mcmc.py             Geyer IAT, ESS, split-R̂
│   ├── stationarity.py     Geweke z, Heidelberger-Welch slope
│   ├── stopper.py          StationarityStopper (Callback)
│   ├── verdict.py          three_referee_verdict, v_score, format_verdict
│   ├── gradients.py        grad norms, gradient SNR
│   ├── parameters.py       θ-step ratios, dead fraction
│   ├── qgt_spectrum.py     eigenvalues, D_eff, D_part
│   ├── compare.py          welch_t_test
│   └── dashboard.py        plot_dashboard() → PNG
├── observables/            ← post-training estimators (host/numpy, 1-D)
│   ├── base.py             blocking_error (Flyvbjerg-Petersen)
│   ├── density.py          n(x), g(r)
│   ├── structure.py        S(k)
│   └── obdm.py             one-body density matrix, natural orbitals, condensate fraction
├── samplers/               ← shared MCMC
│   ├── kernel.py           mh_kernel_log, mh_chain (continuum Gaussian moves)
│   ├── step.py             sample_and_process()
│   ├── discrete.py         spin_flip_kernel, sample_spins()
│   └── diagnostics.py      JIT IAT/ESS on raw chains
├── hamiltonian/            ← shared
│   ├── continuous.py       ContinuousHamiltonian + HO/NN-osc/Calogero/…
│   ├── discrete.py         TFIMHamiltonian (connected-elements local energy)
│   ├── kinetic.py, laplacian.py  log-space kinetic energy, Laplacian estimators
├── models/                 ← shared (MLP, DeepSet, envelopes, LogWavefunction, …)
├── config/                 ← TrainingConfig, SamplingConfig, CoordMode
├── callbacks/              ← NaN/Checkpoint/Progress/RunOutput/Snapshot
├── utils/                  ← checkpoint, exact_diag (ED reference), coord transforms
├── boundaries.py           ← NoBoundary, PeriodicBoundary
└── {train,training_step,vmc_state,train_result,probability,qgt}.py
                            ← thin backward-compat shims re-exporting the moved objects
```

---

## 3. Training loop (`vmc/train.py`)

```
train(shape, model, optimizer, hamiltonian, training_config, sampler_params, ...)
  ├─ effective_apply = coord_mode.wrap_model_apply(model.apply)
  ├─ if use_qgt: optimizer = optax.sgd(qgt_config.learning_rate)   # SR is a preconditioner
  ├─ state = VMCState.create(apply_fn, params, tx=optimizer); load_checkpoint(...)
  ├─ metrics_history = MetricsHistory()
  └─ for step in range(n_epochs):
        (new_state, key, pos, E, σ_E, E_chain, err, acc, step_size, grads, cm…) = full_update(…)
        # diagnostics computed OUTSIDE the jitted full_update (keeps its graph/energy bit-stable)
        grad_norm, theta_ratio = …
        state = new_state
        device_get(…) once  →  metrics dict  →  metrics_history.append(metrics)
        callbacks.on_step_end(step, state, metrics)
  → TrainResult(history=metrics_history)
```

`full_update` (JIT) = `sample_and_process` → optional step-size adaptation → `compute_step`
(energy + per-chain E_loc + score-function gradient → optax update; SR/minSR preconditioning
when `use_qgt`). Metrics stay on-device until **one** `device_get` per epoch (multi-GPU §10
constraint). `compute_step` is **method-agnostic**: it only needs `hamiltonian.local_energy
(params, samples, model_apply, key)` and `model_apply` — which is why `train_discrete` reuses it.

---

## 4. Key data structures

- **`VMCState`** — live optimiser state only (params, tx, opt_state, step). Per-epoch
  diagnostics no longer live here (that was a per-epoch full-copy memory bomb).
- **`MetricsHistory`** — struct-of-arrays of host scalars / small per-chain vectors
  (`energy`, `std`, `error_of_mean`, `E_chain` `(n_chains,)`, `acceptance_rate`, `step_size`,
  `grad_norm`, `theta_ratio`, `cm_*`, `wall_time`). Iterates as lightweight records
  (`[s.energy for s in history]`) and stacks via `history.get(field)`.
- **`TrainResult`** — wraps the history; `.diagnose()` prints the three-referee verdict,
  `.best(metric=…)` ranks epochs (metric factories: `e_plus_sigma_metric`, `v_score_metric`).
  Parameter retention is the explicit job of `SnapshotCallback` (policy none/every_n/all/best_k).

---

## 5. Geometry / optimisation

SR is applied as a **preconditioner** feeding optax (`apply_gradients(S⁻¹∇L)`), so `state.step`
advances and optax LR schedules work on both paths. Solvers: `cholesky`/`direct`/`gmres`/
`diagonal`, plus **`minsr`** — the M×M Gram dual `δ = (2/M)Ōᵀ(ŌŌᵀ/M+εI)⁻¹e`, the same step as
full SR (exact identity) but cheaper when P > M. The QGT lives in `geometry/` because it is also
the future TDVP metric (t-VMC, roadmap step 9).

---

## 6. Diagnostics, observables, strategies

- **Three-referee verdict** (`diagnostics/`): a run is "done" iff stationary (Geweke +
  Heidelberger-Welch, τ_int-deflated), at the Monte-Carlo floor, and chains-mixed (split-R̂).
  `StationarityStopper` early-stops; `multi_seed_run` adds the seed-safe R̂ ≤ 1.1 referee.
- **Observables** (`observables/`): blocking error bars; n(x), g(r), S(k), OBDM / condensate
  fraction — validated against the analytic non-interacting HO.
- **Strategies**: snapshot policies, best-k/V-score selection, update-masking, Welch t-test for
  honest A/B comparison.

---

## 7. Discrete testbed (the factorization proof)

`hamiltonian/discrete.TFIMHamiltonian` + `samplers/discrete` + `utils/exact_diag` give a
known-E₀ system (Lanczos/ED). `vmc/full_sum.train_full_sum` (deterministic, exact sum over 2^N)
and `vmc/discrete_train.train_discrete` (single-spin-flip MCMC) both reach the ED ground state
through the **same** `compute_step`/SR/MetricsHistory used for the continuum — demonstrating the
method/space separation that DMC/PIGS/t-VMC will reuse.

---

## 8. Testing & reproducibility

`tests/` validates: QGT step-counting, MetricsHistory/per-chain energies, import compat,
minSR ≡ full-SR, discrete ED convergence, the diagnostics estimators against AR(1) with known
τ=(1+φ)/(1-φ), observables vs analytic HO, and the strategies. `scripts/baseline_ho.py` is a
CPU-deterministic regression guard for the continuum energy trace. Run: `uv run pytest`,
`uv run ruff check src/`.

Status vs roadmap `../comments/IMPORTANT_NEXT_STEPS.md` §11: steps 1–6 done, step 7 (minSR +
dashboard + this doc) done; steps 8 (multi-GPU) and 9 (t-VMC) pending.
