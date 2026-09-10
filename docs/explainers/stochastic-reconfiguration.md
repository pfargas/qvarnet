# Stochastic reconfiguration: what the knobs mean

Stochastic reconfiguration (SR, natural gradient) replaces the Euclidean gradient
∇E with **S⁻¹∇E**, where S is the quantum geometric tensor — the metric of the
variational manifold. Steps are then measured in *distance between wavefunctions*
rather than distance between parameter vectors, which is what makes SR insensitive
to the arbitrary scaling of the parameterisation.

Turn it on with `TrainingConfig(use_qgt=True)` and configure it with `QGTConfig`.

## SR is a preconditioner, not an optimizer

This is the single most misread thing in the API. `compute_step` hands the natural
gradient S⁻¹∇E to whatever optimizer you passed; the optimizer is still the update
rule.

- `optax.sgd(η)` → classic SR: θ ← θ − η·S⁻¹∇E. This is what `sr_train` passes.
- `optax.adam(...)` → SR-preconditioned Adam. Legitimate, but see the trust-region
  caveat below.

`QGTConfig.learning_rate` does **not** override the optimizer. It is used for two
other things: `sr_train` builds `optax.sgd(learning_rate)` from it, and the trust
region derives its direction cap from it.

## The solver

`solver="auto"` picks the right formulation for the regime at trace time: **minSR**
when the parameter count P exceeds the sample count M, **cholesky** otherwise.

| solver | when |
|---|---|
| `auto` | the default; picks minsr or cholesky by regime |
| `cholesky` | S is SPD. Fails loudly if S is broken — a feature |
| `minsr` | the M×M Gram dual. Same regularised step as full SR via the push-through identity, solved in sample space instead of parameter space |
| `gmres` | iterative, for large systems |
| `diagonal` | cheap approximation |
| `direct` | LU. **Avoid** — silently returns garbage on a non-PSD S |

## Regularisation

`regularization` (default `1e-2`) is ε in a Jacobi-preconditioned (unit-diagonal)
metric, equivalent to Levenberg–Marquardt Tikhonov S + ε·diag(S).

Being per-direction scale-invariant matters: the same ε means the same thing whether
the log-derivatives O_k are O(1) (spin networks) or O(1e5) (Jastrow log terms). The
preconditioning is also what keeps float32 factorisations reliable.

## The trust region — the spike guard

Singular interactions produce heavy-tailed local energies, and a single cusp-residual
spike can blow up a plain SR step. The fix is to bound the **state change per
optimizer step** in the Fisher metric:

    the applied update is rescaled so that  sqrt(Δθᵀ S Δθ)  <=  max_state_change

Default `0.1`. The units are physical: it means the same thing at any learning rate,
because the internal direction cap is derived as `max_state_change / learning_rate`.
A raw direction cap would reintroduce exactly the units trap this avoids.

Note this is *update-norm* control. The energy estimator is untouched — nothing is
clipped in the physics. (Clipping local energies is not an option here; see
`docs/notes/`.)

**Measured (Calogero–Sutherland, N=30):** `0.1` is the validated default. `0.3`
descended about 3× faster and stayed stable on that system, at more spike risk on
harder problems.

`trust_region` is the advanced override, in *direction* units: cap sqrt(δᵀSδ) ≤ Δ
directly, so the state change per step becomes learning_rate·Δ. It takes precedence
over `max_state_change`, and it is **required** instead of `max_state_change` when
`learning_rate` is an optax schedule, since a callable cannot be divided by.

### The caveat under adaptive optimizers

The trust region caps the *direction*. That is exact state-change control only when
the optimizer is `SGD(qgt_config.learning_rate)`. Under Adam the step is rescaled
per-parameter afterwards, so the guarantee weakens — it still trims spike directions,
but not to a known bound. Keep `qgt_config.learning_rate` equal to your SGD learning
rate, or set `trust_region` explicitly.

## Do not use grad_clip_norm as the spike guard

`grad_clip_norm` clips the natural gradient by **Euclidean** global norm. It defaults
to `None` and should usually stay there.

The natural gradient legitimately has a huge Euclidean norm along flat directions of
the model — that is the entire point of the S⁻¹ preconditioning. A Euclidean clip
therefore re-throttles the step the trust region just approved, by orders of magnitude.

**Measured 2026-07-11:** with `grad_clip_norm=10`, 100% of epochs were bound at
|δ| ≈ 3e3 and the energy went flat. With it off, SR matched Adam's descent with a 3×
cleaner tail. If SR "does not descend", this is the first thing to check.

## What the diagnostics tell you

With `use_qgt=True` the metrics carry SR guard fields, so you can see which
constraint shaped each step:

- `trust_scale < 1` → the Fisher trust region bound the step.
- `nat_grad_norm` above `grad_clip_norm` → the Euclidean clip bound it (see above).

`analysis/qgt_spectrum.py` gives the spectrum of S itself: `d_eff` (eigenvalues above
a relative threshold) and `d_part` (participation ratio) say how many directions of
the manifold are actually being used.

## Where the code is

| what | where |
|---|---|
| `QGTConfig`, solvers, minSR, trust region | `optim/qgt.py` |
| applying it in the training step | `vmc/training_step.py` |
| the validated recipe | `recipes.py::sr_train` |
| spectrum diagnostics | `analysis/qgt_spectrum.py` |
| the raw investigation log | `docs/notes/sr-stabilisation-log.md` |
