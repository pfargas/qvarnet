# Analysis methodology: from one VMC run to the thermodynamic-limit equation of state

How the SS10/SS5 results are produced and reduced, end to end — the fit, seed handling, error
bars, units, and the caveats. Code references: `point.run_point`, `analysis.aggregate_seeds`,
`analysis.extrapolate_thermodynamic` / `_wls_intercept`, `vmc/train.py`, `diagnostics/verdict.py`.

All energies are in **paper units** (ℏ²/2m = 1, lengths in units of the scattering length a = 1,
energies in ℏ²/2ma²); the control parameter is the gas parameter x = ρa³.

---

## A. What one VMC run produces (`point.run_point` → one DB row)

**The estimator.** Sample particle configurations `R ~ |ψ_θ(R)|²` by Metropolis–Hastings; for each,
evaluate the *local energy* `E_loc(R) = Ĥψ_θ / ψ_θ`. The mean `⟨E_loc⟩` is the variational energy —
a **rigorous upper bound** to the true ground state, `⟨E⟩ ≥ E₀`. Optimizing θ lowers it.

**Per epoch** (one gradient step) the batch has `M = n_chains · n_eff = 1024 · 1 = 1024` samples,
and the loop records:

- energy `Ē = mean(E_loc)`,
- per-sample spread `σ_E = std(E_loc)`,
- **MC error of the mean** `err = σ_E / √M` — *naive* (assumes the M samples are independent;
  does **not** yet correct for autocorrelation, `train.py:317`).

**The reported number is a tail average.** The run trains, the energy descends and plateaus. The
reported energy is the mean of `Ē` over the **last 50% of epochs** (`tail_frac = 0.5`), and the
reported error is the mean of the per-epoch `err` over that same tail:

```
E_tail   = mean over tail of  Ē
err_tail = mean over tail of  σ_E/√M
```

(Not the best-epoch value — the `best_k` parameter snapshots saved on disk are a *separate*
artifact for re-evaluating observables; the reported energy is the tail mean of the training trace.)

**Units bridge.** The engine runs ℏ = m = 1 (kinetic −½∇²); the paper uses ℏ²/2m = 1 (kinetic
−∇²) — a factor of 2. And we want *per particle*:

```
E/N [paper] = 2 · E_tail / N        err/N = 2 · err_tail / N
```

Stored per run as `e_per_n`, `err_per_n`. (`sigma_e_per_n` = the per-sample spread σ_E ×2/N — the
*width* of the E_loc distribution, not an error bar.)

---

## B. How seeds are combined (`analysis.aggregate_seeds`)

Each seed is a fully independent run — different RNG ⇒ different MCMC *and* a different optimization
trajectory. For a fixed (x, N) with S seeds giving `{Eᵢ ± δᵢ}`:

- **Central value = the plain mean** `Ē = (1/S) Σ Eᵢ`. *Not* the minimum — even though each `Eᵢ` is
  an upper bound, the mean is the unbiased estimator; the lowest would bias the bound downward and
  throw away data.
- **Error = the larger of two things:**

  ```
  err = max(  std_i(Eᵢ)/√S ,   (1/S) Σ δᵢ  )
              ^ across-seed SEM   ^ mean within-seed MC error
  ```

  The across-seed term captures **run-to-run irreproducibility** (optimization noise — did the seeds
  actually agree?); the within-seed term is the **intrinsic MC error**. The `max` is deliberately
  conservative: tight seed agreement can't hide MC noise, and small MC errors can't hide
  optimization scatter.

Output: one `(E/N, err)` per (x, N).

---

## C. The fit — N→∞ extrapolation (`extrapolate_thermodynamic`, `_wls_intercept`)

**Physics of the model.** A finite box of N particles with periodic boundaries at *fixed density*
(fixed x) has finite-size corrections to E/N; the leading one for a homogeneous gas is `O(1/N)`. So

```
E/N (N) = E_∞ + c/N + ...
```

and `E_∞` (the thermodynamic-limit value, what lands on the paper's curve) is the value at 1/N = 0.

**The fit is a weighted linear least squares** of `y = E/N` against `t = 1/N`, with weights
`wᵢ = 1/δᵢ²` (smaller-error points pull harder). With `S_w=Σw`, `S_t=Σw·t`, `S_tt=Σw·t²`,
`S_y=Σw·y`, `S_ty=Σw·t·y`, and `D = S_w·S_tt − S_t²`:

```
E_∞ = a    = (S_tt·S_y − S_t·S_ty) / D
slope = b  = (S_w·S_ty − S_t·S_y) / D
σ(E_∞)     = sqrt(S_tt / D)            # standard WLS intercept std error
```

Only x values present at ≥ 2 different N can be fit (a line needs ≥ 2 points). With the aligned grid
you get all four N (32/64/128/256) per x; with exactly two N the line is exact and `σ(E_∞)` is pure
error propagation of those two points.

---

## D. So what *is* the error bar, precisely?

A chain: per-epoch MC error `σ_E/√M` → averaged over the converged tail → ×2/N to paper units →
combined across seeds via the conservative `max` rule → propagated through the WLS intercept. It is
a **statistical** error bar.

---

## E. Caveats (be ready for these)

1. **Naive MC error (no autocorrelation).** `err = σ_E/√M` assumes independent samples. Within an
   epoch that's *roughly* OK here because warm-walkers + `n_eff=1` (a 21-step chain, keep only the
   last sample) decorrelates each kept sample. But it ignores the integrated autocorrelation time
   τ_int; a rigorous version is `σ_E·√(τ_int/M)` (flagged in the code as a TODO). So the bars may be
   mildly **optimistic**.
2. **The tail-average error isn't reduced by the number of tail epochs.** We report the *mean
   per-epoch* error, not the error of the tail mean — conservative (epochs are autocorrelated, so
   shrinking by √(N_epochs) would be illegitimate anyway).
3. **Statistical only — no variational/systematic term.** Every energy is an *upper* bound; the gap
   to the exact energy (ansatz incompleteness) is a one-sided systematic the error bar does **not**
   capture. That is why DMC comparison matters.
4. **The fit trusts the input errors — no χ²/dof inflation.** `σ(E_∞)` propagates the per-point bars
   but is **not** rescaled by goodness-of-fit. If the four N scatter more than their bars (the 1/N
   law breaking down), the quoted `σ(E_∞)` won't grow to reflect it — you'd see it in the residuals
   of the per-x panels. (Easy to add a reduced-χ² scaling.)
5. **The 1/N form is an assumption.** If there is curvature (a 1/N² or ln N / N term), a linear fit
   over your N is slightly biased; the 4-point ladder lets you *check* linearity rather than assume.
6. **Extrapolating upper bounds.** Each (x, N) point is an upper bound, but the 1/N → 0 extrapolation
   of upper bounds is itself just an *estimate* of the thermodynamic limit — not guaranteed to remain
   an upper bound.

---

## One-line-per-stage summary

> Each run is a tail-averaged VMC energy with a naive MC error; seeds are averaged with a
> conservative `max(across-seed, within-seed)` error; the thermodynamic limit is a
> 1/N-weighted-least-squares extrapolation whose error is the fit's intercept uncertainty.

---

## Validation against the paper

The thermodynamic-limit `E_∞(x)` is benchmarked against Mazzanti–Polls–Fabrocini
(`comments/hard-soft-spheres-bosons.pdf`):

- **Table I** — hard sphere, full x-grid (the solid HS line in their Fig. 12).
- **Table III** — SS10 & SS5, but only at **x = 1e-4 and x = 1e-2**, columns EL / SR / UL / IPC / DMC.
  DMC is the exact benchmark; EL (Euler–Lagrange Jastrow) is the closest analog to this VMC.
  Scaled to E/N ÷ 4πx: SS10 ≈ 1.037 (x=1e-4), ≈ 1.12 (x=1e-2); SS5 ≈ 1.042, ≈ 1.22.

The Lee-Yang curve `E/N / 4πx = 1 + (128/15)√(x/π)` is **universal in a** only to leading orders;
the soft sphere departs from it at higher x by its (large, negative) effective range — that
deviation is the physics being measured, not an error. The corridor `[4πx, Eq31]` is a weak test at
low x (everything piles up near 4πx); the real validation is the DMC comparison.
