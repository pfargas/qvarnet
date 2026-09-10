# ADR-0001: observables should report correlated error bars

**Status:** accepted, not implemented. Recorded so the restructure preserves the
seams rather than designing them shut.

## Context

MCMC samples are correlated. Treating M correlated samples as M independent ones
underestimates the error of the mean by roughly sqrt(τ_int), where τ_int is the
integrated autocorrelation time. The honest estimator is either

    err = sigma_E * sqrt(tau_int / M)          (IAT-inflated)

or a blocking estimate (Flyvbjerg–Petersen), which reaches the same answer by
averaging over blocks long enough to be independent.

Three things are wrong today, all pre-existing:

1. **The energy error bar is knowingly naive.** `vmc/step.py` computes
   `error_of_mean = sigma_e / sqrt(M)`. The original code carried the comment
   *"upgraded to σ_E·sqrt(τ_int/M) once IAT lands"*. IAT landed — twice, in two
   different modules — and the upgrade never happened.

2. **It propagates into decisions.** It reaches `analysis/verdict.py` as
   `tail_error_of_mean`, which referee 2 (`at_mc_floor`) compares against, and
   `callbacks/early_stop.py` divides by it for `target_rel_err`. The two use it in
   opposite directions: a too-small error makes `at_mc_floor` harder to satisfy
   (conservative) but makes `target_rel_err` easier to satisfy (permissive).

3. **It propagates into published numbers.** The Calogero–Sutherland sweep target
   reports its per-point error as `err_total = verdict["tail_error_of_mean"]`. Every
   point of every sweep therefore carries an error bar that ignores autocorrelation.

4. **No observable carries an error bar at all.** `TrainedWavefunction.density`,
   `pair_correlation`, `structure_factor` and `obdm` all return bare values, while
   `blocking_error` and `mean_and_error` sit exported and unused.

`vmc/evaluate.py::EvalResult` is the exception and already has the right shape: it
carries `error` (blocked) beside `error_naive` (σ/√M).

## Decision

Report observables as **estimate ± error**, with the error from blocking and/or
τ_int. Move the `EvalResult` pattern onto `TrainResult` and the estimators.

Not built as part of this restructure. What the restructure does guarantee:

- **The per-chain and per-sample series survive.** `MetricsHistory` keeps `E_chain`
  (per-chain energies per epoch) alongside the scalars, and `TrainedWavefunction`
  caches its raw samples. Both are what make a correlated error computable after the
  fact. Neither may be reduced to a mean.
- **One statistics module, not five.** There were five overlapping implementations:
  two functions named `autocorr` with different algorithms, two byte-duplicate
  `blocking_error`s, and a third blocking routine in `evaluate.py`. The duplicate
  package is gone; the survivors now sit together under `analysis/`.
- **The return types leave the door open.** Estimators return `(centers, values)`
  tuples rather than bare arrays, so `(centers, Estimate[])` stays a drop-in.

The eventual shape is a small frozen `Estimate(value, error, method, n_eff)` that
formats as `1.2345 ± 0.0007`.

## Consequences

Until this is done, treat `error_of_mean` and any sweep error bar derived from it as
a lower bound on the true uncertainty. The correction factor is sqrt(τ_int), which
`analysis/mcmc.py::iat_geyer` will already compute for you from a trace.

Fixing it will change `at_mc_floor` and `target_rel_err` decisions, so
`tests/test_golden_trace.py` will need regenerating with that change recorded as
deliberate.
