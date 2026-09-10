# Is my run actually converged?

VMC gives you a decreasing energy curve. A decreasing curve is not convergence: it
can keep drifting downward for tens of thousands of epochs, it can flatten because
the optimiser stalled rather than because it found the ground state, and it can look
flat in the mean while individual chains sit in different places.

`qvarnet.analysis` answers the question with **three independent referees**, in
`analysis/verdict.py`. A run is called converged only when all three agree.
`result.diagnose()` runs them and prints the verdict.

---

## Referee 1 — is the trace stationary?

"Has the energy stopped going anywhere?" Two tests, both in
`analysis/stationarity.py`, because each misses a different failure.

Both deflate their effective sample count by the integrated autocorrelation time
τ_int — consecutive epochs are correlated, so *N* epochs are worth fewer than *N*
independent observations. See [autocorrelation.md](autocorrelation.md) for τ_int.

### Geweke's z — do the start and the end agree?

Take the first 10% of the trace and the last 50%. If the chain is stationary, both
segments are sampling the same distribution, so their means agree within error:

    z = (mean_early − mean_late) / sqrt(var_early/n_early + var_late/n_late)

where each `n` is deflated by that segment's own τ_int. Under stationarity `z` is
an ordinary standard normal, so |z| = O(1).

> **Read it as:** |z| ≳ 3 means the trace is still drifting. Keep training.

### Heidelberger–Welch's t — is there a slow trend?

Geweke compares two blocks, so it is blind to a slow, steady slide that is small
compared with the noise within each block. This fits a straight line to the whole
trace and asks whether the slope is significant:

    t = slope / SE(slope),  with SE inflated by sqrt(τ_int) of the residuals

> **Read it as:** |t| < 2 means no significant trend. The **sign** is informative:
> a negative slope means the energy is still improving, so stopping now leaves
> variational quality on the table.

`is_stationary()` requires both: |z| < 3 **and** |t| < 2.

---

## Referee 2 — is it at the Monte Carlo floor?

A stationary trace can still be sitting still for the wrong reason: the optimiser
may have stalled. This referee asks whether the residual wobble is *sampling* noise
rather than *optimisation* noise.

It compares the tail's mean distance from the best energy seen against the Monte
Carlo error of the mean:

    at_mc_floor  <=>  mean|E_tail − E_best|  <=  2 * error_of_mean

If the leftover fluctuation is no bigger than your error bar, the optimiser has
nothing left to extract at this sample size, and more epochs will not help — more
*samples* might.

> ⚠️ `error_of_mean` is currently the naive σ_E/√M, which ignores autocorrelation
> and is therefore too small. See
> [../adr/0001-correlated-error-estimates.md](../adr/0001-correlated-error-estimates.md).
> Until that is fixed this referee is **conservative**: it under-reports convergence.

---

## Referee 3 — did the chains mix?

The first two referees look at the mean over all chains. That mean can be perfectly
stationary while half the walkers are stuck in one region of configuration space and
half in another — a real risk for multimodal |ψ|², which is exactly what strongly
interacting or near-degenerate systems produce.

**Split-R̂** (`analysis/mcmc.py`) compares the variance *between* chains with the
variance *within* them. If they are sampling the same distribution the two agree and
R̂ → 1. If chains are trapped in different modes the between-chain variance is
inflated and R̂ climbs.

> **Read it as:** R̂ ≤ 1.1 means the chains mixed. Above that, the walkers are not
> exploring the same distribution and the error bar is meaningless.

This is the *within-run* cousin of the stronger seed-safety check, which is to run
several seeds and compare across them.

---

## The V-score — how good is this ansatz?

Separate from convergence. The V-score ([arXiv:2302.04919](https://arxiv.org/abs/2302.04919))
is dimensionless, so it is comparable across systems and particle numbers:

    V = N · Var(E_loc) / (E − E_∞)²

The local energy of an exact eigenstate is constant, so `Var(E_loc) → 0` there: a
smaller V-score means a better wavefunction. It measures **ansatz quality**, not
whether the optimiser finished. A converged run of a bad ansatz has a fine verdict
and a poor V-score.

---

## Reading a verdict

```
three-referee verdict
  1. stationary    : PASS  (|z|=1.13 < z_thr, |t|=0.84 < t_thr)
  2. at MC floor   : FAIL  (tail |E-E_best|=4.21e-03 vs err=1.10e-03)
  3. chains mixed  : PASS  (split-R̂=1.004)
  tail energy      : -8.514213
  => NOT converged
```

Referees 1 and 3 pass, so the trace is flat and the walkers are mixing — but the
residual wobble is ~4× the error bar, so the optimiser has not reached the sampling
floor. More epochs, or a better optimiser, not more samples.

The common patterns:

| 1 stationary | 2 MC floor | 3 mixed | what it means |
|---|---|---|---|
| FAIL | — | — | still descending; keep training |
| PASS | FAIL | PASS | optimiser stalled above the sampling floor |
| PASS | PASS | FAIL | walkers trapped in different modes; the mean is not trustworthy |
| PASS | PASS | PASS | converged at this sample size |

`StationarityStopper` and `EarlyStopCallback` (in `qvarnet.callbacks`) turn the
verdict into an automatic stop.

## Where the code is

| what | where |
|---|---|
| Geweke, Heidelberger–Welch, `is_stationary` | `analysis/stationarity.py` |
| τ_int (Geyer), ESS, split-R̂, autocorrelation | `analysis/mcmc.py` |
| the three-referee verdict, V-score, formatting | `analysis/verdict.py` |
| stopping on the verdict | `callbacks/stopper.py`, `callbacks/early_stop.py` |
