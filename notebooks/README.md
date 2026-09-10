# Notebooks

A guided tour of the API, from a first run to writing your own sampler.

All four are committed **with their outputs**, so you can read them straight through
without running anything — and so a diff shows when a change moves the numbers. They
pin `JAX_PLATFORMS=cpu` in the first cell: they are small, and it keeps them off your
GPU while real runs are going. Delete that line to use the default device.

Re-run them all after a change with:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

| notebook | what it covers | runtime |
|---|---|---|
| `01-quickstart.ipynb` | one full run: ansatz → `VMC` → `summary()` / `diagnose()` → the trace and what to read in it | ~10 s |
| `02-custom-sampler.ipynb` | writing your own sampler. The 1-D hard-rod ordered sector in one method, plus a reflecting-wall variant | ~15 s |
| `03-estimators.ipynb` | `TrainedWavefunction`: density against the analytic HO, pair correlation, S(k), and blocking error bars | ~40 s |
| `04-diagnostics-and-sr.ipynb` | the three convergence referees, SR vs Adam, the SR guard diagnostics, and early stopping | ~3 min |

## Two things they show that are worth knowing

**A run that looks converged usually is not.** In notebook 1 the energy sits within 1%
of exact and the curve is visually flat, and referee 1 still (correctly) reports it is
descending. That is the judgement the referees exist to replace.

**Referee 2 is conservative.** In notebook 4, SR reaches the exact energy to four
decimals with referees 1 and 3 passing, and referee 2 still objects — because it
measures the tail against the *best single epoch*, which is the minimum over hundreds of
noisy draws and sits ~3 error bars low as a pure order statistic. The notebook computes
that gap rather than asserting it.

## If something looks wrong

These are executed end to end in CI-like conditions, so a failure here is more likely a
real regression than a stale notebook. The library's own guards are:

```bash
uv run pytest -q
uv run pytest tests/test_golden_trace.py    # bit-identical numerics
```
