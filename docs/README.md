# qvarnet documentation

Four kinds of prose, four homes. Which one a thing belongs in is decided by *what
question it answers*, not by how long it is.

| where | answers | ships? |
|---|---|---|
| **docstrings** | what this callable is and what it takes | yes |
| **[adr/](adr/)** | *why* a design choice was made | yes |
| **[explainers/](explainers/)** | what a method is, and how to read its output | yes |
| **[notes/](notes/)** | working notes, open questions, raw investigation logs | **no** |

`scripts/check_docs.py` enforces the first row: a docstring over ~14 lines (24 for a
module) is almost always rationale or a tutorial that belongs in one of the others.

## Explainers

- [convergence-diagnostics.md](explainers/convergence-diagnostics.md) — Geweke,
  Heidelberger–Welch, split-R̂, the three-referee verdict, the V-score. **Start here
  if `result.diagnose()` printed something you could not read.**
- [autocorrelation.md](explainers/autocorrelation.md) — the autocorrelation function,
  τ_int, effective sample size, and how much to thin.
- [stochastic-reconfiguration.md](explainers/stochastic-reconfiguration.md) — SR as a
  preconditioner, solver choice, the Fisher trust region, and why `grad_clip_norm`
  usually breaks it.
- [samplers.md](explainers/samplers.md) — proposal families, why subset moves win at
  large N, and how to write a constrained sampler.
- [coordinates.md](explainers/coordinates.md) — lab vs Jacobi coordinates: what the
  sampler moves and what the ansatz sees.
- [periodic-systems.md](explainers/periodic-systems.md) — the independent PBC toggles
  and how they go wrong together.

## Decision records

- [0001-correlated-error-estimates.md](adr/0001-correlated-error-estimates.md) —
  observables should carry blocking/τ_int error bars. Accepted, not implemented.

## Notes

`notes/` is unpublished working material and is deleted before any public release.
Anything you are unsure about starts there with no ceremony, and gets promoted to
`explainers/` or `adr/` once it is settled.
