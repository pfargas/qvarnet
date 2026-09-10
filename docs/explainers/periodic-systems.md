# Periodic systems: four toggles that must agree

Periodicity in a VMC run is not one switch. It is four, and they are independent by
design because there are legitimate reasons to set them separately — which also means
they can silently disagree.

| toggle | what it does | set by |
|---|---|---|
| periodic **ansatz** | log\|ψ\| is L-periodic | `PeriodicBoundary(L)` as the model's `transform` |
| periodic **sampler** | proposals folded into [0, L) | `SamplingConfig(box_L=L)` |
| periodic **potential** | pair distances use the minimum image | `BoundaryHamiltonian._min_image(dx)` |
| periodic **Jastrow** | the correlation factor is L-periodic | `LogJastrow(n_particles=N, L=L)` |

`VMC` warns on the two ansatz/sampler mismatches rather than forbidding them:

- **Periodic ansatz, unwrapped sampler.** The energy is still unbiased (|ψ|² is
  periodic, so the estimator is fine), but walkers diffuse on the covering space and
  sampled positions come out unwrapped. Fold them before computing any
  position-binned observable, or set `box_L`.
- **Wrapped sampler, non-periodic ansatz.** log\|ψ\| is discontinuous across the box
  face, so the energy is *biased* there. This one is a real error.

## The periodic Jastrow

On a ring the open-boundary form λ·Σ log\|xᵢ−xⱼ\| is not periodic. The Sutherland form

    log J = lambda * sum_{i<j} log |sin(pi (x_i - x_j) / L)|

is the exactly L-periodic analogue: invariant under xₖ → xₖ + L and smooth everywhere
except the physical coincidence cusp. A minimum-image `log|xᵢ−xⱼ|` would instead
acquire a spurious derivative kink at L/2.

## Envelopes do not belong on a ring

A confining envelope (Gaussian, quartic) breaks L-periodicity of log|ψ|: there is no
trap on a ring, and the envelope is applied to raw coordinates. `LogWavefunction`
warns if you combine `PeriodicBoundary` with an envelope. Use `envelope=None` and put
the interaction physics in a periodic Jastrow instead.
