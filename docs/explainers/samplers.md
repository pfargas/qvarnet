# Samplers and proposals

A **Sampler** owns a whole batch draw: it runs the chains, discards burn-in, thins,
and flattens. A **Proposal** owns one narrower question — how a single new
configuration is suggested. They are separate because they vary independently: any
proposal family works with any sampler.

Both are frozen dataclasses, which makes them hashable, which is what lets JAX treat
them as compile-time constants.

## Proposal families

`propose(key, position, step_size) -> (proposal, log_q_correction)`

`log_q_correction` is the Hastings term log q(x|x') − log q(x'|x). It is exactly 0
for every symmetric family below; it exists so asymmetric proposals (MALA and
friends) can plug into the same kernel without touching it.

| family | move |
|---|---|
| `GaussianMove()` | every coordinate by `step_size · N(0,1)`. The default |
| `UniformMove()` | every coordinate by `step_size · U(−1,1)` |
| `ParticleSubsetMove(n_move, n_dim)` | all `n_dim` coordinates of `n_move` randomly chosen particles |
| `DoFSubsetMove(k)` | `k` randomly chosen coordinates, particle-agnostic |

### Why subset moves at large N

A full-configuration move changes N·d coordinates at once, so the log-probability
change is a sum of N per-particle changes and acceptance decays with N at fixed step
size. Moving a few particles keeps acceptance high at a *large* step for the moved
coordinates — better mixing per model evaluation once N ≳ 30.

The trade-off is real: a subset move updates fewer coordinates per accepted step, so
mixing per *chain step* is lower. The win is acceptance at large steps. Tune
`step_size` upward when you switch.

Subset selection is uniform over subsets and independent of the current position, and
the displacement is symmetric, so the total proposal stays symmetric and the Hastings
correction stays 0.

### Coordinate layout

Particle-major: `position.reshape(n_particles, n_dim)`. The same convention as the
Jacobi transforms, the PBC Hamiltonians and the fermionic ansatze.

## Writing a sampler

Subclass `Metropolis` and override `constrain` — the projection onto the space your
walkers may occupy. It is applied when a chain starts and to each proposal after the
periodic wrap, so the ansatz is never evaluated outside the domain. Everything else
is inherited.

```python
@dataclass(frozen=True)
class OrderedMetropolis(Metropolis):
    def constrain(self, x):
        return jnp.sort(x)
```

That is the complete implementation of the 1-D hard-rod sampler. See
`examples/custom_sampler.py`.

> **The invariant that makes sorting legal:** |ψ|² is permutation symmetric, so the
> kernel induced on the ordered wedge is the symmetrised kernel, which is symmetric.
> Detailed balance holds and the Hastings term stays 0. A sampler whose `constrain`
> is *not* a symmetry of |ψ|² is not automatically a valid MH move — check before
> you write one.

## Periodic boxes

`SamplingConfig.box_L > 0` folds every proposal into [0, L). Symmetric proposals stay
symmetric on the torus, so detailed balance is unchanged. This is independent of
whether your *ansatz* is periodic — see [periodic-systems.md](periodic-systems.md).
