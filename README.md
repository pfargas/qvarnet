# qvarnet

Variational Monte Carlo with neural-network wavefunctions, in JAX.

Optimise a log-amplitude ansatz log|ψ_θ(x)| against a Hamiltonian, sampling |ψ|² by
Metropolis-Hastings. Everything runs on the accelerator: sampling, the local energy,
and the gradient step are one compiled graph per epoch.

```python
import optax
from qvarnet import VMC, TrainingConfig, SamplingConfig
from qvarnet.ansatz.compose import LogWavefunction
from qvarnet.ansatz.envelopes import GaussianEnvelope
from qvarnet.ansatz.mlp import MLP
from qvarnet.physics.hamiltonian.continuous import HarmonicOscillatorHamiltonian

psi = LogWavefunction(
    network=MLP(hidden=[64, 64], output_dim=1),
    envelope=GaussianEnvelope(),
)

result = VMC(
    shape=(4096, 5),                       # (n_chains, n_particles * n_dim)
    model=psi,
    optimizer=optax.adam(1e-3),
    hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
    training_config=TrainingConfig(n_epochs=2_000),
    sampler_params=SamplingConfig(
        step_size=0.5, chain_length=21,
        thermalization_steps=20, thinning_factor=1,
    ),
).run()

result.summary()          # energies, error bars, wall time
result.diagnose()         # the three-referee convergence verdict
```

## Install

Needs Python ≥ 3.11. With [uv](https://docs.astral.sh/uv/):

```bash
git clone <this repo> && cd qvarnet
uv sync
uv run python examples/custom_model.py     # ~10s on CPU; should print E ≈ 0.5
```

The default dependency set pulls `jax[cuda12]`. For CPU-only work, set
`JAX_PLATFORMS=cpu`.

## Extending it

You do not edit the library to add a Hamiltonian, an ansatz, a sampler, a loss or a
callback. You write a class and pass the instance — the way you pass an optax
optimizer. There is no registry and no name to invent.

A sampler that keeps 1-D walkers in the ordered sector, for hard rods:

```python
from dataclasses import dataclass
import jax.numpy as jnp
from qvarnet import Metropolis, VMC

@dataclass(frozen=True)
class OrderedSampler(Metropolis):
    def constrain(self, x):
        return jnp.sort(x)

result = VMC(..., sampler=OrderedSampler()).run()
```

That is the complete implementation. `constrain` is applied when a chain starts and to
every proposal after the periodic wrap, so the ansatz is never evaluated outside the
domain. See [`examples/`](examples/) for the same pattern applied to a Hamiltonian and
an ansatz.

## What's in it

| | |
|---|---|
| **Ansatze** | MLP, DeepSet (permutation-invariant), Jastrow factors, Gaussian/polynomial envelopes, Slater-determinant fermionic models, composable via `LogWavefunction` |
| **Hamiltonians** | harmonic oscillator, nearest-neighbour, Calogero-Sutherland, lattice Bose, penetrable sphere; open or periodic; mass-imbalanced species |
| **Kinetic energy** | exact forward-AD, [folx](https://github.com/microsoft/folx) forward-Laplacian (fastest at large N), Hutchinson, finite difference |
| **Optimisation** | any optax optimizer, plus stochastic reconfiguration (QGT) with minSR, a Fisher trust region and auto solver selection |
| **Sampling** | Metropolis-Hastings with Gaussian/uniform/particle-subset/DoF-subset proposals, periodic boxes, constrained samplers |
| **Coordinates** | lab, or Jacobi (centre of mass removed from the sampler) |
| **Diagnostics** | Geweke, Heidelberger-Welch, split-R̂, a three-referee convergence verdict, V-score, IAT/ESS, gradient SNR, QGT spectrum |
| **Estimators** | density, pair correlation, structure factor, one-body density matrix, condensate fraction, with blocking errors |

## Learning it

[`notebooks/`](notebooks/) is a guided tour, committed with outputs so you can read it
without running anything:

| | |
|---|---|
| [`01-quickstart`](notebooks/01-quickstart.ipynb) | one full run, and what to read in the trace |
| [`02-custom-sampler`](notebooks/02-custom-sampler.ipynb) | writing your own sampler — the 1-D hard-rod ordered sector in one method |
| [`03-estimators`](notebooks/03-estimators.ipynb) | measuring density, g(r), S(k) and error bars after training |
| [`04-diagnostics-and-sr`](notebooks/04-diagnostics-and-sr.ipynb) | the convergence referees, and SR vs Adam |

[`examples/`](examples/) has the same extension patterns as standalone scripts.

## Documentation

[`docs/`](docs/) is organised by the question you are asking:

- **"`diagnose()` printed something I can't read"** →
  [convergence-diagnostics.md](docs/explainers/convergence-diagnostics.md)
- **"SR isn't descending"** →
  [stochastic-reconfiguration.md](docs/explainers/stochastic-reconfiguration.md)
- **"which proposal at large N?"** → [samplers.md](docs/explainers/samplers.md)
- **"my periodic run looks wrong"** →
  [periodic-systems.md](docs/explainers/periodic-systems.md)
- **why a design is the way it is** → [docs/adr/](docs/adr/)

## Architecture

The package is a stack, and imports only ever point downward:

```
core        primitives with no qvarnet dependencies
physics     Hamiltonians, boundaries, particle species
ansatz      wavefunctions and their building blocks
sampling    proposals and samplers
optim       QGT / stochastic reconfiguration, TDVP, auxiliary losses
analysis    convergence diagnostics and property estimators
callbacks   hooks into the training loop
vmc         the driver
```

A run is four phases, one module each — `setup` builds the context, `warmup`
equilibrates the walkers, `step` builds the jitted per-epoch update, `loop` runs the
epochs. `VMC.run()` is their order and little else.

The invariants are enforced rather than hoped for:

```bash
uv run python scripts/check_layers.py    # no upward or sideways imports
uv run python scripts/check_docs.py      # prose stays out of the code
uv run pytest tests/test_golden_trace.py # bit-identical energy traces
uv run pytest tests/test_no_retrace.py   # the epoch update compiles once
uv run python scripts/bench_epoch.py     # wall-time per epoch vs baseline
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: extend from outside, keep
imports pointing downward, keep rationale out of docstrings, and never change the
numerics by accident — `tests/test_golden_trace.py` pins fixed-seed traces exactly.
