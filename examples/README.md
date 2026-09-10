# Examples

Three of these are the extension story: adding a Hamiltonian, an ansatz, or a
sampler. In every case the pattern is the same — **write a class, pass an
instance**. Nothing is registered, and no library file is edited.

| file | shows |
|---|---|
| [`custom_hamiltonian.py`](custom_hamiltonian.py) | a new Hamiltonian: subclass `ContinuousHamiltonian`, implement `potential_energy` |
| [`custom_model.py`](custom_model.py) | a new ansatz: subclass `nn.Module`, return log\|ψ\| — plus reloading a run |
| [`custom_sampler.py`](custom_sampler.py) | a new sampler: subclass `Metropolis`, override `constrain` (4 lines for 1-D hard rods) |
| [`harmonic_oscillator_deepset.py`](harmonic_oscillator_deepset.py) | a fuller run with a permutation-invariant DeepSet ansatz |

Each runs standalone:

```bash
uv run python examples/custom_sampler.py
```

They are also the fastest way to check an install is sane: `custom_model.py`
converges a one-parameter Gaussian ansatz to the exact harmonic-oscillator ground
state (E = 0.5) in a few seconds on CPU.

For the concepts behind the knobs — convergence diagnostics, stochastic
reconfiguration, coordinates, periodic boundaries — see [`../docs/`](../docs/).
