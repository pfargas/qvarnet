"""Minimal example: define your own sampler.

Adding a sampler takes one method. Subclass ``Metropolis`` and override
``constrain`` -- the projection onto the space your walkers are allowed to occupy.
It is applied when a chain starts and to every proposal after the periodic wrap,
so the ansatz is never evaluated outside the domain.

Everything else -- the accept/reject test, the scan over steps, the vmap over
chains, burn-in, thinning, flattening -- you inherit.

    @dataclass(frozen=True)
    class MySampler(Metropolis):
        def constrain(self, x):
            return jnp.sort(x)

Nothing is registered and no library file is touched: construct it and pass it as
``train(..., sampler=MySampler())``. Samplers are frozen dataclasses, which makes
them hashable, which is what lets JAX treat them as compile-time constants.

Run it:

    uv run python examples/custom_sampler.py
"""

import tempfile
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import optax

from qvarnet import SamplingConfig, TrainingConfig, train
from qvarnet.ansatz.compose import LogWavefunction
from qvarnet.ansatz.envelopes import GaussianEnvelope
from qvarnet.ansatz.mlp import MLP
from qvarnet.physics.boundaries import NoBoundary
from qvarnet.physics.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.sampling import Metropolis, ParticleSubsetMove


@dataclass(frozen=True)
class OrderedSampler(Metropolis):
    """Keep 1-D walkers in the ordered sector x0 < x1 < ... < x_{N-1}.

    Sorting a symmetric proposal is a valid MH move because |psi|^2 is permutation
    symmetric: the induced kernel on the ordered wedge is the symmetrised kernel,
    which is symmetric, so detailed balance holds and the Hastings term stays 0.
    """

    def constrain(self, x):
        return jnp.sort(x)


if __name__ == "__main__":
    n_particles = 5

    model = LogWavefunction(
        network=MLP(hidden=[32, 32], output_dim=1),
        transform=NoBoundary(),
        envelope=GaussianEnvelope(init=0.5),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train(
            shape=(128, n_particles),
            model=model,
            optimizer=optax.adam(3e-4),
            hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
            training_config=TrainingConfig(
                n_epochs=60, rng_seed=0, checkpoint_path=tmpdir, print_summary=False
            ),
            sampler_params=SamplingConfig(
                step_size=0.2, chain_length=21, thermalization_steps=20, thinning_factor=1
            ),
            # the whole point: your class, passed like any other object
            sampler=OrderedSampler(proposal=ParticleSubsetMove(n_move=1, n_dim=1)),
        )

    positions = np.asarray(result.final_positions)
    ordered = bool(np.all(np.diff(positions, axis=-1) >= 0))
    energies = np.asarray(result.history.get("energy"))
    print(f"epochs: {len(energies)}   E: {energies[0]:.4f} -> {energies[-1]:.4f}")
    print(f"all {positions.shape[0]} final chains still ordered: {ordered}")
    assert ordered, "the constraint leaked"
