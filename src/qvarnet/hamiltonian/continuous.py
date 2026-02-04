import jax
from .base import BaseHamiltonian
from .hamiltonian_registry import register_hamiltonian
from .kinetic import kinetic_term
from flax import struct

import jax.numpy as jnp


class ContinuousHamiltonian(BaseHamiltonian):

    def kinetic_local_energy(self, params, samples, model_apply):
        return kinetic_term(params, samples, model_apply)

    def potential_energy(self, samples):
        raise NotImplementedError(
            "Potential energy function must be implemented by subclass."
        )

    def local_energy(self, params, samples, model_apply):
        kinetic = self.kinetic_local_energy(params, samples, model_apply)
        potential = self.potential_energy(samples)
        return kinetic.squeeze() + potential.squeeze()


@register_hamiltonian("harmonic_oscillator")
@struct.dataclass
class HarmonicOscillatorHamiltonian(ContinuousHamiltonian):
    omega: float = 1.0

    def potential_energy(self, samples):
        return 0.5 * (self.omega**2) * jnp.sum(samples**2, axis=-1)
