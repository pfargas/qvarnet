from .base import BaseHamiltonian
from .hamiltonian_registry import register_hamiltonian
from .kinetic import (
    kinetic_term,
    kinetic_term_log,
    kinetic_term_divergence_theorem,
    kinetic_term_log_wavefunction,
)
from flax import struct

from .laplacian import laplacian_autodiff_new as laplacian_AD
from .laplacian import laplacian_central_difference

import jax.numpy as jnp


class ContinuousHamiltonian(BaseHamiltonian):
    """
    samples has shape (batch, DoF). The computation has to be vectorized over the batch dimension.
    """

    def kinetic_local_energy(self, params, samples, model_apply):
        return kinetic_term(params, samples, model_apply, laplacian=laplacian_AD)
        # return kinetic_term_divergence_theorem(params, samples, model_apply)

    def kinetic_local_energy_central_difference(self, params, samples, model_apply):
        return kinetic_term(
            params, samples, model_apply, laplacian=laplacian_central_difference
        )

    def kinetic_local_energy_log_model(self, params, samples, model_apply):
        return kinetic_term_log_wavefunction(params, samples, model_apply)

    def potential_energy(self, samples):
        raise NotImplementedError(
            "Potential energy function must be implemented by subclass."
        )

    def local_energy(self, params, samples, model_apply, is_log_model, use_AD=True):
        if is_log_model and use_AD:
            kinetic = self.kinetic_local_energy_log_model(params, samples, model_apply)
        elif is_log_model and not use_AD:
            raise NotImplementedError(
                "Central difference for log model not implemented yet."
            )
        elif not is_log_model and use_AD:
            kinetic = self.kinetic_local_energy(params, samples, model_apply)
        else:
            kinetic = self.kinetic_local_energy_central_difference(
                params, samples, model_apply
            )
        potential = self.potential_energy(samples)
        return kinetic.squeeze() + potential.squeeze()


@register_hamiltonian("harmonic-oscillator")
@struct.dataclass
class HarmonicOscillatorHamiltonian(ContinuousHamiltonian):
    omega: float = 1.0

    def potential_energy(self, samples):
        return 0.5 * (self.omega**2) * jnp.sum(samples**2, axis=-1)


@register_hamiltonian("nn-oscillator")
@struct.dataclass
class NN_OscillatorHamiltonian(ContinuousHamiltonian):
    omega_trap: float = 1.0
    omega_interaction: float = 1.0
    with_pbc: bool = struct.field(pytree_node=False, default=True)

    def potential_energy(self, samples):
        trap = 0.5 * (self.omega_trap**2) * jnp.sum(samples**2, axis=-1)
        if self.with_pbc:
            diffs = samples - jnp.roll(samples, shift=1, axis=-1)
        else:
            diffs = samples[:, :-1] - samples[:, 1:]
        nn_term = 0.5*self.omega_interaction**2 * jnp.sum(diffs**2, axis=-1)
        return trap + nn_term

@register_hamiltonian("soft-core")
@struct.dataclass
class SoftCoreHamiltonian(ContinuousHamiltonian):
    R: float = 1.0
    V0: float = 1.0
    def potential_energy(self, samples):
        r = jnp.linalg.norm(samples, axis=-1)
        mask = r < self.R
        return jnp.where(mask, self.V0, 0.0)


@register_hamiltonian("gross-struct-hamiltonian")
@struct.dataclass
class GrossStructHamiltonian(ContinuousHamiltonian):
    Z: int = struct.field(pytree_node=False, default=1)
    n_fermions: int = struct.field(pytree_node=False, default=1)

    def potential_energy(self, samples):
        # samples shape: (batch, n_fermions * 3)
        # Reshape to (batch, n_fermions, 3)
        pos = samples.reshape(-1, self.n_fermions, 3)

        # 1. Electron-Nuclear Attraction: -Z / |r_i|
        r_i = jnp.linalg.norm(pos, axis=-1)
        v_en = -self.Z * jnp.sum(1.0 / (r_i + 1e-12), axis=-1)

        v_ee = 0.0
        # # 2. Electron-Electron Repulsion: 1 / |r_i - r_j|
        # if self.n_fermions > 1:
        #     # Compute all pairwise differences (batch, n_fermions, n_fermions, 3)
        #     diff = pos[:, :, jnp.newaxis, :] - pos[:, jnp.newaxis, :, :]

        #     # Compute distances r_ij
        #     # Add a small epsilon or use jnp.where to avoid 1/0 on the diagonal (i=j)
        #     dist_ij = jnp.linalg.norm(diff, axis=-1)

        #     # Mask the diagonal (i=j) and sum only unique pairs i < j
        #     mask = jnp.triu(jnp.ones((self.n_fermions, self.n_fermions)), k=1)
        #     v_ee = jnp.sum(mask * (1.0 / (dist_ij + 1e-15)), axis=(-1, -2))
        # else:
        #     v_ee = 0.0

        return v_en + v_ee
