from .base import BaseHamiltonian
from .hamiltonian_registry import register_hamiltonian
from .kinetic import kinetic_log
from .laplacian import laplacian_forward_ad, laplacian_full_hessian, laplacian_central_difference
from flax import struct

import jax.numpy as jnp
from qvarnet.utils.jacobi import from_jacobi_to_lab
from qvarnet.config.coord_mode import CoordMode, LabCoords, JacobiCoords


@struct.dataclass
class ContinuousHamiltonian(BaseHamiltonian):
    """Base class for continuous-space Hamiltonians.

    Convention:
        samples: (batch, DoF)      — always in sampler coordinates (Jacobi or lab)
        potential_energy receives: (batch, DoF_lab)  — always lab coordinates
        kinetic/potential returns: (batch,)
        local_energy returns: (batch,)

    The coordinate transform (Jacobi → lab) happens here in local_energy, so
    every subclass potential_energy always receives lab coordinates.
    coord_mode is set by train() — subclasses never need to handle it.

    laplacian_method options:
        "forward_ad"          — forward-over-reverse AD, O(DoF), recommended
        "central_difference"  — finite differences, no AD required
        "full_hessian"        — full Hessian trace, O(DoF^2), debug only
    """

    laplacian_method: str = struct.field(pytree_node=False, default="forward_ad")
    coord_mode: CoordMode = struct.field(pytree_node=False, default=None)

    def _get_laplacian_fn(self):
        method = self.laplacian_method
        if method == "forward_ad":
            return laplacian_forward_ad
        if method == "central_difference":
            return laplacian_central_difference
        if method == "full_hessian":
            return laplacian_full_hessian
        raise ValueError(f"Unknown laplacian_method: {method!r}")

    def _samples_to_lab(self, samples):
        """Transform sampler coordinates to lab coordinates for potential evaluation.

        Kinetic energy always differentiates w.r.t. sampler coordinates (correct).
        Potential energy always receives lab coordinates (general, works for any V).
        """
        if self.coord_mode is None or isinstance(self.coord_mode, LabCoords):
            return samples
        if isinstance(self.coord_mode, JacobiCoords):
            n_phys = self.coord_mode.n_particles_physical
            n_d = self.coord_mode.n_dim
            zeros = jnp.zeros((*samples.shape[:-1], 1))
            u_tilde = jnp.concatenate([samples, zeros], axis=-1)
            return from_jacobi_to_lab(u_tilde, n_phys, n_d)
        raise TypeError(f"Unknown CoordMode: {type(self.coord_mode)}")

    def kinetic_local_energy(self, params, samples, model_apply):
        return kinetic_log(params, samples, model_apply, self._get_laplacian_fn())

    def potential_energy(self, samples):
        raise NotImplementedError("Subclass must implement potential_energy().")

    def local_energy(self, params, samples, model_apply):
        kinetic = self.kinetic_local_energy(params, samples, model_apply)
        lab_samples = self._samples_to_lab(samples)
        return kinetic.squeeze() + self.potential_energy(lab_samples).squeeze()


@register_hamiltonian("harmonic-oscillator")
@struct.dataclass
class HarmonicOscillatorHamiltonian(ContinuousHamiltonian):
    omega: float = 1.0

    def potential_energy(self, samples):
        return 0.5 * (self.omega**2) * jnp.sum(samples**2, axis=-1)


@register_hamiltonian("nn-oscillator")
@struct.dataclass
class NN_OscillatorHamiltonian(ContinuousHamiltonian):
    """Nearest-neighbour harmonic oscillator."""

    omega_trap: float = 1.0
    omega_interaction: float = 1.0
    with_pbc: bool = struct.field(pytree_node=False, default=True)

    def potential_energy(self, samples):
        trap = 0.5 * (self.omega_trap**2) * jnp.sum(samples**2, axis=-1)
        if self.with_pbc:
            diffs = samples - jnp.roll(samples, shift=1, axis=-1)
        else:
            diffs = samples[:, :-1] - samples[:, 1:]
        nn_term = 0.5 * self.omega_interaction**2 * jnp.sum(diffs**2, axis=-1)
        return trap + nn_term


@register_hamiltonian("soft-core")
@struct.dataclass
class SoftCoreHamiltonian(ContinuousHamiltonian):
    R: float = 1.0
    V0: float = 1.0

    def potential_energy(self, samples):
        r = jnp.linalg.norm(samples, axis=-1)
        return jnp.where(r < self.R, self.V0, 0.0)


@register_hamiltonian("gross-struct-hamiltonian")
@struct.dataclass
class GrossStructHamiltonian(ContinuousHamiltonian):
    """Electron-nuclear attraction. DoF = n_fermions * 3 (3D)."""

    Z: int = struct.field(pytree_node=False, default=1)
    n_fermions: int = struct.field(pytree_node=False, default=1)

    def potential_energy(self, samples):
        pos = samples.reshape(-1, self.n_fermions, 3)
        r_i = jnp.linalg.norm(pos, axis=-1)
        return -self.Z * jnp.sum(1.0 / (r_i + 1e-12), axis=-1)


@register_hamiltonian("CS-model")
@struct.dataclass
class CalogeroSutherlandHamiltonian(ContinuousHamiltonian):
    """Calogero-Sutherland model: particles on a line with inverse-square interactions."""

    L: float = struct.field(pytree_node=False, default=1.0)
    epsilon: float = struct.field(pytree_node=False, default=0.0)
    omega_trap: float = struct.field(pytree_node=False, default=1.0)

    def kinetic_local_energy(self, params, samples, model_apply):
        # Factor of 2: CS convention is H = -d²/dx² + V (ℏ²/m = 1, not ℏ²/2m = 1).
        return 2 * kinetic_log(params, samples, model_apply, self._get_laplacian_fn())

    def potential_energy(self, samples):
        diffs = samples[:, :, jnp.newaxis] - samples[:, jnp.newaxis, :]
        mask = jnp.triu(jnp.ones(diffs.shape[1:]), k=1)
        inv_square = jnp.where(
            mask, self.L * (self.L - 1) / (diffs**2 + self.epsilon), 0.0
        )
        trap = (self.omega_trap**2) * jnp.sum(samples**2, axis=-1)
        return 2 * jnp.sum(inv_square, axis=(-1, -2)) + trap
