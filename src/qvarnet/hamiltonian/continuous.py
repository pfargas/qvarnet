from .base import BaseHamiltonian
from .hamiltonian_registry import register_hamiltonian
from .kinetic import kinetic_log
from .laplacian import laplacian_forward_ad, laplacian_full_hessian, laplacian_central_difference, laplacian_hutchinson
from flax import struct

import jax.numpy as jnp
from qvarnet.config.coord_mode import CoordMode, LabCoords


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
    hutchinson_n_terms: int = struct.field(pytree_node=False, default=10)
    hutchinson_distribution: str = struct.field(pytree_node=False, default="rademacher")

    def _get_laplacian_fn(self):
        method = self.laplacian_method
        if method == "forward_ad":
            return laplacian_forward_ad
        if method == "central_difference":
            return laplacian_central_difference
        if method == "full_hessian":
            return laplacian_full_hessian
        if method == "hutchinson":
            from functools import partial
            return partial(
                laplacian_hutchinson,
                n_terms=self.hutchinson_n_terms,
                distribution=self.hutchinson_distribution,
            )
        raise ValueError(f"Unknown laplacian_method: {method!r}")

    def kinetic_local_energy(self, params, samples, model_apply, key=None):
        return kinetic_log(params, samples, model_apply, self._get_laplacian_fn(), key=key)

    def potential_energy(self, samples):
        raise NotImplementedError("Subclass must implement potential_energy().")

    def local_energy(self, params, samples, model_apply, key=None):
        kinetic = self.kinetic_local_energy(params, samples, model_apply, key=key)
        coord_mode = self.coord_mode if self.coord_mode is not None else LabCoords()
        lab_samples = coord_mode.samples_to_lab(samples)
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
    """Calogero-Sutherland model: particles on a line with inverse-square interactions.

    pairwise_impl controls the pairwise sum strategy:
        "vectorized" (default) — all N*(N-1)/2 pairs at once.
                                 Time O(B·N²), Memory O(B·N²).  Best on GPU for N ≲ 100.
        "scan"                 — one particle row at a time via fori_loop.
                                 Time O(B·N²), Memory O(B·N).   Better for large N.
    """

    L: float = struct.field(pytree_node=False, default=1.0)
    epsilon: float = struct.field(pytree_node=False, default=0.0)
    omega_trap: float = struct.field(pytree_node=False, default=1.0)
    pairwise_impl: str = struct.field(pytree_node=False, default="vectorized")

    def kinetic_local_energy(self, params, samples, model_apply, key=None):
        # Factor of 2: CS convention is H = -d²/dx² + V (ℏ²/m = 1, not ℏ²/2m = 1).
        return 2 * kinetic_log(params, samples, model_apply, self._get_laplacian_fn(), key=key)

    def potential_energy(self, samples):
        import jax
        n = samples.shape[-1]
        L, eps = self.L, self.epsilon
        trap = (self.omega_trap**2) * jnp.sum(samples**2, axis=-1)

        if self.pairwise_impl == "scan":
            # O(B·N) peak memory: one row at a time via fori_loop.
            # i is a traced integer inside fori_loop — dynamic slices are forbidden,
            # so we gather particle i and mask out j <= i positions.
            col = jnp.arange(n)

            def body(i, acc):
                xi = jax.lax.dynamic_index_in_dim(samples, i, axis=1, keepdims=False)
                diffs = xi[:, None] - samples                       # (batch, N)
                mask = (col > i).astype(jnp.float32)
                inv_sq = L * (L - 1) / (diffs**2 + eps) * mask
                return acc + jnp.sum(inv_sq, axis=-1)

            interaction = jax.lax.fori_loop(0, n - 1, body, jnp.zeros(samples.shape[0]))
        else:
            # O(B·N²/2) peak memory: all pairs at once, max GPU parallelism.
            i_idx, j_idx = jnp.triu_indices(n, k=1)
            diffs = samples[:, i_idx] - samples[:, j_idx]     # (batch, n_pairs)
            interaction = jnp.sum(L * (L - 1) / (diffs**2 + eps), axis=-1)

        return 2 * interaction + trap # factor 2 cause g = 2L(L-1) in CS convention, not L(L-1)
