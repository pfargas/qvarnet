"""Periodic continuum Hamiltonians.

``LatticeBoseHamiltonian`` is promoted verbatim from ``notebooks/scripts/bose-hubbard.ipynb``
so the engine ships (and tests) a real PBC system instead of redefining one per notebook. It
subclasses :class:`~qvarnet.boundaries.BoundaryHamiltonian`, so periodicity is controlled by the
``boundary`` field (``PeriodicBoundary(L)`` → minimum-image pair distances; ``NoBoundary`` →
free space) and reuses the shared ``_min_image`` machinery.
"""

import jax.numpy as jnp
from flax import struct

from ..boundaries import BoundaryHamiltonian
from .hamiltonian_registry import register_hamiltonian


@register_hamiltonian("lattice-bose")
@struct.dataclass
class LatticeBoseHamiltonian(BoundaryHamiltonian):
    """N bosons in a 1D optical lattice with contact interactions.

    H = -½ Σᵢ ∂²/∂xᵢ² + V₀ Σᵢ sin²(π xᵢ/a) + (g/σ√2π) Σᵢ<ⱼ exp(−rᵢⱼ²/2σ²)

    The Gaussian approximates a δ-function contact interaction when σ ≪ a.
    ``boundary`` controls whether pairwise distances use minimum-image (PBC) or
    raw coordinates (free space) — inherited from ``BoundaryHamiltonian``.

    The lattice potential sin²(π x/a) is exactly periodic with period ``a``; pick ``a``
    so the box length L is an integer multiple of ``a`` for a consistent PBC system.
    """

    a: float = struct.field(pytree_node=False, default=1.0)    # lattice spacing
    V0: float = 1.0   # lattice depth
    g: float = 1.0    # contact interaction strength g_1D
    sigma: float = struct.field(pytree_node=False, default=0.05)  # Gaussian width

    def potential_energy(self, samples):
        # samples: (batch, N)

        # ── Lattice potential ─────────────────────────────────────────────────
        V_latt = self.V0 * jnp.sum(jnp.sin(jnp.pi * samples / self.a) ** 2, axis=-1)

        # ── Contact interaction (Gaussian approx of δ) ────────────────────────
        n_part = samples.shape[-1]
        i_idx, j_idx = jnp.triu_indices(n_part, k=1)
        dx = samples[:, i_idx] - samples[:, j_idx]           # raw separations
        dx = self._min_image(dx)                              # minimum image (PBC or identity)
        amplitude = self.g / (self.sigma * jnp.sqrt(2 * jnp.pi))
        V_int = amplitude * jnp.sum(jnp.exp(-dx**2 / (2 * self.sigma**2)), axis=-1)

        return V_latt + V_int


@register_hamiltonian("penetrable-sphere")
@struct.dataclass
class PenetrableSphereHamiltonian(BoundaryHamiltonian):
    r"""N "soft" bosons with a pairwise penetrable-sphere (soft-core step) interaction.

    .. math::
        V = V_0 \sum_{i<j} \theta(R - r_{ij}),
        \qquad r_{ij} = \lvert \mathrm{min\_image}(\mathbf x_i - \mathbf x_j)\rvert .

    A *penetrable* (finite-height, flat) step: particles may overlap at finite energy cost ``V0``
    — no hard-core divergence, so the wavefunction stays smooth and NN-friendly. Kinetic energy is
    ``-½∇²`` (from ``ContinuousHamiltonian``). ``boundary=PeriodicBoundary(L)`` gives a cubic PBC
    box via minimum-image distances; ``NoBoundary`` gives free space.

    Works in **any spatial dimension**: set ``n_dim`` (1/2/3). Positions are the usual flat
    ``(batch, N*n_dim)``; the per-particle reshape happens here. ``V0`` and ``R`` are the two
    physical knobs (interaction height and range); together with the number density they set the
    phase of the gas.
    """

    n_dim: int = struct.field(pytree_node=False, default=1)
    R: float = 1.0
    V0: float = 1.0

    def potential_energy(self, samples):
        b = samples.shape[0]
        n = samples.shape[-1] // self.n_dim
        x = samples.reshape(b, n, self.n_dim)              # (batch, N, d)
        dx = x[:, :, None, :] - x[:, None, :, :]           # (batch, N, N, d)
        dx = self._min_image(dx)                           # per-component minimum image
        r = jnp.sqrt(jnp.sum(dx**2, axis=-1) + 1e-12)      # (batch, N, N) pair distances
        i_idx, j_idx = jnp.triu_indices(n, k=1)
        r_pairs = r[:, i_idx, j_idx]                       # (batch, n_pairs)
        inside = (r_pairs < self.R).astype(samples.dtype)
        return self.V0 * jnp.sum(inside, axis=-1)
