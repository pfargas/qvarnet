"""Static particle-structure metadata for Hamiltonians.

The engine works on flat (batch, dof) sample vectors, so by itself the kinetic layer
cannot distinguish 2 particles in 1D from 1 particle in 2D — for equal masses the two
are the *same* operator, so nothing was lost. :class:`Particles` makes the grouping
explicit the moment an operator needs per-particle attributes; the first user is mass
imbalance, H = -Σ_i 1/(2 m_i) ∇²_i + V, written as-is on the Hamiltonian:

    H = SomeHamiltonian(..., particles=Particles(n=3, n_dim=1, masses=(1.0, 1.0, 5.0)))

Samples everywhere stay flat; only the kinetic energy consumes the per-dof weights.
Frozen + hashable (masses is a tuple), so it is safe as a static (``pytree_node=False``)
field on a ``flax.struct`` Hamiltonian. Future per-particle attributes (species labels,
charges) belong here too.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Particles:
    """(n, n_dim) structure of the dof vector, with optional per-particle masses."""

    n: int
    n_dim: int = 1
    masses: tuple[float, ...] | None = None  # per particle; None = all equal (m = 1)

    def __post_init__(self):
        if self.n <= 0 or self.n_dim <= 0:
            raise ValueError(f"need n > 0 and n_dim > 0, got n={self.n}, n_dim={self.n_dim}")
        if self.masses is not None:
            if len(self.masses) != self.n:
                raise ValueError(
                    f"masses has {len(self.masses)} entries for n={self.n} particles"
                )
            if any(m <= 0 for m in self.masses):
                raise ValueError(f"masses must be positive, got {self.masses}")

    @property
    def dof(self) -> int:
        return self.n * self.n_dim

    def dof_weights(self):
        """Per-dof kinetic weights 1/m (mass repeated over its n_dim coordinates).

        Returns None when all masses are equal (= 1), so the unweighted fast paths run.
        """
        if self.masses is None:
            return None
        m = jnp.repeat(jnp.asarray(self.masses, dtype=jnp.float32), self.n_dim)
        return 1.0 / m
