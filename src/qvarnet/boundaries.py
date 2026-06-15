"""Boundary condition objects for VMC simulations.

Usage
-----
boundary = PeriodicBoundary(L=10.0)   # or NoBoundary()

# Wrap any Flax model:
model = BoundaryModel(inner=MyModel(...), boundary=boundary)

# Write Hamiltonians that call self._min_image(dx):
@struct.dataclass
class MyHamiltonian(BoundaryHamiltonian):
    def potential_energy(self, samples):
        dx = samples[:, i_idx] - samples[:, j_idx]
        dx = self._min_image(dx)   # correct for both NoBoundary and PeriodicBoundary
        ...

# To add a new boundary type, implement encode / feature_dim / min_image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from flax import struct

from .hamiltonian.continuous import ContinuousHamiltonian

# ── boundary condition objects ────────────────────────────────────────────────


@dataclass(frozen=True)
class NoBoundary:
    """Free space — no wrapping, no minimum image."""

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def feature_dim(self, n: int) -> int:
        return n

    def min_image(self, dx: jnp.ndarray) -> jnp.ndarray:
        return dx

    def per_particle_dim(self, n_spatial: int) -> int:
        """Feature dimension per particle for a DeepSet inner model.

        Pass this as ``n_dim`` when constructing any model that reshapes its
        input to ``(batch, n_particles, n_dim)`` (e.g. DeepSet, DeepSetNoEnvelope).
        """
        return n_spatial

    def output_dim(self, n_particles: int, n_spatial: int) -> int:
        """Total encoded feature dimension: n_particles * n_spatial."""
        return n_particles * n_spatial

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.encode(x)


@dataclass(frozen=True)
class PeriodicBoundary:
    """Hypercubic periodic box of side length L.

    encode   : x  →  [sin(2π x / L),  cos(2π x / L)]   — exact periodicity
    min_image: dx →  dx - L * round(dx / L)             — shortest image vector
    """

    L: float

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        # Interleave sin/cos **per coordinate** so a downstream per-particle reshape
        # (..., N*d) -> (..., N, 2d) keeps each particle's own features together:
        #   [sin(x0), cos(x0), sin(x1), cos(x1), ...]
        # A naive concatenate([sin(all), cos(all)]) would instead group sines of
        # different particles into the same row, scrambling the DeepSet input.
        phase = 2.0 * jnp.pi * x / self.L
        stacked = jnp.stack([jnp.sin(phase), jnp.cos(phase)], axis=-1)  # (..., D, 2)
        return stacked.reshape(*x.shape[:-1], 2 * x.shape[-1])

    def feature_dim(self, n: int) -> int:
        return 2 * n

    def min_image(self, dx: jnp.ndarray) -> jnp.ndarray:
        return dx - self.L * jnp.round(dx / self.L)

    def per_particle_dim(self, n_spatial: int) -> int:
        """Feature dimension per particle for a DeepSet inner model.

        Pass this as ``n_dim`` when constructing any model that reshapes its
        input to ``(batch, n_particles, n_dim)`` (e.g. DeepSet, DeepSetNoEnvelope).
        sin/cos encoding doubles the dimension: returns ``2 * n_spatial``.
        """
        return 2 * n_spatial

    def output_dim(self, n_particles: int, n_spatial: int) -> int:
        """Total encoded feature dimension: 2 * n_particles * n_spatial."""
        return 2 * n_particles * n_spatial

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.encode(x)


# ── model wrapper ─────────────────────────────────────────────────────────────


class BoundaryModel(nn.Module):
    """Prepends boundary encoding to any Flax model.

    The inner model never sees raw coordinates — it always receives the
    encoded representation produced by ``boundary.encode``.  Flax's lazy
    Dense layers absorb the (possibly changed) input dimension automatically,
    so ``inner`` does not need to be redesigned when switching boundaries.

    Parameters
    ----------
    inner:
        Any ``nn.Module`` that maps (..., encoded_dim) → (..., 1).
    boundary:
        A ``NoBoundary`` or ``PeriodicBoundary`` instance.
    """

    inner: nn.Module
    boundary: Any  # NoBoundary | PeriodicBoundary — not a JAX array

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.inner(self.boundary.encode(x))


# ── Hamiltonian base ──────────────────────────────────────────────────────────


@struct.dataclass
class BoundaryHamiltonian(ContinuousHamiltonian):
    """Base class for Hamiltonians with a configurable boundary condition.

    Subclasses should call ``self._min_image(dx)`` on any pairwise displacement
    vector instead of using ``dx`` directly.  This is the only change needed to
    support both open and periodic boundary conditions from the same code.

    Parameters
    ----------
    boundary:
        A ``NoBoundary`` or ``PeriodicBoundary`` instance.  Stored as a
        non-pytree field so it is treated as a static compile-time constant
        by JAX/Flax.
    """

    boundary: Any = struct.field(pytree_node=False, default=None)

    def _min_image(self, dx: jnp.ndarray) -> jnp.ndarray:
        if self.boundary is None or isinstance(self.boundary, NoBoundary):
            return dx
        return self.boundary.min_image(dx)
