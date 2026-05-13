from dataclasses import dataclass


@dataclass(frozen=True)
class CoordMode:
    pass


@dataclass(frozen=True)
class LabCoords(CoordMode):
    """Default: sampler and model operate in Cartesian lab coordinates."""
    pass


@dataclass(frozen=True)
class JacobiCoords(CoordMode):
    """Sampler works in N Jacobi relative coords; model receives N+1 reconstructed lab coords.

    n_particles_physical: total number of physical particles (N + 1)
    n_dim: spatial dimension (currently only 1 is supported)
    """
    n_particles_physical: int
    n_dim: int = 1
