"""Observables package (roadmap step 5): post-training physics estimators with blocking errors.

- ``base``: blocking error analysis (Flyvbjerg-Petersen).
- ``density``: single-particle density n(x), pair correlation g(r).
- ``structure``: static structure factor S(k).
- ``obdm``: one-body density matrix, natural orbitals, condensate fraction.

All currently 1-D; evaluate from a batch drawn from |ψ|² (thin by the IAT).
"""

from .base import blocking_error, mean_and_error
from .density import density_histogram, pair_correlation
from .obdm import (
    condensate_fraction,
    natural_orbitals,
    obdm_displacement,
    obdm_grid,
)
from .structure import commensurate_k, structure_factor

__all__ = [
    "blocking_error",
    "mean_and_error",
    "density_histogram",
    "pair_correlation",
    "structure_factor",
    "commensurate_k",
    "obdm_grid",
    "natural_orbitals",
    "condensate_fraction",
    "obdm_displacement",
]
