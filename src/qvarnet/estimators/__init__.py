"""Post-training property estimators of a trained many-body wavefunction.

The entry point is ``TrainedWavefunction`` — a frozen ψ with (model, params,
coord_mode, box) baked in, owning its sampler and cached lab-coordinate
samples, with estimator methods for density, pair correlation, structure
factor and the one-body density matrix.

The numerical kernels live in ``kernels`` and are usable standalone with raw
lab-coordinate samples.
"""

from .kernels import blocking_error, mean_and_error
from .trained_wavefunction import TrainedWavefunction

__all__ = [
    "TrainedWavefunction",
    "blocking_error",
    "mean_and_error",
]
