"""VMC method package: ground-state variational Monte Carlo.

Holds the method-specific training loop, step, state, and result. Shared machinery
(samplers, hamiltonians, models, geometry/QGT, diagnostics, observables) lives in
sibling packages so future methods (DMC/PIGS/t-VMC) can reuse it.
"""

from .probability import build_prob_fn
from .train import train
from .train_result import TrainResult
from .training_step import (
    compute_local_energy,
    compute_step,
    energy_and_grads,
    energy_fn,
)
from .vmc_state import VMCState

__all__ = [
    "train",
    "TrainResult",
    "VMCState",
    "build_prob_fn",
    "compute_step",
    "energy_fn",
    "energy_and_grads",
    "compute_local_energy",
]
