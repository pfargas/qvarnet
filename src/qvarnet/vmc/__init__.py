"""Ground-state variational Monte Carlo.

The driver is :class:`VMC` (``train`` is its function form). A run is four phases,
one module each: ``setup`` builds the context, ``warmup`` equilibrates the walkers,
``step`` builds the jitted per-epoch update, ``loop`` runs the epochs.

Shared machinery -- samplers, Hamiltonians, ansatze, geometry/QGT, analysis --
lives in sibling packages, so a future method (DMC, PIGS, t-VMC) can reuse it.
"""

from .context import TrainContext
from .driver import VMC, train
from .probability import build_prob_fn
from .step import make_update_fn
from .train_result import TrainResult
from .training_step import (
    compute_local_energy,
    compute_step,
    energy_and_grads,
    energy_fn,
)
from .vmc_state import VMCState

__all__ = [
    "VMC",
    "train",
    "TrainContext",
    "TrainResult",
    "VMCState",
    "build_prob_fn",
    "make_update_fn",
    "compute_step",
    "energy_fn",
    "energy_and_grads",
    "compute_local_energy",
]
