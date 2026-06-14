"""Backward-compat shim — moved to ``qvarnet.vmc.training_step``."""

from .vmc.training_step import (  # noqa: F401
    compute_local_energy,
    compute_step,
    energy_and_grads,
    energy_fn,
)
