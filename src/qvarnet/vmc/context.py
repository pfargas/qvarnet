"""The state a VMC run carries between its phases."""

from dataclasses import dataclass, field
from typing import Any

from qvarnet.core.metrics import MetricsHistory


@dataclass
class TrainContext:
    """Everything one VMC run needs, assembled once by ``setup.build_context``.

    Split in two: the fields above ``state`` are fixed for the run (and the frozen
    configs among them are jit-static), the ones below evolve as it proceeds.
    Phases read and mutate this rather than passing a dozen positional arguments.
    """

    # -- fixed for the run ----------------------------------------------------
    shape: tuple
    model: Any
    hamiltonian: Any
    coord_mode: Any
    sampler: Any
    prob_fn: Any
    sampling_config: Any
    training_config: Any
    initial_chain_config: Any
    qgt_config: Any
    auxiliary_losses: tuple

    # -- evolves during the run -----------------------------------------------
    state: Any
    key: Any
    positions: Any
    step_size: Any

    # -- bookkeeping ----------------------------------------------------------
    callbacks: list = field(default_factory=list)
    snapshot_cb: Any = None
    history: MetricsHistory = field(default_factory=MetricsHistory)

    @property
    def n_chains(self) -> int:
        return self.shape[0]

    @property
    def dof(self) -> int:
        return self.shape[1]

    @property
    def box_L(self) -> float:
        """The sampler's periodic box; 0.0 means no wrapping."""
        return self.sampling_config.box_L or 0.0
