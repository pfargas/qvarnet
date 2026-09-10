"""How long the MCMC chains run.

How they *move* is the Sampler you hand to ``VMC(sampler=...)`` -- it owns the
proposal family and any constraint. One knob, one owner.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingConfig:
    """Immutable sampling configuration for MCMC.

How *long* to run the chains. *How* they move is the ``Sampler`` you pass to
    ``train(sampler=...)`` -- which owns the proposal family and any constraint
    (see ``qvarnet.sampling.sampler``). One knob, one owner.
    """

    step_size: float = 1.0
    chain_length: int = 500
    thermalization_steps: int = 50
    thinning_factor: int = 5
    box_L: float | None = None  # PBC sampler: wrap proposals into [0, L). None = off.

    def __post_init__(self):
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")
        if self.box_L is not None and self.box_L <= 0:
            raise ValueError(f"box_L must be positive when set, got {self.box_L}")
        if self.thinning_factor < 1:
            raise ValueError(f"thinning_factor must be >= 1, got {self.thinning_factor}")
        if self.thermalization_steps >= self.chain_length:
            raise ValueError(
                f"thermalization_steps ({self.thermalization_steps}) must be "
                f"< chain_length ({self.chain_length})"
            )
        if self.thermalization_steps < 0:
            raise ValueError(f"thermalization_steps must be >= 0, got {self.thermalization_steps}")
        if self.chain_length < self.thermalization_steps + 1:
            raise ValueError(
                f"chain_length must be >= thermalization_steps, got {self.chain_length} < {self.thermalization_steps}"
            )
