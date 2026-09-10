"""Configuration for a VMC run: epochs, seeds, checkpointing, warmup, cusp loss.

All frozen dataclasses, and all passed to ``jax.jit`` as static arguments --
so they must stay hashable, and changing any field triggers a retrace.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChainInitAndWarmupConfig:
    """Configuration for chain initialization and warmup.

    ``init_positions`` is either a strategy name ("normal" | "zeros" | "uniform") or an
    explicit ``(n_chains, dof)`` array — e.g. ``result.final_positions`` of a previous
    run, so a warm-started rerun resumes from equilibrated walkers instead of
    re-thermalising from scratch.

    ``warmup_adapt_step_size`` runs the warmup in ``warmup_n_blocks`` blocks and retunes
    the proposal step between blocks toward ``TrainingConfig.target_acceptance``
    (proportional update, factor clipped per block). The adapted step then seeds the
    training sampler when ``TrainingConfig.is_update_step_size`` is set — this removes
    the near-frozen-chain regime of a warm-started run whose converged |ψ|² needs a much
    smaller step than ``SamplingConfig.step_size``.
    """

    init_position_params: dict[str, Any] | None = None
    warmup_starting_positions: bool = True  # if True, the chains are warmed up before the first epoch, otherwise they are initialized from init_positions
    init_positions: Any = "normal"  # "normal" | "zeros" | "uniform" | (n_chains, dof) array
    warmup_steps: int = 300
    warmup_step_size: float = 0.5
    warmup_adapt_step_size: bool = False
    warmup_n_blocks: int = 10

    def __post_init__(self):
        if isinstance(self.init_positions, str) and self.init_positions not in (
            "normal",
            "zeros",
            "uniform",
        ):
            raise ValueError(
                "init_positions must be 'normal', 'zeros', 'uniform' or an "
                f"(n_chains, dof) array, got {self.init_positions!r}"
            )
        if self.warmup_n_blocks < 1:
            raise ValueError(f"warmup_n_blocks must be >= 1, got {self.warmup_n_blocks}")


@dataclass(frozen=True)
class CuspConfig:
    """Configuration for the cusp condition auxiliary loss.

    alpha:             weight of the cusp loss relative to the VMC loss
    epsilon:           regularisation distance at which cusp condition is enforced
    n_configs_per_pair: number of sample points per particle pair
    rng_seed:          seed for cusp config generation
    n:                 potential exponent (2 for CS, >2 for other power-law potentials)
    C_n:               target cusp value (λ for CS n=2, sqrt(g) for n>2)
    L:                 box/ring size for cusp config generation.  If None, train()
                       will try hamiltonian.L; raises if neither is set.
    """

    alpha: float = 0.01
    epsilon: float = 1e-2
    n_configs_per_pair: int = 5
    rng_seed: int = 42
    n: float = 2.0
    C_n: float = 1.0
    L: float | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration."""

    n_epochs: int
    rng_seed: int = 0
    init_chains_config: ChainInitAndWarmupConfig = ChainInitAndWarmupConfig()
    init_positions: str = init_chains_config.init_positions  # "normal" | "zeros" | "uniform"
    # Carry walker positions across epochs, so the chain equilibrates once instead of
    # re-thermalising from scratch every epoch. False is only useful for sampler debugging.
    warm_walkers: bool = True
    # Print TrainResult.summary() when training ends (notebook-friendly run report).
    print_summary: bool = True
    is_update_step_size: bool = False
    min_step: float = 1e-5
    max_step: float = 5.0
    use_qgt: bool = False
    checkpoint_path: str = "./"
    save_checkpoints: bool = False
    target_acceptance: float = 0.5
    adaptation_rate: float = 0.1
    cusp: CuspConfig | None = None  # None = cusp disabled

    def __post_init__(self):
        if self.min_step >= self.max_step:
            raise ValueError(f"min_step ({self.min_step}) must be < max_step ({self.max_step})")
