"""Training configuration setup and parsing."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SamplingConfig:
    """Immutable sampling configuration for MCMC."""

    step_size: float
    chain_length: int
    thermalization_steps: int
    thinning_factor: int
    PBC: float
    is_log_prob: bool


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration."""

    n_epochs: int
    rng_seed: int = 0
    init_positions: str = "normal"       # "normal" | "zeros"
    warm_walkers: bool = False
    is_update_step_size: bool = False
    min_step: float = 1e-5
    max_step: float = 5.0
    is_log_model: bool = False
    use_qgt: bool = False
    checkpoint_path: str = "./"
    save_checkpoints: bool = False
    target_acceptance: float = 0.5
    adaptation_rate: float = 0.1


def parse_sampler_params(sampler_args: Dict[str, Any], is_log_prob: bool = False) -> SamplingConfig:
    """
    Convert dict-based sampler configuration to typed dataclass.

    Args:
        sampler_args: Dictionary with sampler parameters
        is_log_prob: Whether model outputs log(ψ)

    Returns:
        SamplingConfig: Typed, immutable configuration
    """
    return SamplingConfig(
        step_size=float(sampler_args.get("step_size", 1.0)),
        chain_length=int(sampler_args.get("chain_length", 500)),
        thermalization_steps=int(sampler_args.get("thermalization_steps", 50)),
        thinning_factor=int(sampler_args.get("thinning_factor", 5)),
        PBC=float(sampler_args.get("PBC", 40.0)),
        is_log_prob=is_log_prob,
    )


def parse_training_params(train_args: Dict[str, Any]) -> TrainingConfig:
    """Convert dict-based training configuration to typed dataclass."""
    return TrainingConfig(
        n_epochs=int(train_args.get("num_epochs", 3000)),
        rng_seed=int(train_args.get("rng_seed", 0)),
        init_positions=str(train_args.get("init_positions", "normal")),
        warm_walkers=bool(train_args.get("warm_walkers", False)),
        is_update_step_size=bool(train_args.get("is_update_step_size", False)),
        min_step=float(train_args.get("min_step", 1e-5)),
        max_step=float(train_args.get("max_step", 5.0)),
        is_log_model=bool(train_args.get("is_log_model", False)),
        use_qgt=bool(train_args.get("use_qgt", False)),
        checkpoint_path=str(train_args.get("checkpoint_path", "./")),
        save_checkpoints=bool(train_args.get("save_checkpoints", False)),
        target_acceptance=float(train_args.get("target_acceptance", 0.5)),
        adaptation_rate=float(train_args.get("adaptation_rate", 0.1)),
    )
