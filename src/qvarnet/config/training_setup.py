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
    init_positions: str  # "normal" or "zeros"
    is_update_step_size: bool
    min_step: float
    max_step: float
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
    """
    Convert dict-based training configuration to typed dataclass.

    Args:
        train_args: Dictionary with training parameters

    Returns:
        TrainingConfig: Typed, immutable configuration
    """
    return TrainingConfig(
        n_epochs=int(train_args.get("num_epochs", 3000)),
        init_positions=str(train_args.get("init_positions", "normal")),
        is_update_step_size=bool(train_args.get("is_update_step_size", False)),
        min_step=float(train_args.get("min_step", 1e-5)),
        max_step=float(train_args.get("max_step", 5.0)),
        target_acceptance=float(train_args.get("target_acceptance", 0.5)),
        adaptation_rate=float(train_args.get("adaptation_rate", 0.1)),
    )
