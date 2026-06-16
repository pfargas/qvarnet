"""Training configuration setup and parsing."""

from dataclasses import dataclass
from typing import Any


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
class SamplingConfig:
    """Immutable sampling configuration for MCMC."""

    step_size: float
    chain_length: int
    thermalization_steps: int
    thinning_factor: int
    block_size: int = 0  # 0 = disabled; >0 caps peak random-number memory
    box_L: float | None = None  # PBC sampler: wrap proposals into [0, L). None = off.
    # Parallel tempering (generic barrier-crossing addition; see samplers/parallel_tempering.py).
    # sampler="mh" (default) is the plain local-move chain; "pt" stacks replicas per chain at
    # inverse temperatures 1=β₁>…>β_R sampling |ψ|^{2β} and returns only the cold (β=1) replica.
    sampler: str = "mh"            # "mh" | "pt"
    pt_n_replicas: int = 4         # number of temperature replicas (used if pt_betas is None)
    pt_beta_min: float = 0.1       # coldest→hottest geometric ladder endpoint
    pt_betas: tuple | None = None  # explicit ladder (must start at 1.0); overrides the two above
    swap_every: int = 1            # attempt a replica swap every this many steps
    pt_scale_steps: bool = True    # hotter replicas take larger steps (σ/√β)

    def __post_init__(self):
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")
        if self.box_L is not None and self.box_L <= 0:
            raise ValueError(f"box_L must be positive when set, got {self.box_L}")
        if self.sampler not in ("mh", "pt"):
            raise ValueError(f"sampler must be 'mh' or 'pt', got {self.sampler!r}")
        if self.pt_betas is not None and abs(self.pt_betas[0] - 1.0) > 1e-12:
            raise ValueError(f"pt_betas[0] must be 1.0 (physical replica), got {self.pt_betas[0]}")
        if self.pt_n_replicas < 1:
            raise ValueError(f"pt_n_replicas must be >= 1, got {self.pt_n_replicas}")
        if self.thinning_factor < 1:
            raise ValueError(f"thinning_factor must be >= 1, got {self.thinning_factor}")
        if self.thermalization_steps >= self.chain_length:
            raise ValueError(
                f"thermalization_steps ({self.thermalization_steps}) must be "
                f"< chain_length ({self.chain_length})"
            )
        if self.block_size < 0:
            raise ValueError(f"block_size must be >= 0, got {self.block_size}")
        if self.block_size > 0 and self.chain_length % self.block_size != 0:
            raise ValueError(
                f"block_size ({self.block_size}) must divide "
                f"chain_length ({self.chain_length}) exactly."
            )
        if self.thermalization_steps < 0:
            raise ValueError(f"thermalization_steps must be >= 0, got {self.thermalization_steps}")
        if self.chain_length < self.thermalization_steps + 1:
            raise ValueError(
                f"chain_length must be >= thermalization_steps, got {self.chain_length} < {self.thermalization_steps}"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration."""

    n_epochs: int
    rng_seed: int = 0
    init_positions: str = "normal"  # "normal" | "zeros"
    warm_walkers: bool = False
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
        if self.init_positions not in ("normal", "zeros", "uniform"):
            raise ValueError(
                f"init_positions must be 'normal', 'zeros' or 'uniform', got {self.init_positions!r}"
            )


def parse_sampler_params(sampler_args: dict[str, Any]) -> SamplingConfig:
    """Convert dict-based sampler configuration to typed dataclass."""
    raw_box_L = sampler_args.get("box_L", None)
    raw_betas = sampler_args.get("pt_betas", None)
    return SamplingConfig(
        step_size=float(sampler_args.get("step_size", 1.0)),
        chain_length=int(sampler_args.get("chain_length", 500)),
        thermalization_steps=int(sampler_args.get("thermalization_steps", 50)),
        thinning_factor=int(sampler_args.get("thinning_factor", 5)),
        block_size=int(sampler_args.get("block_size", 0)),
        box_L=float(raw_box_L) if raw_box_L is not None else None,
        sampler=str(sampler_args.get("sampler", "mh")),
        pt_n_replicas=int(sampler_args.get("pt_n_replicas", 4)),
        pt_beta_min=float(sampler_args.get("pt_beta_min", 0.1)),
        pt_betas=tuple(float(b) for b in raw_betas) if raw_betas is not None else None,
        swap_every=int(sampler_args.get("swap_every", 1)),
        pt_scale_steps=bool(sampler_args.get("pt_scale_steps", True)),
    )


def parse_training_params(train_args: dict[str, Any]) -> TrainingConfig:
    """Convert dict-based training configuration to typed dataclass."""
    cusp = None
    if bool(train_args.get("use_cusp_condition", False)):
        raw_L = train_args.get("cusp_L", None)
        cusp = CuspConfig(
            alpha=float(train_args.get("cusp_alpha", 0.01)),
            epsilon=float(train_args.get("cusp_epsilon", 1e-2)),
            n_configs_per_pair=int(train_args.get("cusp_n_configs_per_pair", 5)),
            rng_seed=int(train_args.get("cusp_rng_seed", 42)),
            n=float(train_args.get("cusp_n", 2.0)),
            C_n=float(train_args.get("cusp_C_n", 1.0)),
            L=float(raw_L) if raw_L is not None else None,
        )
    return TrainingConfig(
        n_epochs=int(train_args.get("num_epochs", 3000)),
        rng_seed=int(train_args.get("rng_seed", 0)),
        init_positions=str(train_args.get("init_positions", "normal")),
        warm_walkers=bool(train_args.get("warm_walkers", False)),
        is_update_step_size=bool(train_args.get("is_update_step_size", False)),
        min_step=float(train_args.get("min_step", 1e-5)),
        max_step=float(train_args.get("max_step", 5.0)),
        use_qgt=bool(train_args.get("use_qgt", False)),
        checkpoint_path=str(train_args.get("checkpoint_path", "./")),
        save_checkpoints=bool(train_args.get("save_checkpoints", False)),
        target_acceptance=float(train_args.get("target_acceptance", 0.5)),
        adaptation_rate=float(train_args.get("adaptation_rate", 0.1)),
        cusp=cusp,
    )
