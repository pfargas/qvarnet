"""Training state management, history tracking, and checkpointing."""

from typing import List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import jax.numpy as jnp

from .vmc_state import VMCState
from .utils import save_checkpoint, load_checkpoint


@dataclass
class TrainingHistory:
    """Track and record training progress over epochs."""

    energies: List[float] = field(default_factory=list)
    stds: List[float] = field(default_factory=list)
    acceptance_rates: List[float] = field(default_factory=list)
    step_sizes: List[float] = field(default_factory=list)

    def record(
        self, energy: float, std: float, acceptance_rate: float, step_size: float
    ) -> None:
        """
        Record metrics from one training epoch.

        Args:
            energy: Energy expectation value
            std: Standard error of energy
            acceptance_rate: MCMC acceptance rate
            step_size: Current MH step size
        """
        self.energies.append(float(energy))
        self.stds.append(float(std))
        self.acceptance_rates.append(float(acceptance_rate))
        self.step_sizes.append(float(step_size))

    def to_arrays(self) -> Dict[str, jnp.ndarray]:
        """
        Convert history lists to JAX arrays for efficient saving.

        Returns:
            Dictionary with 'energies', 'stds', 'acceptance_rates', 'step_sizes' arrays
        """
        return {
            "energies": jnp.array(self.energies),
            "stds": jnp.array(self.stds),
            "acceptance_rates": jnp.array(self.acceptance_rates),
            "step_sizes": jnp.array(self.step_sizes),
        }

    def to_dict(self) -> Dict[str, List[float]]:
        """
        Convert history to a plain dict of lists.

        Returns:
            Dictionary with lists of recorded values
        """
        return {
            "energies": self.energies,
            "stds": self.stds,
            "acceptance_rates": self.acceptance_rates,
            "step_sizes": self.step_sizes,
        }


class StateManager:
    """Manage VMC training state, history tracking, and checkpointing."""

    def __init__(self, checkpoint_path: Path = None):
        """
        Initialize state manager.

        Args:
            checkpoint_path: Directory for saving/loading checkpoints
        """
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else Path("./")
        self.history = TrainingHistory()

    def load_checkpoint(self, state: VMCState) -> VMCState:
        """
        Load checkpoint if it exists, otherwise return current state.

        Args:
            state: Initial VMCState

        Returns:
            Updated VMCState loaded from checkpoint, or original state if no checkpoint
        """
        try:
            return load_checkpoint(
                state,
                path=str(self.checkpoint_path),
                filename="checkpoint.msgpack",
            )
        except Exception as e:
            # No checkpoint found or error loading, return original state
            return state

    def save_checkpoint(self, state: VMCState) -> None:
        """
        Save current training state to checkpoint.

        Args:
            state: VMCState to save
        """
        try:
            save_checkpoint(
                state,
                path=str(self.checkpoint_path),
                filename="checkpoint.msgpack",
            )
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def record_epoch(
        self,
        energy: float,
        std: float,
        acceptance_rate: float,
        step_size: float,
    ) -> None:
        """
        Record metrics from one training epoch.

        Args:
            energy: Energy expectation value
            std: Standard error of energy
            acceptance_rate: MCMC acceptance rate
            step_size: Current MH step size
        """
        self.history.record(energy, std, acceptance_rate, step_size)

    def get_energy_history(self) -> jnp.ndarray:
        """Get array of energy values."""
        return jnp.array(self.history.energies)

    def get_std_history(self) -> jnp.ndarray:
        """Get array of energy standard deviations."""
        return jnp.array(self.history.stds)

    def get_acceptance_rates(self) -> jnp.ndarray:
        """Get array of acceptance rates."""
        return jnp.array(self.history.acceptance_rates)

    def get_step_sizes(self) -> jnp.ndarray:
        """Get array of step sizes."""
        return jnp.array(self.history.step_sizes)

    def get_history_dict(self) -> Dict[str, jnp.ndarray]:
        """Get all history as dictionary of arrays."""
        return self.history.to_arrays()

    @property
    def num_epochs_recorded(self) -> int:
        """Number of epochs recorded so far."""
        return len(self.history.energies)

    @property
    def final_energy(self) -> float:
        """Final energy value, or None if no epochs recorded."""
        return self.history.energies[-1] if self.history.energies else None

    @property
    def min_energy(self) -> float:
        """Minimum energy recorded."""
        return min(self.history.energies) if self.history.energies else float("inf")

    @property
    def avg_acceptance_rate(self) -> float:
        """Average acceptance rate over all recorded epochs."""
        if not self.history.acceptance_rates:
            return 0.0
        return float(jnp.mean(jnp.array(self.history.acceptance_rates)))
