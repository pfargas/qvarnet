from flax.training import train_state
import jax.numpy as jnp
from flax import struct


class VMCState(train_state.TrainState):
    """Train state for VMC training, extending Flax's TrainState."""

    acceptance_rate: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(0.0))
    energy: float = float("inf")
    std: float = float("inf")
