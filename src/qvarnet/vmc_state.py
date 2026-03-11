from flax.training import train_state


class VMCState(train_state.TrainState):
    """Train state for VMC training, extending Flax's TrainState."""

    energy: float = float("inf")
    std: float = float("inf")
