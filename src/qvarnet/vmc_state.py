from flax.training import train_state
import jax.numpy as jnp
from flax import struct


class VMCState(train_state.TrainState):
    """Training state for Variational Monte Carlo, extending Flax's TrainState.

    Augments the standard optimizer state with VMC-specific diagnostics recorded
    at every training step:

    - ``acceptance_rate`` — Mean MH acceptance rate across chains.
    - ``energy`` — Energy expectation value :math:`\\langle E \\rangle`.
    - ``std`` — Standard error of the local energy estimator.
    - ``energy_num`` — Energy from numerical gradients (for debugging).
    - ``std_num`` — Standard error of the numerical-gradient energy.
    - ``step_size`` — Current MH proposal step size.
    - ``grads`` — Parameter gradients :math:`\\nabla_\\theta \\mathcal{L}`.
    """

    acceptance_rate: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(0.0))
    energy: float = float("inf")
    std: float = float("inf")
    energy_num: float = float("inf")
    std_num: float = float("inf")
    step_size: float = float("inf")
    grads: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
