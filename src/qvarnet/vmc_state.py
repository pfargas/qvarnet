from flax.training import train_state
import jax.numpy as jnp
from flax import struct


class VMCState(train_state.TrainState):
    """Training state for Variational Monte Carlo, extending Flax's TrainState.

    Fields recorded at every training step:

    - ``acceptance_rate`` — Mean MH acceptance rate across chains.
    - ``energy`` — Energy expectation value ⟨E⟩.
    - ``std`` — Standard error of the local energy estimator.
    - ``step_size`` — Current MH proposal step size.
    - ``grads`` — Parameter gradients ∇_θ L.
    - ``cm_mean`` — Mean centre-of-mass across chains (diagnostic).
    - ``cm_std`` — Std of centre-of-mass across chains (diagnostic).
    """

    acceptance_rate: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(0.0))
    energy: float = 0.0
    std: float = 0.0
    step_size: float = 0.0
    grads: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    cm_mean: float = 0.0
    cm_std: float = 0.0
