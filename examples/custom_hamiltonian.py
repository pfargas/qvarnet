"""Minimal example: define a new Hamiltonian and register it.

Adding a new physical system to qvarnet requires three things:
  1. Subclass ContinuousHamiltonian
  2. Implement potential_energy(samples) → (batch,)
  3. Decorate with @register_hamiltonian("my-name") and @struct.dataclass

potential_energy always receives lab coordinates regardless of which
CoordMode is active — the transform is handled by ContinuousHamiltonian.
"""

import tempfile

import jax.numpy as jnp
import optax
from flax import struct

from qvarnet.config.coord_mode import LabCoords
from qvarnet.config.training_setup import TrainingConfig
from qvarnet.hamiltonian.continuous import ContinuousHamiltonian
from qvarnet.hamiltonian.hamiltonian_registry import register_hamiltonian
from qvarnet.models.mlp import MLP
from qvarnet.train import train

# ---------------------------------------------------------------------------
# Step 1–3: define and register the Hamiltonian
# ---------------------------------------------------------------------------

@register_hamiltonian("double-well")
@struct.dataclass
class DoubleWellHamiltonian(ContinuousHamiltonian):
    """1-D double-well: V(x) = a·x⁴ − b·x².

    Minima at x = ±sqrt(b / 2a).  With a=1, b=4: minima at x = ±√2.
    """

    a: float = 1.0
    b: float = 4.0

    def potential_energy(self, samples):
        # samples: (batch, dof)  — lab coordinates, always
        # return:  (batch,)
        x2 = jnp.sum(samples ** 2, axis=-1)
        x4 = jnp.sum(samples ** 4, axis=-1)
        return self.a * x4 - self.b * x2


# ---------------------------------------------------------------------------
# Use it in a short VMC run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    hamiltonian = DoubleWellHamiltonian(a=1.0, b=4.0)
    model = MLP(architecture=[1, 16, 16, 1])

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train(
            shape=(256, 1),
            model=model,
            optimizer=optax.adam(3e-3),
            hamiltonian=hamiltonian,
            training_config=TrainingConfig(
                n_epochs=500,
                checkpoint_path=tmpdir,
                save_checkpoints=False,
                rng_seed=0,
            ),
            sampler_params={
                "step_size": 0.3,
                "chain_length": 200,
                "thermalization_steps": 20,
            },
            coord_mode=LabCoords(),
        )

    best = result.best(n=1)[0]
    print(f"Best energy: {float(best.energy):.6f}")
