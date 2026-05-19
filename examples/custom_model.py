"""Minimal example: define a new model and register it.

Adding a new ansatz to qvarnet requires:
  1. Subclass BaseModel (a Flax nn.Module)
  2. Implement __call__(x) → (batch, 1)  — output is log|ψ(x)|
  3. Implement from_config(config) so load_run() can reconstruct it
  4. Decorate with @register_model("my-name")

The model must output log|ψ|, not ψ directly.
"""

import tempfile

import flax.linen as nn
import jax.numpy as jnp
import optax

from qvarnet.config.coord_mode import LabCoords
from qvarnet.config.training_setup import TrainingConfig
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.models.base import BaseModel
from qvarnet.models.registry import register_model
from qvarnet.train import train
from qvarnet.utils.checkpoint import load_run

# ---------------------------------------------------------------------------
# Step 1–4: define and register the model
# ---------------------------------------------------------------------------

@register_model("gaussian-ansatz")
class GaussianAnsatz(BaseModel):
    """Log-Gaussian ansatz: log|ψ(x)| = −α Σ xᵢ².

    One learnable parameter α (initialised to 1).
    The exact ground state of a 1-D harmonic oscillator (ω=1) has α = 0.5.
    VMC should converge toward this value.
    """

    @nn.compact
    def __call__(self, x):
        # x: (batch, dof)
        # output: (batch, 1)  — log|ψ|
        alpha = self.param("alpha", nn.initializers.ones, (1,))
        return (-jnp.abs(alpha) * jnp.sum(x ** 2, axis=-1, keepdims=True))

    @classmethod
    def from_config(cls, config: dict):
        return cls()


# ---------------------------------------------------------------------------
# Use it in a short VMC run, then reload from disk
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    hamiltonian = HarmonicOscillatorHamiltonian(omega=1.0)
    model = GaussianAnsatz()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train(
            shape=(256, 1),
            model=model,
            optimizer=optax.adam(1e-2),
            hamiltonian=hamiltonian,
            training_config=TrainingConfig(
                n_epochs=300,
                checkpoint_path=tmpdir,
                save_checkpoints=True,
                rng_seed=0,
            ),
            sampler_params={
                "step_size": 0.5,
                "chain_length": 100,
                "thermalization_steps": 10,
            },
            coord_mode=LabCoords(),
            model_name="gaussian-ansatz",
            model_args={},
        )

        best = result.best(n=1)[0]
        print(f"Best energy: {float(best.energy):.6f}  (exact: 0.5)")

        # Reload from disk — no manual archaeology
        run = load_run(tmpdir)
        alpha_val = float(run.params["params"]["alpha"].squeeze())
        print(f"Loaded α = {alpha_val:.4f}  (exact: 0.5000)")
