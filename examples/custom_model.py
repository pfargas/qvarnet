"""Minimal example: define your own ansatz and use it.

Adding a new ansatz to qvarnet takes two steps:
  1. Subclass ``flax.linen.Module``
  2. Implement ``__call__(x) -> (batch, 1)``, returning **log|psi(x)|**

That is the whole contract. There is nothing to register and no name to invent --
you construct the object and pass it to ``train()``, exactly as you pass an optax
optimizer. (Earlier versions needed a ``@register_model`` decorator and a
``from_config`` classmethod so a registry could rebuild the model from a string;
both are gone.)

Run it:

    uv run python examples/custom_model.py
"""

import tempfile

import flax.linen as nn
import jax.numpy as jnp
import optax

from qvarnet import train
from qvarnet.config.coord_mode import LabCoords
from qvarnet.config.training_setup import SamplingConfig, TrainingConfig
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.utils.checkpoint import load_run


class GaussianAnsatz(nn.Module):
    """Log-Gaussian ansatz: log|psi(x)| = -alpha * sum_i x_i^2.

    One learnable parameter alpha (initialised to 1). The exact ground state of a
    1-D harmonic oscillator (omega=1) has alpha = 0.5, so VMC should converge there.
    """

    @nn.compact
    def __call__(self, x):
        # x: (batch, dof)  ->  (batch, 1), the log-amplitude
        alpha = self.param("alpha", nn.initializers.ones, (1,))
        return -jnp.abs(alpha) * jnp.sum(x**2, axis=-1, keepdims=True)


if __name__ == "__main__":
    model = GaussianAnsatz()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train(
            shape=(256, 1),
            model=model,
            optimizer=optax.adam(1e-2),
            hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
            training_config=TrainingConfig(
                n_epochs=300,
                checkpoint_path=tmpdir,
                save_checkpoints=True,
                rng_seed=0,
            ),
            sampler_params=SamplingConfig(
                step_size=0.5,
                chain_length=100,
                thermalization_steps=10,
                thinning_factor=5,
            ),
            coord_mode=LabCoords(),
            model_name="gaussian-ansatz",  # provenance label only
            model_args={},
        )

        best = result.best(n=1)[0]
        print(f"Best energy: {float(best.energy):.6f}  (exact: 0.5)")

        # Reloading needs the ansatz object: construct the same one you trained.
        run = load_run(tmpdir, GaussianAnsatz())
        alpha_val = float(run.params["params"]["alpha"].squeeze())
        print(f"Loaded alpha = {alpha_val:.4f}  (exact: 0.5000)")
