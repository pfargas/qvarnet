"""Pytest config: pin to single-thread CPU so tests are fast and deterministic
(GPU atomic reductions are non-deterministic). Must run before JAX is imported."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

from flax import linen as nn  # noqa: E402

from qvarnet.boundaries import NoBoundary  # noqa: E402
from qvarnet.models.compose import LogWavefunction  # noqa: E402
from qvarnet.models.envelopes import GaussianEnvelope  # noqa: E402
from qvarnet.models.mlp import MLP  # noqa: E402


def make_ho_model():
    """Small log-wavefunction (MLP + Gaussian envelope) for harmonic-oscillator tests.

    The envelope is required for a *normalizable* |ψ|²: a bare tanh-MLP saturates to a
    constant log|ψ| at large |x|, so its |ψ|² has flat tails and warm-walker chains
    random-walk to infinity (energies diverge monotonically). The envelope makes the
    model physical under both warm and cold walker modes.
    """
    return LogWavefunction(
        network=MLP(hidden=[16, 16], output_dim=1, hidden_activation=nn.tanh),
        envelope=GaussianEnvelope(),
        transform=NoBoundary(),
    )
