import warnings
from typing import Any

from flax import linen as nn

from qvarnet.physics.boundaries import NoBoundary, PeriodicBoundary


class LogWavefunction(nn.Module):
    """Composable log-wavefunction.

    log|psi(x)| = network(transform(x)) [+ envelope(x)] [+ jastrow(x)]

    All active components output ``(..., 1)`` and are summed in log space. The
    envelope and Jastrow see the *raw* coordinates, so their physics stays tied to
    real-space geometry.

    Args:
        network: maps encoded coordinates to ``(..., 1)``. An MLP takes flat
            ``(..., encoded_dim)``; a DeepSet takes ``(..., n_particles, ppd)``, so
            set ``n_particles`` below for it.
        transform: ``x -> x_encoded``; typically NoBoundary() or PeriodicBoundary(L).
        n_particles: when set, the encoded input is reshaped from flat ``(..., N*ppd)``
            to ``(..., N, ppd)``. Required for DeepSet, leave None for MLP.
        n_dim: spatial dimension; stored for documentation, not used in the reshape.
        envelope: optional log-space envelope on raw x (not valid on a ring).
        jastrow: optional Jastrow factor on raw x.
    """

    network: nn.Module
    transform: Any = None  # NoBoundary() or PeriodicBoundary(L) — not a JAX array
    n_particles: int = None
    n_dim: int = None
    envelope: Any = None
    jastrow: Any = None

    @nn.compact
    def __call__(self, x):
        # A local, not self.transform = ...: a Flax module is frozen outside setup(),
        # so assigning here raised SetAttributeFrozenModuleError for every caller who
        # left transform at its default.
        transform = self.transform if self.transform is not None else NoBoundary()
        if isinstance(transform, PeriodicBoundary) and self.envelope is not None:
            # A confining envelope (e.g. Gaussian) breaks L-periodicity of log|ψ|:
            # there is no trap on a ring. The envelope is applied to *raw* x and is
            # not periodic, so it silently corrupts the PBC wavefunction.
            warnings.warn(
                "LogWavefunction has a PeriodicBoundary transform but a non-None "
                "envelope. The envelope is applied to raw coordinates and is not "
                "L-periodic, breaking periodicity of log|ψ|. Set envelope=None for "
                "periodic systems (use a periodic Jastrow for interactions instead).",
                stacklevel=2,
            )
        x_enc = transform(x)
        if self.n_particles is not None:
            ppd = x_enc.shape[-1] // self.n_particles
            x_for_net = x_enc.reshape(*x_enc.shape[:-1], self.n_particles, ppd)
        else:
            x_for_net = x_enc
        log_psi = self.network(x_for_net)
        if self.envelope is not None:
            log_psi = log_psi + self.envelope(x)
        if self.jastrow is not None:
            log_psi = log_psi + self.jastrow(x)
        return log_psi  # (..., 1)
