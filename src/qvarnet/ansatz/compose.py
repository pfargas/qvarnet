import warnings
from typing import Any

import jax.numpy as jnp
from flax import linen as nn

from qvarnet.physics.boundaries import NoBoundary, PeriodicBoundary


class LogWavefunction(nn.Module):
    """Composable log-wavefunction.

    log|psi(x)| = network(transform(x)) [+ envelope(x)] [+ jastrow(x)]
    phase(x)    = network phase, or 0

    Returns ``(log_abs, phase)``, each ``(...,)`` (see ``ansatz/output.py``).
    Envelope and Jastrow output ``(..., 1)`` and add to log|ψ| only. The network
    outputs either ``(..., 1)`` (bosonic: phase 0) or its own ``(log_abs, phase)``
    pair (e.g. a Slater determinant). The envelope and Jastrow see the *raw*
    coordinates, so their physics stays tied to real-space geometry.

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
        log_abs, phase = self._network_out(x)
        if self.envelope is not None:
            log_abs = log_abs + self.envelope(x).reshape(log_abs.shape)  # (..., 1) or (...,)
        if self.jastrow is not None:
            log_abs = log_abs + self.jastrow(x).reshape(log_abs.shape)
        return log_abs, phase  # (...,), (...,)

    def _network_out(self, x):
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
        net_out = self.network(x_for_net)
        if isinstance(net_out, tuple):
            log_abs, phase = net_out  # (...,) each
        else:
            log_abs = net_out[..., 0]  # (..., 1) -> (...,)
            phase = jnp.zeros_like(log_abs)
        return log_abs, phase

    def analytic_parts(self):
        """The envelope/Jastrow modules with a closed-form ``derivs`` method."""
        return [m for m in (self.envelope, self.jastrow) if m is not None and hasattr(m, "derivs")]

    def log_abs_ad(self, x):
        """log|ψ| without the parts that have closed-form ``derivs``, (...,): what autodiff must handle."""
        log_abs = self._network_out(x)[0]
        for m in (self.envelope, self.jastrow):
            if m is not None and not hasattr(m, "derivs"):
                log_abs = log_abs + m(x).reshape(log_abs.shape)
        return log_abs

    def analytic_derivs(self, x):
        """Sum of the closed-form parts' (∂_k log|ψ|, ∂²_k log|ψ|), each (..., dof); zeros if none."""
        g, d2 = jnp.zeros_like(x), jnp.zeros_like(x)
        for m in self.analytic_parts():
            gm, d2m = m.derivs(x)
            g, d2 = g + gm, d2 + d2m
        return g, d2
