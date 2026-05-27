from typing import Any

from flax import linen as nn


class LogWavefunction(nn.Module):
    """Composable log-wavefunction.

    log|ψ(x)| = network(transform(x))  [+  envelope(x)]  [+  jastrow(x)]

    All active components output ``(..., 1)`` and are summed in log space.
    The raw (pre-transform) coordinates ``x`` are passed to the envelope and
    Jastrow so their physics is tied to the real-space geometry.

    Parameters
    ----------
    transform:
        Any callable ``x → x_encoded``.  Typically ``NoBoundary()`` or
        ``PeriodicBoundary(L)``.
    network:
        Any ``nn.Module`` mapping encoded coordinates to ``(..., 1)``.
        ``MLP``: receives flat ``(..., encoded_dim)`` — leave ``n_particles=None``.
        ``DeepSet``: receives ``(..., n_particles, per_particle_dim)`` —
        set ``n_particles`` (and optionally ``n_dim``) here.
    n_particles:
        Number of particles.  When set, ``LogWavefunction`` reshapes the
        encoded input from flat ``(..., N*ppd)`` to ``(..., N, ppd)`` before
        passing to ``network``.  Required for ``DeepSet``; leave ``None`` for ``MLP``.
    n_dim:
        Spatial dimension (informational; stored for documentation / validation).
        Not used in the reshape — ``ppd`` is always inferred as
        ``x_enc.shape[-1] // n_particles``.
    envelope:
        Optional log-space envelope applied to *raw* ``x``.
    jastrow:
        Optional Jastrow factor applied to *raw* ``x``.
    """

    transform: Any
    network: nn.Module
    n_particles: int = None
    n_dim: int = None
    envelope: Any = None
    jastrow: Any = None

    @nn.compact
    def __call__(self, x):
        x_enc = self.transform(x)
        if self.n_particles is not None:
            ppd = x_enc.shape[-1] // self.n_particles
            # assert (
            #     ppd == self.n_dim
            # ), f"Expected ppd={self.n_dim} but got {ppd} from transform output shape {x_enc.shape}"
            x_for_net = x_enc.reshape(*x_enc.shape[:-1], self.n_particles, ppd)
        else:
            x_for_net = x_enc
        log_psi = self.network(x_for_net)
        if self.envelope is not None:
            log_psi = log_psi + self.envelope(x)
        if self.jastrow is not None:
            log_psi = log_psi + self.jastrow(x)
        return log_psi  # (..., 1)
