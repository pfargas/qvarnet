"""Static structure factor S(k) (roadmap §6.2) — host/numpy, 1-D.

    ρ_k = Σ_j e^{i k x_j},   S(k) = ⟨|ρ_k|²⟩ / N   (k ≠ 0).

Sanity checks: S(k) → 1 as k → ∞; for a periodic box only commensurate k = 2π n / L are valid.
"""

import numpy as np


def structure_factor(samples, n_particles, k_values, n_dim=1):
    """S(k) for a list of wavevectors ``k_values``. Returns ``(k_values, S)``. Currently 1-D."""
    if n_dim != 1:
        raise NotImplementedError("structure_factor currently supports n_dim=1")
    s = np.asarray(samples)
    M = s.shape[0]
    x = s.reshape(M, n_particles)
    k = np.asarray(k_values, dtype=float)
    phases = np.exp(1j * x[:, :, None] * k[None, None, :])  # (M, N, K)
    rho_k = phases.sum(axis=1)  # (M, K)
    S = (np.abs(rho_k) ** 2).mean(axis=0) / n_particles
    return k, S.real


def commensurate_k(L, n_max):
    """Allowed wavevectors k = 2π n / L for n = 1..n_max in a periodic box of size L."""
    return 2.0 * np.pi * np.arange(1, n_max + 1) / L
