"""One-body density matrix ρ₁(x, x′) (roadmap §6.3) — host/numpy + jax model evals, 1-D.

McMillan ratio estimator: from samples R ~ |ψ|², for each particle bin its current position x
into the grid and, for every grid point x′, accumulate the wavefunction ratio
ψ(…x′…)/ψ(…x…) = exp(logψ(replace) − logψ(R)). Averaging gives ρ₁ on the grid, whose
eigen-decomposition yields the natural orbitals and the condensate fraction n₀ = λ_max / Σλ
(off-diagonal long-range order).

Validation target: a single particle in a harmonic trap has a rank-1 ρ₁ → n₀ = 1.
"""

import numpy as np


def obdm_grid(model, params, samples, grid, n_particles, n_dim=1):
    """Estimate ρ₁(x, x′) on ``grid`` (1-D). Returns ``(grid, rho)`` with rho symmetric (G, G)."""
    if n_dim != 1:
        raise NotImplementedError("obdm_grid currently supports n_dim=1")
    s = np.asarray(samples)
    M = s.shape[0]
    coords = s.reshape(M, n_particles)
    grid = np.asarray(grid, dtype=float)
    G = grid.shape[0]
    width = grid[1] - grid[0]

    def logpsi(c):
        return np.asarray(model.apply(params, c)).reshape(-1)

    log0 = logpsi(s)  # (M,)
    rho = np.zeros((G, G))
    for p in range(n_particles):
        idx = np.clip(((coords[:, p] - grid[0]) / width + 0.5).astype(int), 0, G - 1)  # bin of x
        rep = np.repeat(s[:, None, :], G, axis=1)  # (M, G, N)
        rep[:, :, p] = grid[None, :]  # replace particle p by each grid point x′
        log_disp = logpsi(rep.reshape(M * G, n_particles)).reshape(M, G)
        ratio = np.exp(log_disp - log0[:, None])  # (M, G)
        np.add.at(rho, idx, ratio)  # rho[bin(x), :] += ratio
    rho /= M * width
    return grid, 0.5 * (rho + rho.T)


def natural_orbitals(rho, grid):
    """Diagonalise the discretised ρ₁ → (occupations desc, orbitals). λ are natural occupations."""
    width = float(grid[1] - grid[0])
    vals, vecs = np.linalg.eigh(rho * width)  # kernel of the integral operator
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def condensate_fraction(rho, grid):
    """n₀ = largest natural occupation / Σ occupations (= λ_max / N for a normalised ρ₁)."""
    vals, _ = natural_orbitals(rho, grid)
    vals = np.clip(vals, 0.0, None)
    return float(vals[0] / (vals.sum() + 1e-12))


def obdm_displacement(model, params, samples, displacements, particle=0, n_dim=1):
    """Translationally-averaged ρ₁(Δ) = ⟨ψ(x+Δ,…)/ψ(x,…)⟩ for a homogeneous system.

    ρ₁(0) = 1; its large-Δ plateau is the condensate fraction (ODLRO). Returns
    ``(displacements, rho1)``. Currently 1-D, displacing a single ``particle``.
    """
    if n_dim != 1:
        raise NotImplementedError("obdm_displacement currently supports n_dim=1")
    s = np.asarray(samples)
    M, N = s.shape
    deltas = np.asarray(displacements, dtype=float)

    def logpsi(c):
        return np.asarray(model.apply(params, c)).reshape(-1)

    log0 = logpsi(s)  # (M,)
    out = np.empty(deltas.shape[0])
    for di, d in enumerate(deltas):
        disp = s.copy()
        disp[:, particle] = disp[:, particle] + d
        out[di] = np.mean(np.exp(logpsi(disp) - log0))
    return deltas, out
