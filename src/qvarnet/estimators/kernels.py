"""Numerical kernels for post-training estimators — host/numpy, 1-D.

Standalone copies of the estimator maths (independent of ``qvarnet.observables``
so that module can evolve or disappear without breaking this package). All
functions take raw **lab-coordinate** samples ``(M, N*d)`` drawn from |ψ|².

The OBDM kernels take a ``log_psi`` callable ``(batch, dof) -> (batch,)``
instead of ``(model, params)`` — coordinate handling and parameter baking is
the job of ``TrainedWavefunction``, not the kernels.

Error bars on correlated series come from Flyvbjerg-Petersen blocking.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Blocking error analysis
# ---------------------------------------------------------------------------


def blocking_error(x) -> tuple[float, list]:
    """Flyvbjerg-Petersen blocking error of the mean of a 1-D correlated series.

    Returns ``(error, errors_by_level)``. The reported error is the maximum over
    block levels — the plateau the standard error converges to as correlations
    are averaged out. (For i.i.d. data it is flat ≈ σ/√N from the first level.)
    """
    x = np.asarray(x, dtype=float).ravel()
    errors = []
    while x.shape[0] >= 2:
        n = x.shape[0]
        errors.append(float(np.sqrt(np.var(x, ddof=1) / n)))
        if n % 2:
            x = x[:-1]
        x = 0.5 * (x[0::2] + x[1::2])
    if not errors:
        return 0.0, errors
    return max(errors), errors


def mean_and_error(x) -> tuple[float, float]:
    """Sample mean and blocking error of a 1-D series."""
    x = np.asarray(x, dtype=float).ravel()
    return float(x.mean()), blocking_error(x)[0]


# ---------------------------------------------------------------------------
# Density n(x) and pair correlation
# ---------------------------------------------------------------------------


def density_histogram(samples, n_particles, n_dim=1, bins=60, value_range=None, L=None):
    """Single-particle density n(x), normalised so that ∫ n(x) dx = N.

    samples: ``(M, N*d)`` lab coords. Returns ``(centers, n)``. Currently 1-D.

    For a periodic box pass ``L``: coordinates are folded into ``[0, L)`` before
    histogramming (PBC samplers leave walkers on the covering space), and the
    bin range defaults to ``[0, L)``.
    """
    if n_dim != 1:
        raise NotImplementedError("density_histogram currently supports n_dim=1")
    s = np.asarray(samples)
    M = s.shape[0]
    coords = s.reshape(M, n_particles)
    if L is not None:
        coords = np.mod(coords, L)
        if value_range is None:
            value_range = (0.0, L)
    hist, edges = np.histogram(coords.ravel(), bins=bins, range=value_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    n_x = hist / (M * width)  # ∫ n dx = (M*N)/M = N
    return centers, n_x


def pair_correlation(samples, n_particles, n_dim=1, bins=60, L=None, value_range=None):
    """Pair-distance distribution (→ g(r)) over all i<j pairs.

    For a periodic box pass ``L`` for minimum-image distances (then only
    r ≤ L/2 is meaningful). For trapped systems this is the (unnormalised)
    pair-distance distribution. Returns ``(centers, counts_per_sample)``.
    Currently 1-D.
    """
    if n_dim != 1:
        raise NotImplementedError("pair_correlation currently supports n_dim=1")
    if n_particles < 2:
        raise ValueError("pair_correlation needs at least 2 particles")
    s = np.asarray(samples)
    M = s.shape[0]
    x = s.reshape(M, n_particles)
    i, j = np.triu_indices(n_particles, k=1)
    d = np.abs(x[:, i] - x[:, j])  # (M, n_pairs)
    if L is not None:
        d = np.minimum(d, L - d)
    hist, edges = np.histogram(d.ravel(), bins=bins, range=value_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist / M  # average pair count per sample per bin


def pair_correlation_grid(samples, grid, n_particles, n_dim=1, L=None):
    """Full pair correlation g(x, x′) = ρ₂(x, x′) / (ρ(x)·ρ(x′)) on ``grid`` (1-D).

    Unlike ``pair_correlation`` (which collapses to a distribution of |x_i − x_j|
    and only has a clean g(r) normalisation for periodic/homogeneous systems), this
    keeps both coordinates explicit and divides by the *actual* (possibly
    position-dependent) one-body density at each point, so it's valid for trapped
    systems too. g → 1 (or (N−1)/N exactly, for independent particles) where the
    system is uncorrelated; bins where ρ(x)·ρ(x′) ≈ 0 (empty tails) are ``nan``
    rather than spuriously large.

    For a periodic box pass ``L`` to fold coordinates into ``[0, L)`` first
    (consistent with ``density_histogram``).

    Returns ``(grid, g)``, g symmetric ``(G, G)``.
    """
    if n_dim != 1:
        raise NotImplementedError("pair_correlation_grid currently supports n_dim=1")
    if n_particles < 2:
        raise ValueError("pair_correlation_grid needs at least 2 particles")
    s = np.asarray(samples)
    M = s.shape[0]
    x = s.reshape(M, n_particles)
    if L is not None:
        x = np.mod(x, L)
    grid = np.asarray(grid, dtype=float)
    G = grid.shape[0]
    width = grid[1] - grid[0]
    edges = np.concatenate([grid - width / 2, grid[-1:] + width / 2])

    # ρ₂(x,x′): 2-D histogram of *ordered* pairs (i,j), i≠j. triu_indices gives each
    # unordered pair once; adding the transpose supplies the missing (j,i) term, since
    # histogramming (x_j, x_i) is exactly the mirror image of histogramming (x_i, x_j).
    i, j = np.triu_indices(n_particles, k=1)
    hist2d, _, _ = np.histogram2d(x[:, i].ravel(), x[:, j].ravel(), bins=[edges, edges])
    hist2d = hist2d + hist2d.T
    rho2 = hist2d / (M * width**2)  # ∫∫ ρ₂ dx dx′ = N(N−1)

    _, rho = density_histogram(s, n_particles, n_dim=n_dim, bins=edges, L=L)
    denom = np.outer(rho, rho)
    g = np.divide(rho2, denom, out=np.full_like(rho2, np.nan), where=denom > 1e-12)
    return grid, g


# ---------------------------------------------------------------------------
# Static structure factor S(k)
# ---------------------------------------------------------------------------


def structure_factor(samples, n_particles, k_values, n_dim=1):
    """S(k) = ⟨|ρ_k|²⟩ / N with ρ_k = Σ_j e^{i k x_j}, for k ≠ 0.

    Returns ``(k_values, S)``. Sanity: S(k) → 1 as k → ∞; in a periodic box
    only commensurate k = 2πn/L are valid. Currently 1-D.
    """
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


# ---------------------------------------------------------------------------
# One-body density matrix (McMillan ratio estimator)
# ---------------------------------------------------------------------------


def obdm_grid(log_psi, samples, grid, n_particles, n_dim=1):
    """Estimate ρ₁(x, x′) on ``grid`` (1-D). Returns ``(grid, rho)``, rho symmetric (G, G).

    McMillan ratio estimator: for each particle bin its current position x and,
    for every grid point x′, accumulate ψ(…x′…)/ψ(…x…) = exp(logψ′ − logψ).
    Eigen-decomposition of ρ₁ gives natural orbitals and the condensate
    fraction n₀ = λ_max / Σλ (ODLRO).

    log_psi: callable ``(batch, N*d) -> (batch,)`` of lab coords, everything baked.
    """
    if n_dim != 1:
        raise NotImplementedError("obdm_grid currently supports n_dim=1")
    s = np.asarray(samples)
    M = s.shape[0]
    coords = s.reshape(M, n_particles)
    grid = np.asarray(grid, dtype=float)
    G = grid.shape[0]
    width = grid[1] - grid[0]

    def logpsi(c):
        return np.asarray(log_psi(c)).reshape(-1)

    log0 = logpsi(s)  # (M,)
    rho = np.zeros((G, G))
    for p in range(n_particles):
        idx = np.clip(((coords[:, p] - grid[0]) / width + 0.5).astype(int), 0, G - 1)
        rep = np.repeat(s[:, None, :], G, axis=1)  # (M, G, N)
        rep[:, :, p] = grid[None, :]  # replace particle p by each grid point x′
        log_disp = logpsi(rep.reshape(M * G, n_particles)).reshape(M, G)
        ratio = np.exp(log_disp - log0[:, None])  # (M, G)
        np.add.at(rho, idx, ratio)  # rho[bin(x), :] += ratio
    rho /= M * width
    return grid, 0.5 * (rho + rho.T)


def natural_orbitals(rho, grid):
    """Diagonalise the discretised ρ₁ → (occupations desc, orbitals)."""
    width = float(grid[1] - grid[0])
    vals, vecs = np.linalg.eigh(rho * width)  # kernel of the integral operator
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def condensate_fraction(rho, grid):
    """n₀ = largest natural occupation / Σ occupations (= λ_max / N for normalised ρ₁)."""
    vals, _ = natural_orbitals(rho, grid)
    vals = np.clip(vals, 0.0, None)
    return float(vals[0] / (vals.sum() + 1e-12))


def obdm_displacement(log_psi, samples, displacements, particle=0, n_dim=1):
    """Translationally-averaged ρ₁(Δ) = ⟨ψ(x+Δ,…)/ψ(x,…)⟩ for a homogeneous system.

    ρ₁(0) = 1; the large-Δ plateau is the condensate fraction (ODLRO). Returns
    ``(displacements, rho1)``. Currently 1-D, displacing a single ``particle``.

    log_psi: callable ``(batch, N*d) -> (batch,)`` of lab coords, everything baked.
    """
    if n_dim != 1:
        raise NotImplementedError("obdm_displacement currently supports n_dim=1")
    s = np.asarray(samples)
    M, _ = s.shape
    deltas = np.asarray(displacements, dtype=float)

    def logpsi(c):
        return np.asarray(log_psi(c)).reshape(-1)

    log0 = logpsi(s)  # (M,)
    out = np.empty(deltas.shape[0])
    for di, d in enumerate(deltas):
        disp = s.copy()
        disp[:, particle] = disp[:, particle] + d
        out[di] = np.mean(np.exp(logpsi(disp) - log0))
    return deltas, out
