"""Density n(x) and pair correlation g(r) (roadmap §6.1) — host/numpy, 1-D.

Both are histograms over particle coordinates from samples R ~ |ψ|².
"""

import numpy as np


def density_histogram(samples, n_particles, n_dim=1, bins=60, value_range=None):
    """Single-particle density n(x), normalised so that ∫ n(x) dx = N.

    Check this actually is correct. n(x) = int dx2dx3...dxN psi(x1, x2, ...)

    samples: ``(M, N*d)``. Returns ``(centers, n)``. Currently 1-D (d=1).
    """
    if n_dim != 1:
        raise NotImplementedError("density_histogram currently supports n_dim=1")
    s = np.asarray(samples)
    M = s.shape[0]
    coords = s.reshape(M, n_particles)
    hist, edges = np.histogram(coords.ravel(), bins=bins, range=value_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    n_x = hist / (M * width)  # ∫ n dx = (M*N)/M = N
    return centers, n_x


def pair_correlation(samples, n_particles, n_dim=1, bins=60, L=None, value_range=None):
    """Pair-distance distribution (→ g(r)) over all i<j pairs.

    For a periodic box pass ``L`` for minimum-image distances (then only r ≤ L/2 is meaningful).
    For trapped systems this is the (unnormalised) pair-distance distribution. Returns
    ``(centers, counts_per_sample)``. Currently 1-D.
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
