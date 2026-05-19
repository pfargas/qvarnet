"""Cusp condition utilities: fixed near-coalescence configuration generation."""

import numpy as np
import jax.numpy as jnp


def make_cusp_configs(n_particles, L, epsilon, n_configs_per_pair, rng_seed=0):
    """Generate fixed near-coalescence configurations for all particle pairs.

    For each pair (i,j), generates n_configs_per_pair configs where
    x_j = c, x_i = (c + epsilon) % L for random centers c, other
    particles at random positions in [0, L).

    Args:
        n_particles: number of particles N
        L: system size (ring circumference)
        epsilon: coalescence separation (should be << L/N)
        n_configs_per_pair: M configs per pair
        rng_seed: numpy seed for reproducibility

    Returns:
        Array of shape (n_pairs * n_configs_per_pair, n_particles).
        n_pairs = N*(N-1)//2
    """
    rng = np.random.default_rng(rng_seed)
    pairs = [(i, j) for i in range(n_particles) for j in range(i + 1, n_particles)]
    configs = []

    for i, j in pairs:
        for _ in range(n_configs_per_pair):
            x = rng.uniform(0.0, L, size=n_particles)
            c = rng.uniform(0.0, L)
            x[j] = c
            x[i] = (c + epsilon) % L
            configs.append(x.copy())

    return jnp.array(np.stack(configs), dtype=jnp.float32)


def make_cusp_pair_indices(n_particles, n_configs_per_pair):
    """Return (pair_i, pair_j) index arrays matching make_cusp_configs output order.

    Each cusp config at position k tests the pair (pair_i[k], pair_j[k]).
    """
    pairs = [(i, j) for i in range(n_particles) for j in range(i + 1, n_particles)]
    pair_i, pair_j = [], []
    for i, j in pairs:
        for _ in range(n_configs_per_pair):
            pair_i.append(i)
            pair_j.append(j)
    return jnp.array(pair_i, dtype=jnp.int32), jnp.array(pair_j, dtype=jnp.int32)
