"""Input transformation layers — drop-in first layers for wavefunction models.

These layers reshape or augment the raw coordinate input before passing it to
the main network. They are all square or expanding transforms that compose
cleanly with any downstream nn.Module.
"""

import jax.numpy as jnp
from flax import linen as nn


class SubtractCM(nn.Module):
    """Subtract the centre-of-mass from particle coordinates.

    Input:  (..., n_particles * n_dim)
    Output: (..., n_particles * n_dim)   — same shape, CM-subtracted

    Use as the first layer whenever you want translational invariance baked
    into the model rather than handled externally.
    """

    n_particles: int
    n_dim: int

    @nn.compact
    def __call__(self, x):
        shape = x.shape[:-1]
        r = x.reshape(*shape, self.n_particles, self.n_dim)   # (..., N, d)
        cm = r.mean(axis=-2, keepdims=True)                   # (..., 1, d)
        return (r - cm).reshape(*shape, self.n_particles * self.n_dim)


class AppendPairwiseDiffs(nn.Module):
    """Augment coordinates with all pairwise differences.

    Input:  (..., n_particles * n_dim)
    Output: (..., n_particles * n_dim + n_pairs * n_dim)
            where n_pairs = n_particles * (n_particles - 1) // 2

    Useful when the model should be aware of relative distances without
    committing to a specific permutation-invariant architecture.
    """

    n_particles: int
    n_dim: int

    @nn.compact
    def __call__(self, x):
        shape = x.shape[:-1]
        r = x.reshape(*shape, self.n_particles, self.n_dim)   # (..., N, d)
        i_idx, j_idx = jnp.triu_indices(self.n_particles, k=1)
        diffs = r[..., i_idx, :] - r[..., j_idx, :]          # (..., n_pairs, d)
        diffs_flat = diffs.reshape(*shape, -1)                 # (..., n_pairs * d)
        return jnp.concatenate([x, diffs_flat], axis=-1)
