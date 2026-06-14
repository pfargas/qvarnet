"""Gradient diagnostics (roadmap §4.1, §4.2) — host/numpy + small jax reductions.

Gradient norms (global + per-layer) for clipping decisions, and the gradient signal-to-noise
ratio that distinguishes a *converged* plateau (high SNR) from one where the gradient is
*drowned in MC noise* (low SNR) — those need opposite responses (stop vs. more samples).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax


def global_grad_norm(grads) -> float:
    """L2 norm of the full flattened gradient pytree."""
    return float(optax.tree.norm(grads))


def per_layer_grad_norms(grads) -> dict:
    """Per-leaf L2 gradient norms, keyed by pytree path (for layers×epochs heatmaps)."""
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        out[jax.tree_util.keystr(path)] = float(jnp.sqrt(jnp.sum(jnp.asarray(leaf) ** 2)))
    return out


def gradient_snr(log_derivs, e_loc) -> np.ndarray:
    """Per-parameter gradient signal-to-noise ratio over the batch.

    The VMC gradient is itself an MC estimate g_k = 2⟨(E_loc-Ē) O_k⟩. With per-sample
    contributions g_k^(i) = 2(E_loc_i-Ē) O_ik,

        SNR_k = |mean_i g_k^(i)| / (std_i g_k^(i) / sqrt(M)).

    Args:
        log_derivs: O = ∂_θ log|ψ| per sample, shape (M, P).
        e_loc:      local energies, shape (M,).

    Returns:
        (P,) array of SNR per parameter. Track its median over params per epoch.
    """
    o_mat = np.asarray(log_derivs)
    e = np.asarray(e_loc)
    M = o_mat.shape[0]
    g_per = 2.0 * (e - e.mean())[:, None] * o_mat  # (M, P)
    g_mean = g_per.mean(axis=0)
    g_std = g_per.std(axis=0)
    return np.abs(g_mean) / (g_std / np.sqrt(M) + 1e-12)
