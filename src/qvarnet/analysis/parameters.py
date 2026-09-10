"""Parameter-update diagnostics (roadmap §4.3) — host/numpy + small jax reductions.

The relative per-step update magnitude r = ‖θ_{t+1}-θ_t‖ / (‖θ_t‖+ε):
- healthy training: r ~ 1e-3–1e-2;
- **dead region**: r → 0 (a layer stopped learning — with Adam, often v_t exploded);
- instability: r ≳ 1e-1.

Per-layer values plotted as a layers×epochs heatmap localise dead regions; this cross-validates
the QGT-spectrum null space (``qgt_spectrum``).
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def global_theta_ratio(old_params, new_params, eps: float = 1e-8) -> float:
    """‖Δθ‖ / (‖θ‖+ε) over the whole flattened parameter vector."""
    o = ravel_pytree(old_params)[0]
    n = ravel_pytree(new_params)[0]
    return float(jnp.linalg.norm(n - o) / (jnp.linalg.norm(o) + eps))


def theta_ratios(old_params, new_params, eps: float = 1e-8) -> dict:
    """Per-leaf relative update ‖Δθ^(ℓ)‖/(‖θ^(ℓ)‖+ε), keyed by pytree path."""
    out = {}
    old_leaves = dict(
        (jax.tree_util.keystr(p), leaf)
        for p, leaf in jax.tree_util.tree_leaves_with_path(old_params)
    )
    new_leaves = dict(
        (jax.tree_util.keystr(p), leaf)
        for p, leaf in jax.tree_util.tree_leaves_with_path(new_params)
    )
    for key, o in old_leaves.items():
        n = new_leaves[key]
        o = jnp.asarray(o).ravel()
        d = jnp.asarray(n).ravel() - o
        out[key] = float(jnp.linalg.norm(d) / (jnp.linalg.norm(o) + eps))
    return out


def dead_fraction(old_params, new_params, delta: float = 1e-6, eps: float = 1e-8) -> float:
    """Fraction of parameters whose relative update |Δθ_k|/(|θ_k|+ε) < delta (≈ frozen)."""
    o = ravel_pytree(old_params)[0]
    n = ravel_pytree(new_params)[0]
    rel = jnp.abs(n - o) / (jnp.abs(o) + eps)
    return float(jnp.mean(rel < delta))
