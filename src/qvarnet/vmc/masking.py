"""Weight masking via optax wrappers (roadmap §5.1, PARAMETER_EFFICIENCY.md §2/§5).

Pruning and "random weight switch-off during training" are the same ten lines with different
mask sources — both mask the *updates* (never the forward pass), so |ψ_θ|² is exact and
consistent between sampling and gradient evaluation (unbiased), unlike forward-pass DropConnect.

- ``mask_updates(tx, mask)``        — freeze a fixed set of weights (fine-tune after pruning).
- ``magnitude_mask(params, q)``     — keep the largest (1−q) fraction of weights by |value|.
- ``random_update_masking(tx, p)``  — Bernoulli-drop a fraction p of *updates* each step,
                                      forcing the network to spread information (§5.1).
"""

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree


def mask_updates(tx: optax.GradientTransformation, mask) -> optax.GradientTransformation:
    """Wrap ``tx`` so updates are multiplied by a fixed ``mask`` pytree (same structure)."""

    def update(updates, state, params=None):
        updates, state = tx.update(updates, state, params)
        updates = jax.tree_util.tree_map(lambda g, m: g * m, updates, mask)
        return updates, state

    return optax.GradientTransformation(tx.init, update)


def magnitude_mask(params, sparsity: float):
    """Binary mask keeping the largest ``(1 - sparsity)`` fraction of weights by |value|
    (global threshold). ``sparsity`` in [0, 1)."""
    flat, _ = ravel_pytree(params)
    k = int(sparsity * flat.size)
    thr = -1.0 if k <= 0 else float(jnp.sort(jnp.abs(flat))[k - 1])
    return jax.tree_util.tree_map(lambda p: (jnp.abs(p) > thr).astype(p.dtype), params)


def random_update_masking(
    tx: optax.GradientTransformation, drop_prob: float, seed: int = 0
) -> optax.GradientTransformation:
    """Wrap ``tx`` to zero a random Bernoulli fraction ``drop_prob`` of updates each step.

    The wavefunction at every step is exact and consistent (the mask touches only the optax
    update), so the VMC estimator stays unbiased — this is the safe form of §5.1.
    """

    def init(params):
        return {"inner": tx.init(params), "key": jax.random.PRNGKey(seed)}

    def update(updates, state, params=None):
        inner_updates, inner_state = tx.update(updates, state["inner"], params)
        key, sub = jax.random.split(state["key"])
        leaves, treedef = jax.tree_util.tree_flatten(inner_updates)
        keys = jax.random.split(sub, len(leaves))
        masked = [
            g * jax.random.bernoulli(k, 1.0 - drop_prob, g.shape).astype(g.dtype)
            for g, k in zip(leaves, keys)
        ]
        return jax.tree_util.tree_unflatten(treedef, masked), {"inner": inner_state, "key": key}

    return optax.GradientTransformation(init, update)
