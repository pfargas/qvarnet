"""Laplacian implementations for kinetic energy computation.

All functions share the signature:
    laplacian_fn(fn, xs, key=None, weights=None) -> (batch,)

where fn: (dof,) -> scalar is the function to differentiate (log|ψ|),
xs: (batch, dof) is the batch of configurations, key is a JAX PRNGKey
(only used by laplacian_hutchinson; ignored by the exact methods), and
weights: (dof,) are optional per-dof factors — the estimators then return the
weighted trace Σ_k w_k ∂²_k fn (mass-imbalanced kinetic energy, w_k = 1/m_k).
weights=None is the plain Laplacian.

The folx (forward-Laplacian) method does not live here: it produces gradient and
Laplacian together, so it replaces the whole kinetic computation — see kinetic.py.
"""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["fn"])
def laplacian_forward_ad(fn, xs, key=None, weights=None):
    """Δf (or Σ w_k ∂²_k f) via forward-over-reverse AD (exact).

    Computes diagonal of the Hessian without materialising it.
    Cost: O(dof) JVPs.  Memory: O(dof).

    xs:  (batch, dof)
    key: ignored (exists for interface uniformity)
    returns: (batch,)
    """
    n = xs.shape[-1]
    w = jnp.ones(n) if weights is None else weights

    def laplacian_single(x):
        def body(i, acc):
            e_i = jnp.zeros(n).at[i].set(1.0)
            _, hess_col = jax.jvp(jax.grad(fn), (x,), (e_i,))
            return acc + w[i] * hess_col[i]

        return jax.lax.fori_loop(0, n, body, 0.0)

    return jax.vmap(laplacian_single)(xs)


@partial(jax.jit, static_argnames=["fn"])
def laplacian_full_hessian(fn, xs, key=None, weights=None):
    """Δf (or Σ w_k ∂²_k f) via full Hessian diagonal (exact, debug only).

    Cost: O(dof) backward passes.  Memory: O(dof²).

    xs:  (batch, dof)
    key: ignored
    returns: (batch,)
    """
    w = jnp.ones(xs.shape[-1]) if weights is None else weights
    return jax.vmap(lambda x: jnp.sum(w * jnp.diagonal(jax.hessian(fn)(x))))(xs)


@partial(jax.jit, static_argnames=["fn"])
def laplacian_central_difference(fn, xs, key=None, weights=None, h=1e-4):
    """Δf (or Σ w_k ∂²_k f) via central finite differences (no AD required).

    Cost: 2*dof+1 model evaluations per sample.  Memory: O(dof).

    xs:  (batch, dof)
    key: ignored
    returns: (batch,)
    """
    batch_fn = jax.vmap(fn)
    n = xs.shape[-1]
    w = jnp.ones(n) if weights is None else weights
    f0 = batch_fn(xs)

    def body(acc, i):
        e_i = jnp.zeros(n).at[i].set(1.0)
        d2 = (batch_fn(xs + h * e_i) - 2 * f0 + batch_fn(xs - h * e_i)) / h**2
        return acc + w[i] * d2, None

    result, _ = jax.lax.scan(body, jnp.zeros(xs.shape[0]), jnp.arange(n))
    return result


@partial(jax.jit, static_argnames=["fn", "n_terms", "distribution"])
def laplacian_hutchinson(fn, xs, key, n_terms=10, distribution="rademacher", weights=None):
    """Δf (or Σ w_k ∂²_k f) via stochastic Hutchinson trace estimator.

    Unbiased estimator:
        Tr(W H_f(x)) ≈ (1/n) Σᵢ zᵢᵀ W H_f(x) zᵢ      (E[z zᵀ] = I)

    Each term uses one forward-over-reverse JVP, same as forward_ad.
    For n_terms << dof the cost is O(n_terms) JVPs instead of O(dof),
    at the cost of estimator variance.  For n_terms = dof both are equivalent.

    On GPU, the n_terms probes are parallelised via vmap, which can make
    Hutchinson faster than forward_ad's sequential fori_loop even at equal FLOP.

    xs:           (batch, dof)
    key:          JAX PRNGKey — fresh key each call for unbiased estimates
    n_terms:    number of probe vectors (higher = lower variance, higher cost)
    distribution: "rademacher" (±1, variance-optimal) or "gaussian"
    returns:      (batch,) — stochastic estimate of Tr(W H_f)
    """
    dof = xs.shape[-1]
    w = jnp.ones(dof) if weights is None else weights
    if distribution == "rademacher":
        # Bernoulli → {0,1} → {-1, +1}
        z = 2 * jax.random.bernoulli(key, shape=(n_terms, dof)).astype(jnp.float32) - 1
    else:
        z = jax.random.normal(key, shape=(n_terms, dof))

    # Same probe vectors for every sample in the batch (cheap + still unbiased)
    def estimate_single(x):
        def single_probe(zi):
            # ∇(zᵢ · ∇f)(x) = H_f(x) · zᵢ  via one JVP of grad(fn)
            _, hess_zi = jax.jvp(jax.grad(fn), (x,), (zi,))
            return jnp.dot(w * zi, hess_zi)

        return jnp.mean(jax.vmap(single_probe)(z))

    return jax.vmap(estimate_single)(xs)
