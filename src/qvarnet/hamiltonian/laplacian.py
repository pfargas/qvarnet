"""Laplacian implementations for kinetic energy computation.

All functions share the signature:
    laplacian_fn(fn, xs, key=None) -> (batch,)

where fn: (dof,) -> scalar is the function to differentiate (log|ψ|),
xs: (batch, dof) is the batch of configurations, and key is a JAX PRNGKey
(only used by laplacian_hutchinson; ignored by the exact methods).
"""

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=["fn"])
def laplacian_forward_ad(fn, xs, key=None):
    """Δf via forward-over-reverse AD (exact).

    Computes diagonal of the Hessian without materialising it.
    Cost: O(dof) JVPs.  Memory: O(dof).

    xs:  (batch, dof)
    key: ignored (exists for interface uniformity)
    returns: (batch,)
    """

    def laplacian_single(x):
        n = x.shape[0]

        def body(i, acc):
            e_i = jnp.zeros(n).at[i].set(1.0)
            _, hess_col = jax.jvp(jax.grad(fn), (x,), (e_i,))
            return acc + hess_col[i]

        return jax.lax.fori_loop(0, n, body, 0.0)

    return jax.vmap(laplacian_single)(xs)


@partial(jax.jit, static_argnames=["fn"])
def laplacian_full_hessian(fn, xs, key=None):
    """Δf via full Hessian trace (exact, debug only).

    Cost: O(dof) backward passes.  Memory: O(dof²).

    xs:  (batch, dof)
    key: ignored
    returns: (batch,)
    """
    return jax.vmap(lambda x: jnp.trace(jax.hessian(fn)(x)))(xs)


@partial(jax.jit, static_argnames=["fn"])
def laplacian_central_difference(fn, xs, key=None, h=1e-4):
    """Δf via central finite differences (no AD required).

    Cost: 2*dof+1 model evaluations per sample.  Memory: O(dof).

    xs:  (batch, dof)
    key: ignored
    returns: (batch,)
    """
    batch_fn = jax.vmap(fn)
    n = xs.shape[-1]
    f0 = batch_fn(xs)

    def body(acc, i):
        e_i = jnp.eye(n)[i]
        d2 = (batch_fn(xs + h * e_i) - 2 * f0 + batch_fn(xs - h * e_i)) / h**2
        return acc + d2, None

    result, _ = jax.lax.scan(body, jnp.zeros(xs.shape[0]), jnp.arange(n))
    return result


@partial(jax.jit, static_argnames=["fn", "n_terms", "distribution"])
def laplacian_hutchinson(fn, xs, key, n_terms=10, distribution="rademacher"):
    """Δf via stochastic Hutchinson trace estimator.

    Unbiased estimator:
        Tr(H_f(x)) ≈ (1/n) Σᵢ zᵢᵀ H_f(x) zᵢ  =  (1/n) Σᵢ zᵢ · ∇(zᵢ · ∇f)(x)

    Each term uses one forward-over-reverse JVP, same as forward_ad.
    For n_terms << dof the cost is O(n_terms) JVPs instead of O(dof),
    at the cost of estimator variance.  For n_terms = dof both are equivalent.

    On GPU, the n_terms probes are parallelised via vmap, which can make
    Hutchinson faster than forward_ad's sequential fori_loop even at equal FLOP.

    xs:           (batch, dof)
    key:          JAX PRNGKey — fresh key each call for unbiased estimates
    n_terms:    number of probe vectors (higher = lower variance, higher cost)
    distribution: "rademacher" (±1, variance-optimal) or "gaussian"
    returns:      (batch,) — stochastic estimate of Tr(H_f)
    """
    dof = xs.shape[-1]
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
            return jnp.dot(zi, hess_zi)

        return jnp.mean(jax.vmap(single_probe)(z))

    return jax.vmap(estimate_single)(xs)
