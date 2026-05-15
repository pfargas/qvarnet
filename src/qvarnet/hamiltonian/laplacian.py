"""Laplacian implementations for kinetic energy computation.

All functions share the signature:
    laplacian_fn(fn, xs) -> (batch,)

where fn: (DoF,) -> scalar is the function to differentiate (log|psi| or psi),
and xs: (batch, DoF) is the batch of configurations.
"""

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=["fn"])
def laplacian_forward_ad(fn, xs):
    """Delta f via forward-over-reverse AD.

    Computes diagonal of the Hessian without materialising it.
    Cost: O(DoF) JVPs.  Memory: O(DoF).  Exact.

    xs: (batch, DoF)
    returns: (batch,)
    """
    def laplacian_single(x):
        n = x.shape[0]

        def body(i, acc):
            e_i = jnp.zeros(n).at[i].set(1.0) # e_i = jnp.eye(n)[i]  # (DoF,)
            # JVP of grad(fn): tangent output is i-th column of Hessian
            _, hess_col = jax.jvp(jax.grad(fn), (x,), (e_i,))
            return acc + hess_col[i]

        return jax.lax.fori_loop(0, n, body, 0.0)

    return jax.vmap(laplacian_single)(xs)


@partial(jax.jit, static_argnames=["fn"])
def laplacian_full_hessian(fn, xs):
    """Delta f via full Hessian trace.

    Cost: O(DoF) backward passes.  Memory: O(DoF^2).  Exact.
    Only use for small DoF or debugging.

    xs: (batch, DoF)
    returns: (batch,)
    """
    return jax.vmap(lambda x: jnp.trace(jax.hessian(fn)(x)))(xs)


@partial(jax.jit, static_argnames=["fn"])
def laplacian_central_difference(fn, xs, h=1e-4):
    """Delta f via central finite differences (lax.scan, no AD required).

    Cost: 2*DoF+1 model evaluations per sample.  Memory: O(DoF).

    xs: (batch, DoF)
    returns: (batch,)
    """
    batch_fn = jax.vmap(fn)      # (batch, DoF) -> (batch,)
    n = xs.shape[-1]
    f0 = batch_fn(xs)            # (batch,)

    def body(acc, i):
        e_i = jnp.eye(n)[i]     # (DoF,)
        d2 = (batch_fn(xs + h * e_i) - 2 * f0 + batch_fn(xs - h * e_i)) / h**2
        return acc + d2, None

    result, _ = jax.lax.scan(body, jnp.zeros(xs.shape[0]), jnp.arange(n))
    return result
