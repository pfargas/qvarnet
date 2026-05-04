import jax
import jax.numpy as jnp
from functools import partial


# @partial(jax.jit, static_argnames=["func"])
def laplacian_OLD(func, x):
    """Compute the laplacian operator of the model output with respect to inputs."""
    grad_fn = jax.grad(func)
    d2_dx2 = 0
    for i in range(x.shape[1]):
        d2_dx2 += jax.vmap(jax.grad(lambda xi: grad_fn(xi)[i]))(x)[:, i]
    return d2_dx2


@partial(jax.jit, static_argnames=["model_apply"])
def laplacian_autodiff_new(params, xs, model_apply):
    """Computes Δψ using Forward-over-Reverse AD.  O(DoF) memory, O(DoF) model evals.

    xs: (batch, DoF)
    returns: (batch,)  — ∇²ψ(x) for each sample
    """

    def psi_fn(x):
        # x: (DoF,) — single config; model needs (1, DoF)
        return model_apply(params, x.reshape(1, -1)).squeeze()  # scalar

    def laplacian_single(x):
        # x: (DoF,)  → scalar
        # Compute Σ_i ∂²ψ/∂x_i²  via jvp(grad(ψ), x, e_i)[1][i] = H[i,i]
        n_dims = x.shape[0]

        def body_fun(i, val):
            e_i = jnp.eye(n_dims)[i]  # (DoF,)
            # jvp of grad: tangent output is H·e_i = i-th column of Hessian
            grad_dot_hessian = jax.jvp(jax.grad(psi_fn), (x,), (e_i,))[1]  # (DoF,)
            return val + grad_dot_hessian[i]  # accumulate H[i,i]

        return jax.lax.fori_loop(0, n_dims, body_fun, 0.0)  # scalar

    return jax.vmap(laplacian_single)(xs)  # (batch,)


def laplacian_autodiff_FULL_HESSIAN(params, xs, model_apply):
    """Computes Δψ via full Hessian trace.  O(DoF²) memory — avoid for large DoF.

    xs: (batch, DoF)
    returns: (batch,)
    """

    def psi_fn(x):
        # x: (DoF,) → scalar
        return model_apply(params, x.reshape(1, -1)).squeeze()

    def laplacian_fn(x):
        # hessian: (DoF, DoF) — trace = Δψ
        return jnp.trace(jax.hessian(psi_fn)(x))  # scalar

    return jax.vmap(laplacian_fn)(xs)  # (batch,)


def laplacian_central_difference(params, xs, model_apply, h=0.05):
    """Computes Δψ via central differences.  2*DoF+1 model evals per sample.

    xs: (batch, DoF)
    returns: (batch,)
    """

    def psi_single(x_single):
        # x_single: (DoF,) → scalar; model needs (1, DoF)
        return model_apply(params, x_single.reshape(1, -1)).squeeze()

    def single_point_laplacian(x):
        # x: (DoF,) → scalar
        # Σ_i [ψ(x+h·e_i) - 2ψ(x) + ψ(x-h·e_i)] / h²
        d2psi = 0.0
        for i in range(x.shape[0]):
            ei = jnp.eye(x.shape[0])[i]
            f_plus = psi_single(x + h * ei)
            f_main = psi_single(x)
            f_minus = psi_single(x - h * ei)
            d2psi += (f_plus - 2 * f_main + f_minus) / (h**2)
        return d2psi  # scalar

    return jax.vmap(single_point_laplacian)(xs)  # (batch,)


@partial(jax.jit, static_argnames=["model_apply", "h"])
def laplacian_central_difference_scan(params, xs, model_apply, h=1e-4):
    """Computes Δψ via central differences using lax.scan (memory-efficient).

    Evaluates the full batch at once per perturbation; O(DoF) sequential steps.
    xs: (batch, DoF)
    returns: (batch,)
    """
    batch_size, n_dims = xs.shape
    # xs: (batch, DoF)

    f_main = model_apply(params, xs).squeeze()  # (batch,)

    def scan_body(carry, i):
        # carry: (batch,) accumulated laplacian
        # i: scalar dimension index
        e_i = jnp.eye(n_dims)[i]  # (DoF,) — broadcast to (batch, DoF) below

        x_plus = xs + h * e_i   # (batch, DoF)
        x_minus = xs - h * e_i  # (batch, DoF)

        psi_plus = model_apply(params, x_plus).squeeze()    # (batch,)
        psi_minus = model_apply(params, x_minus).squeeze()  # (batch,)

        d2_dx2 = (psi_plus - 2 * f_main + psi_minus) / (h**2)  # (batch,)
        return carry + d2_dx2, None

    final_laplacian, _ = jax.lax.scan(
        scan_body, init=jnp.zeros(batch_size), xs=jnp.arange(n_dims)
    )
    return final_laplacian  # (batch,)
