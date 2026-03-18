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
    """
    Computes Laplacian using Forward-over-Reverse AD.
    Memory efficient: O(DoF) instead of O(DoF^2).
    """

    def psi_fn(x):
        return model_apply(params, x.reshape(1, -1)).squeeze()

    def laplacian_single(x):
        # We want sum_i (d^2 psi / dx_i^2)
        # We can calculate this by looping over dimensions and projecting
        # the gradient onto the unit vector e_i.

        n_dims = x.shape[0]

        def body_fun(i, val):
            # Create unit vector e_i
            # e_i = jnp.eye(n_dims)[i]
            e_i = jnp.zeros(n_dims).at[i].set(1.0)

            # jvp(grad(psi), (primal,), (tangent,))
            # resulting tangent is (Hessian * e_i)
            # We take the i-th component of that vector.
            grad_dot_hessian = jax.jvp(jax.grad(psi_fn), (x,), (e_i,))[1]

            return val + grad_dot_hessian[i]

        return jax.lax.fori_loop(0, n_dims, body_fun, 0.0)

    return jax.vmap(laplacian_single)(xs)


def laplacian_autodiff_FULL_HESSIAN(params, xs, model_apply):
    """
    Computes the Laplacian Δψ = ∇²ψ using JAX's Automatic Differentiation.
    This is shape-safe and significantly faster than central differences.
    """

    def psi_fn(x):
        # x is a single point (DoF,)
        return model_apply(params, x.reshape(1, -1)).squeeze()

    def laplacian_fn(x):  # memory throttle: DoF^2
        return jnp.trace(jax.hessian(psi_fn)(x))

    # Vectorize over the batch
    return jax.vmap(laplacian_fn)(xs)


def laplacian_central_difference(params, xs, model_apply, h=0.05):
    """
    Computes Laplacian using central difference, properly handling JAX batching.
    xs shape: (batch, DoF)
    """

    def psi_single(x_single):
        # x_single shape: (DoF,)
        # Reshape to (1, DoF) because Flax models expect a batch dimension
        return model_apply(params, x_single.reshape(1, -1)).squeeze()

    def single_point_laplacian(x):
        # x shape: (DoF,)
        d2psi = 0.0
        for i in range(x.shape[0]):
            # Create unit vector for coordinate i
            ei = jnp.eye(x.shape[0])[i]

            # Central difference formula: [f(x+h) - 2f(x) + f(x-h)] / h^2
            f_plus = psi_single(x + h * ei)
            f_main = psi_single(x)
            f_minus = psi_single(x - h * ei)

            d2psi += (f_plus - 2 * f_main + f_minus) / (h**2)
        return d2psi

    # Vectorize the single-point logic over the entire batch
    return jax.vmap(single_point_laplacian)(xs)


@partial(jax.jit, static_argnames=["model_apply", "h"])
def laplacian_central_difference_scan(params, xs, model_apply, h=1e-4):
    """
    Computes Laplacian using jax.lax.scan.
    Memory: O(N * D) (Linear scaling)
    Speed: Single compiled kernel (Fast)
    """
    batch_size, n_dims = xs.shape

    # 1. Pre-calculate the center value once
    # f_main shape: (batch,)
    f_main = model_apply(params, xs).squeeze()

    def scan_body(carry, i):
        # i is the dimension index we are currently perturbing

        # Create unit vector e_i
        e_i = jnp.eye(n_dims)[i]  # Shape: (D,)

        # Perturb current dimension i for the whole batch
        # We broadcast e_i to (batch, D)
        x_plus = xs + h * e_i
        x_minus = xs - h * e_i

        # Evaluate model at perturbed points
        # These run sequentially inside the compiled kernel, saving memory
        psi_plus = model_apply(params, x_plus).squeeze()
        psi_minus = model_apply(params, x_minus).squeeze()

        # Finite difference for dimension i
        d2_dx2 = (psi_plus - 2 * f_main + psi_minus) / (h**2)

        # Accumulate the result (Laplacian is sum of d2_dx2)
        new_laplacian = carry + d2_dx2

        return new_laplacian, None  # We don't need to stack outputs

    # 2. Run the loop inside XLA
    # init_val is zeros of shape (batch,)
    # xs=jnp.arange(n_dims) gives us the loop indices 0..D-1
    final_laplacian, _ = jax.lax.scan(
        scan_body, init=jnp.zeros(batch_size), xs=jnp.arange(n_dims)
    )

    return final_laplacian
