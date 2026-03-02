import jax
import jax.numpy as jnp
from .laplacian import laplacian_autodiff_new as laplacian


def kinetic_term(params, xs, model_apply, laplacian=laplacian):
    def psi_fn(x):
        # ensure input has shape (1,) as model expects last-dim features
        x = jnp.atleast_1d(x).reshape(1, -1)  # (1, DoF)
        return model_apply(params, x).squeeze()

    d2psi = laplacian(params, xs, model_apply)

    psi_vals = jax.vmap(lambda x: psi_fn(x))(xs)  # shape (batch,)

    psi_safe = psi_vals + 1e-12

    return -0.5 * (d2psi / psi_safe)  # shape (batch,)


def kinetic_term_divergence_theorem(params, xs, model_apply):
    def log_psi_fn(x):
        x = jnp.atleast_1d(x).reshape(1, -1)  # (1, DoF)
        psi = model_apply(params, x).squeeze()
        return jnp.log(jnp.abs(psi) + 1e-12)

    # Compute Gradient of Log Psi
    grad_log_psi_fn = jax.grad(log_psi_fn)
    grad_val = jax.vmap(grad_log_psi_fn)(xs)  # shape: (batch, n_dim)

    return 0.5 * jnp.sum(grad_val**2, axis=-1)


def kinetic_term_log(params, samples, model_apply):
    # This computes: -0.5 * laplacian(log_psi) - 0.5 * (grad(log_psi))^2
    # This form is numerically much more stable than (nabla^2 psi) / psi

    def log_psi_fn(x):
        psi = model_apply(params, x)
        # Sign trick: work in log domain to avoid underflow
        return jnp.log(jnp.abs(psi) + 1e-12)

    # 1. Compute Gradient of Log Psi
    # shape: (batch, n_dim)
    grad_log_psi_fn = jax.grad(log_psi_fn)
    grad_val = jax.vmap(grad_log_psi_fn)(samples)

    # 2. Compute Laplacian of Log Psi (Trace of Hessian)
    # Forward-mode AD is faster and safer for Laplacians
    def laplacian_log_psi(x):
        # We want trace of hessian.
        # Efficient trick: divergence of gradient
        return jnp.trace(jax.hessian(log_psi_fn)(x))

    lap_val = jax.vmap(laplacian_log_psi)(samples)

    # 3. Kinetic Energy Formula (Log Domain)
    # T = -0.5 * ( Laplacian(ln Psi) + |Grad(ln Psi)|^2 )
    kinetic = -0.5 * (lap_val + jnp.sum(grad_val**2, axis=-1))

    return kinetic
