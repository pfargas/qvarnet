import jax
import jax.numpy as jnp
from .laplacian import laplacian_autodiff_new as laplacian_AD
from .laplacian import laplacian_central_difference


def kinetic_term(params, xs, model_apply, laplacian=laplacian_AD):
    """T = -0.5 * ∇²ψ / ψ for direct (non-log) models.

    xs: (batch, DoF)
    returns: (batch,)  — kinetic energy per sample
    """
    def psi_fn(x):
        # x: (DoF,) → reshape to (1, DoF) for model's expected batch dim
        x = jnp.atleast_1d(x).reshape(1, -1)
        return model_apply(params, x).squeeze()  # scalar

    d2psi = laplacian(params, xs, model_apply)  # (batch,)
    psi_vals = jax.vmap(lambda x: psi_fn(x))(xs)  # (batch,)
    return -0.5 * (d2psi / psi_vals)  # (batch,)


def kinetic_term_divergence_theorem(params, xs, model_apply):
    """T ≈ +0.5 * |∇ log ψ|² (gradient-only approximation, no laplacian).

    xs: (batch, DoF)
    returns: (batch,)
    """
    def log_psi_fn(x):
        x = jnp.atleast_1d(x).reshape(1, -1)
        psi = model_apply(params, x).squeeze()
        return jnp.log(jnp.abs(psi) + 1e-12)  # scalar

    grad_log_psi_fn = jax.grad(log_psi_fn)
    grad_val = jax.vmap(grad_log_psi_fn)(xs)  # (batch, DoF)
    return 0.5 * jnp.sum(grad_val**2, axis=-1)  # (batch,)


def kinetic_term_log(params, samples, model_apply):
    """T = -0.5 * (Δ log ψ + |∇ log ψ|²) for direct models converted to log-space.

    samples: (batch, DoF)
    returns: (batch,)
    """
    def log_psi_fn(x):
        psi = model_apply(params, x)
        return jnp.log(jnp.abs(psi) + 1e-12).squeeze()  # scalar

    grad_log_psi_fn = jax.grad(log_psi_fn)
    grad_val = jax.vmap(grad_log_psi_fn)(samples)  # (batch, DoF)

    def laplacian_log_psi(x):
        return jnp.trace(jax.hessian(log_psi_fn)(x))  # scalar — O(DoF²) memory

    lap_val = jax.vmap(laplacian_log_psi)(samples)  # (batch,)

    return -0.5 * (lap_val + jnp.sum(grad_val**2, axis=-1))  # (batch,)


def kinetic_term_log_wavefunction(params, samples, model_apply, laplacian=laplacian_AD):
    """T = -0.5 * (Δ log ψ + |∇ log ψ|²) for log-space models (model outputs log|ψ|).

    samples: (batch, DoF)
    returns: (batch,)

    Uses the AD-based laplacian (memory-efficient O(DoF) vs full Hessian O(DoF²)).
    """
    def log_psi_fn(x):
        # model already returns log|ψ|; x is a single sample (DoF,)
        psi = model_apply(params, x)
        return psi.squeeze()  # scalar

    grad_log_psi_fn = jax.grad(log_psi_fn)
    grad_val = jax.vmap(grad_log_psi_fn)(samples)  # (batch, DoF)

    lap_psi = laplacian(params, samples, model_apply)  # (batch,)

    return -0.5 * (lap_psi + jnp.sum(grad_val**2, axis=-1))  # (batch,)
