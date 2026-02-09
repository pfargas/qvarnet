"""
Quantum Geometric Tensor (QGT) implementation for natural gradient optimization
in Variational Monte Carlo (VMC) calculations.

This module provides functions to compute the QGT and use it as a preconditioner
for energy minimization, implementing the stochastic reconfiguration method.

Mathematical foundation:
    S_ij(θ) = ⟨O_i* O_j⟩ - ⟨O_i*⟩⟨O_j⟩
where O_i = ∂/∂θ_i log|ψ(θ,x)⟩

Natural gradient update:
    θ_{t+1} = θ_t - η S^{-1}(θ_t) ∇_θ E(θ_t)
"""

import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import warnings


def compute_log_derivatives(params, batch, model_apply):
    """
    Compute logarithmic derivatives O_i = ∂/∂θ_i log|ψ(θ,x)⟩.

    Args:
        params: Model parameters (PyTree)
        batch: Batch of configurations (batch_size, DoF)
        model_apply: Function to apply model with given parameters

    Returns:
        O_i: Logarithmic derivatives of shape (batch_size, n_params)
    """

    def log_psi_fn(p, x):
        """Log of wavefunction with numerical stability."""
        psi = model_apply(p, x)
        return jnp.log(jnp.abs(psi) + 1e-8).squeeze()

    # Vectorized gradient computation for all parameters
    grad_log_psi_fn = jax.grad(log_psi_fn, argnums=0)
    O_i = jax.vmap(grad_log_psi_fn, in_axes=(None, 0))(params, batch)

    return O_i


def compute_qgt(params, batch, model_apply, regularization=1e-6):
    """
    Compute the Quantum Geometric Tensor S_ij = ⟨O_i* O_j⟩ - ⟨O_i*⟩⟨O_j⟩.

    Args:
        params: Model parameters (PyTree)
        batch: Batch of configurations (batch_size, DoF)
        model_apply: Function to apply model with given parameters
        regularization: Small constant for numerical stability

    Returns:
        S: Regularized QGT matrix of shape (n_params, n_params)
        O_i_mean: Mean of logarithmic derivatives
    """
    # 1. Compute log-derivatives O_i = ∂/∂θ_i log|ψ|
    O_i = compute_log_derivatives(params, batch, model_apply)
    O_i_conj = jnp.conj(O_i)

    # 2. Compute statistical averages
    O_i_mean = jnp.mean(O_i_conj, axis=0)  # (n_params,)

    # 3. Compute correlation matrix elements
    # S_ij = ⟨O_i* O_j⟩ - ⟨O_i*⟩⟨O_j⟩
    correlation_matrix = jnp.mean(jnp.einsum("bi,bj->bij", O_i_conj, O_i), axis=0)

    # Subtract outer product of means
    S = correlation_matrix - jnp.outer(O_i_mean, jnp.conj(O_i_mean))

    # 4. Add regularization for numerical stability
    S_reg = S + regularization * jnp.eye(S.shape[0])

    # Ensure Hermitian property (important for numerical stability)
    S_reg = 0.5 * (S_reg + jnp.conj(S_reg.T))

    return S_reg, O_i_mean


def solve_qgt_direct(S, grads):
    """
    Solve S·δθ = -∇E using direct matrix inversion.

    Args:
        S: QGT matrix (n_params, n_params)
        grads: Energy gradient vector or PyTree

    Returns:
        natural_grad: S^{-1}·grads
    """
    return jnp.linalg.solve(S, grads)


def solve_qgt_cholesky(S, grads):
    """
    Solve S·δθ = -∇E using Cholesky decomposition.
    More stable for symmetric positive definite matrices.

    Args:
        S: QGT matrix (n_params, n_params)
        grads: Energy gradient vector or PyTree

    Returns:
        natural_grad: S^{-1}·grads
    """
    try:
        L = jnp.linalg.cholesky(S)
        # Solve L·y = grads, then Lᵀ·x = y
        y = jnp.linalg.solve_triangular(L, grads, lower=True)
        natural_grad = jnp.linalg.solve_triangular(L.T, y, lower=False)
        return natural_grad
    except Exception as e:
        warnings.warn(
            f"Cholesky decomposition failed: {e}. Falling back to direct solve."
        )
        return solve_qgt_direct(S, grads)


def solve_qgt_gmres(S, grads, maxiter=1000, tol=1e-8):
    """
    Solve S·δθ = -∇E using GMRES iterative method.
    Suitable for large systems where direct inversion is expensive.

    Args:
        S: QGT matrix (n_params, n_params)
        grads: Energy gradient vector or PyTree
        maxiter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        natural_grad: S^{-1}·grads
    """

    def matvec(x):
        """Matrix-vector product for GMRES."""
        return jnp.dot(S, x)

    try:
        result, info = jax.scipy.sparse.linalg.gmres(
            matvec, grads, maxiter=maxiter, tol=tol
        )
        if info != 0:
            warnings.warn(f"GMRES did not converge: info={info}. Using direct solve.")
            return solve_qgt_direct(S, grads)
        return result
    except Exception as e:
        warnings.warn(f"GMRES failed: {e}. Falling back to direct solve.")
        return solve_qgt_direct(S, grads)


def solve_qgt_diagonal(S, grads):
    """
    Solve using only diagonal elements (computationally cheap approximation).
    Useful for very large parameter sets or as a preconditioner.

    Args:
        S: QGT matrix (n_params, n_params)
        grads: Energy gradient vector or PyTree

    Returns:
        natural_grad: Approximate S^{-1}·grads
    """
    diag_S = jnp.diag(S)
    # Avoid division by zero
    diag_S = jnp.where(diag_S < 1e-12, 1e-12, diag_S)
    return grads / diag_S


def flatten_params(params):
    """
    Flatten PyTree parameters to a single vector.

    Args:
        params: PyTree of parameters

    Returns:
        flat_params: 1D array
        unravel_fn: Function to restore PyTree structure
    """
    from jax.flatten_util import ravel_pytree

    flat_params, unravel_fn = ravel_pytree(params)
    return flat_params, unravel_fn


def unflatten_params(flat_params, unravel_fn):
    """
    Restore PyTree structure from flattened parameters.

    Args:
        flat_params: 1D array
        unravel_fn: Function to restore PyTree structure

    Returns:
        params: PyTree of parameters
    """
    return unravel_fn(flat_params)


def compute_natural_gradient(params, batch, model_apply, energy_grads, qgt_config):
    """
    Compute natural gradient using QGT as preconditioner.

    Args:
        params: Current parameters (PyTree)
        batch: Batch of configurations
        model_apply: Model application function
        energy_grads: Energy gradient (∇_θ E)
        qgt_config: Configuration dictionary for QGT computation

    Returns:
        natural_grad: Flattened natural gradient
        unravel_fn: Function to restore PyTree structure
    """
    # Flatten parameters for QGT computation
    flat_params, unravel_fn = flatten_params(params)
    flat_grads, _ = flatten_params(energy_grads)

    # Compute QGT
    S, _ = compute_qgt(
        flat_params,
        batch,
        lambda p, x: model_apply(unflatten_params(p, unravel_fn), x),
        qgt_config.get("regularization", 1e-6),
    )

    # Choose solver based on configuration
    solver = qgt_config.get("solver", "cholesky")

    if solver == "direct":
        natural_grad = solve_qgt_direct(S, flat_grads)
    elif solver == "cholesky":
        natural_grad = solve_qgt_cholesky(S, flat_grads)
    elif solver == "gmres":
        solver_opts = qgt_config.get("solver_options", {})
        natural_grad = solve_qgt_gmres(
            S,
            flat_grads,
            maxiter=solver_opts.get("maxiter", 1000),
            tol=solver_opts.get("tolerance", 1e-8),
        )
    elif solver == "diagonal":
        natural_grad = solve_qgt_diagonal(S, flat_grads)
    else:
        raise ValueError(f"Unknown QGT solver: {solver}")

    return natural_grad, unravel_fn


def train_step_qgt(state, batch, qgt_config):
    """
    Single training step using QGT-preconditioned natural gradient.

    Args:
        state: Flax TrainState
        batch: Batch of configurations
        qgt_config: QGT configuration dictionary

    Returns:
        new_state: Updated TrainState
        energy: Current energy value
    """
    # Import here to avoid circular imports
    from .train import loss_and_grads

    # Compute energy and its gradient
    energy, grads = loss_and_grads(state.params, batch, state.apply_fn)

    # Compute natural gradient using QGT
    natural_grad_flat, unravel_fn = compute_natural_gradient(
        state.params, batch, state.apply_fn, grads, qgt_config
    )

    # Apply natural gradient with learning rate
    learning_rate = qgt_config.get("learning_rate", 1e-3)
    new_params_flat = (
        flatten_params(state.params)[0] - learning_rate * natural_grad_flat
    )
    new_params = unflatten_params(new_params_flat, unravel_fn)

    # Create new state
    new_state = state.replace(params=new_params)
    return new_state, energy


def compute_qgt_block_diagonal(
    params, batch, model_apply, layer_sizes, regularization=1e-6
):
    """
    Compute block-diagonal approximation of QGT based on network layers.
    Reduces computational cost for large networks (K-FAC style).

    Args:
        params: Model parameters (PyTree)
        batch: Batch of configurations
        model_apply: Model application function
        layer_sizes: List of parameter counts per layer
        regularization: Regularization parameter

    Returns:
        S_block: Block-diagonal QGT matrix
    """
    # Flatten parameters for easier indexing
    flat_params, unravel_fn = flatten_params(params)
    O_i = compute_log_derivatives(params, batch, model_apply)

    # Compute full QGT first
    S_full, _ = compute_qgt(params, batch, model_apply, 0.0)

    # Create block-diagonal matrix
    n_params = flat_params.shape[0]
    S_block = jnp.zeros((n_params, n_params))

    start_idx = 0
    for layer_size in layer_sizes:
        end_idx = start_idx + layer_size
        # Extract and copy block
        S_block = S_block.at[start_idx:end_idx, start_idx:end_idx].set(
            S_full[start_idx:end_idx, start_idx:end_idx]
        )
        start_idx = end_idx

    # Add regularization
    S_block = S_block + regularization * jnp.eye(n_params)
    return S_block


def compute_qgt_statistics(S):
    """
    Compute diagnostic statistics for QGT matrix.

    Args:
        S: QGT matrix

    Returns:
        stats: Dictionary with condition number, eigenvalue distribution, etc.
    """
    try:
        eigenvals = jnp.linalg.eigvalsh(S)
        stats = {
            "condition_number": jnp.max(eigenvals) / (jnp.min(eigenvals) + 1e-12),
            "max_eigenvalue": jnp.max(eigenvals),
            "min_eigenvalue": jnp.min(eigenvals),
            "rank": jnp.sum(eigenvals > 1e-12),
            "trace": jnp.trace(S),
            "determinant": jnp.linalg.det(S),
        }
        return stats
    except Exception as e:
        warnings.warn(f"QGT statistics computation failed: {e}")
        return {}


class QGTConfig:
    """
    Configuration class for QGT optimization parameters.
    """

    def __init__(
        self,
        solver="cholesky",
        learning_rate=1e-3,
        regularization=1e-6,
        solver_options=None,
    ):
        """
        Initialize QGT configuration.

        Args:
            solver: Solver type ('direct', 'cholesky', 'gmres', 'diagonal')
            learning_rate: Learning rate for natural gradient updates
            regularization: Regularization parameter for numerical stability
            solver_options: Additional options for iterative solvers
        """
        self.solver = solver
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.solver_options = solver_options or {}

    def to_dict(self):
        """Convert to dictionary format."""
        return {
            "solver": self.solver,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "solver_options": self.solver_options,
        }


# Default QGT configurations for common use cases
DEFAULT_QGT_CONFIG = QGTConfig(
    solver="cholesky", learning_rate=1e-3, regularization=1e-6
)

MEMORY_EFFICIENT_QGT_CONFIG = QGTConfig(
    solver="diagonal", learning_rate=1e-3, regularization=1e-4
)

LARGE_SYSTEM_QGT_CONFIG = QGTConfig(
    solver="gmres",
    learning_rate=5e-4,
    regularization=1e-4,
    solver_options={"maxiter": 500, "tolerance": 1e-6},
)
