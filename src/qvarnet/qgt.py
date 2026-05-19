"""Quantum Geometric Tensor (QGT) for natural-gradient / stochastic reconfiguration.

S_{kl}(θ) = ⟨O_k O_l⟩ − ⟨O_k⟩⟨O_l⟩,   O_k = ∂log|ψ_θ(x)|/∂θ_k

Natural gradient update (stochastic reconfiguration):
    θ_{t+1} = θ_t − η S⁻¹(θ_t) ∇_θ E(θ_t)

Memory note: compute_qgt materialises a full (n_params, n_params) matrix.
For large models this is O(n_params²) memory.  The matmul implementation
avoids the O(B * n_params²) einsum intermediate, but the matrix itself still
exists.  For n_params > ~10_000, prefer a matrix-free CG solver:
jax.scipy.sparse.linalg.cg(lambda v: S_matvec(v), grad) where S_matvec
computes S·v without materialising S.
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class QGTConfig:
    """Configuration for QGT-based (stochastic reconfiguration) optimisation.

    Pass an instance to train(..., qgt_config=QGTConfig(...), use_qgt=True).

    Args:
        solver:         "cholesky" (default, SPD, stable), "direct" (LU),
                        "gmres" (iterative, large systems), "diagonal" (cheap approx).
        learning_rate:  Step size for the natural gradient update (separate
                        from the Adam/SGD lr — this replaces it when use_qgt=True).
        regularization: ε added to the diagonal of S before solving (default 1e-6).
        solver_options: Extra kwargs forwarded to the iterative solver (GMRES only).
    """

    def __init__(
        self,
        solver: str = "cholesky",
        learning_rate: float = 1e-3,
        regularization: float = 1e-6,
        solver_options: dict = None,
    ):
        self.solver = solver
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.solver_options = solver_options or {}

    def to_dict(self):
        return {
            "solver": self.solver,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "solver_options": self.solver_options,
        }


DEFAULT_QGT_CONFIG = QGTConfig(solver="cholesky", learning_rate=1e-3, regularization=1e-6)
MEMORY_EFFICIENT_QGT_CONFIG = QGTConfig(solver="diagonal", learning_rate=1e-3, regularization=1e-4)
LARGE_SYSTEM_QGT_CONFIG = QGTConfig(
    solver="gmres",
    learning_rate=5e-4,
    regularization=1e-4,
    solver_options={"maxiter": 500, "tolerance": 1e-6},
)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_log_derivatives(params, batch, model_apply):
    """Compute O_k = ∂log|ψ_θ(x)|/∂θ_k for every sample in batch.

    Args:
        params:      flat 1-D parameter vector, shape (n_params,).
        batch:       configurations, shape (batch_size, dof).
        model_apply: callable(params, x) → log|ψ| scalar.
                     Must already output log|ψ| (log-model convention).

    Returns:
        shape (batch_size, n_params).
    """
    def log_psi(p, x):
        return model_apply(p, x[None]).squeeze()  # x[None]: (1, dof) → scalar

    return jax.vmap(jax.grad(log_psi, argnums=0), in_axes=(None, 0))(params, batch)


def compute_qgt(params, batch, model_apply, regularization: float = 1e-6):
    """Compute the regularised QGT matrix S, shape (n_params, n_params).

    S_{kl} = ⟨O_k O_l⟩ − ⟨O_k⟩⟨O_l⟩  + ε I

    Uses a matmul instead of einsum to avoid the O(B * n_params²) intermediate.
    """
    log_derivs = compute_log_derivatives(params, batch, model_apply)  # (B, P)
    B = log_derivs.shape[0]
    O_mean = jnp.mean(log_derivs, axis=0)                             # (P,)
    # log_derivs.T @ log_derivs / B  =  ⟨O_k O_l⟩  (O(B*P²) FLOP, O(P²) memory)
    S = log_derivs.T @ log_derivs / B - jnp.outer(O_mean, O_mean)
    S = S + regularization * jnp.eye(S.shape[0])
    return 0.5 * (S + S.T), O_mean   # symmetrise to cancel floating-point skew


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solve_qgt_direct(S, grads):
    """Solve S·x = grads via LU decomposition."""
    return jnp.linalg.solve(S, grads)


def solve_qgt_cholesky(S, grads):
    """Solve S·x = grads via Cholesky (S must be SPD — guaranteed by regularisation)."""
    factor = jax.scipy.linalg.cho_factor(S)
    return jax.scipy.linalg.cho_solve(factor, grads)


def solve_qgt_gmres(S, grads, maxiter: int = 1000, tol: float = 1e-8):
    """Solve S·x = grads via GMRES iterative solver."""
    result, _ = jax.scipy.sparse.linalg.gmres(
        lambda x: jnp.dot(S, x), grads, maxiter=maxiter, tol=tol
    )
    return result


def solve_qgt_diagonal(S, grads):
    """Approximate solve using only the diagonal of S (cheap preconditioner)."""
    diag = jnp.where(jnp.diag(S) < 1e-12, 1e-12, jnp.diag(S))
    return grads / diag


# ---------------------------------------------------------------------------
# High-level interface
# ---------------------------------------------------------------------------

def compute_natural_gradient(params, batch, model_apply, energy_grads, qgt_config: QGTConfig):
    """Compute S⁻¹ ∇E — the natural gradient.

    Args:
        params:       parameter pytree.
        batch:        MCMC batch, shape (batch_size, dof).
        model_apply:  callable(params, x) → log|ψ| scalar.
        energy_grads: ∇_θ E, same pytree structure as params.
        qgt_config:   QGTConfig instance (solver, regularization, learning_rate).

    Returns:
        (natural_grad_flat, unravel_fn)
    """
    flat_params, unravel_fn = ravel_pytree(params)
    flat_grads, _ = ravel_pytree(energy_grads)

    S, _ = compute_qgt(
        flat_params,
        batch,
        lambda p, x: model_apply(unravel_fn(p), x),
        qgt_config.regularization,
    )

    solver = qgt_config.solver
    if solver == "direct":
        natural_grad = solve_qgt_direct(S, flat_grads)
    elif solver == "cholesky":
        natural_grad = solve_qgt_cholesky(S, flat_grads)
    elif solver == "gmres":
        opts = qgt_config.solver_options
        natural_grad = solve_qgt_gmres(
            S, flat_grads,
            maxiter=opts.get("maxiter", 1000),
            tol=opts.get("tolerance", 1e-8),
        )
    elif solver == "diagonal":
        natural_grad = solve_qgt_diagonal(S, flat_grads)
    else:
        raise ValueError(f"Unknown QGT solver: {solver!r}")

    return natural_grad, unravel_fn
