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
    """Configuration for stochastic reconfiguration (natural-gradient) optimisation.

    SR is a *preconditioner*: the optimizer you pass to VMC is still the update rule,
    and ``learning_rate`` here only feeds the ``sr_train`` recipe and the trust-region
    derivation. Defaults are validated on Calogero-Sutherland N=30. What these mean,
    and the measurements behind them: docs/explainers/stochastic-reconfiguration.md.

    Args:
        solver: "auto" (minsr when P > M, else cholesky) | "cholesky" | "minsr" |
            "gmres" | "diagonal" | "direct" (avoid: silent on non-PSD S).
        learning_rate: the classic SR step eta. Does NOT override the optimizer.
        regularization: epsilon in the Jacobi-preconditioned metric.
        max_state_change: Fisher cap on the state change per step.
        trust_region: advanced override in direction units; required instead of
            max_state_change when learning_rate is an optax schedule.
        grad_clip_norm: Euclidean clip on the natural gradient. Usually wrong to
            enable -- it fights the preconditioning.
        solver_options: extra kwargs for the iterative solver (GMRES only).
    """

    def __init__(
        self,
        solver: str = "auto",
        learning_rate: float = 1e-3,
        regularization: float = 1e-2,
        max_state_change: float = 0.1,
        trust_region: float = None,
        grad_clip_norm: float = None,
        solver_options: dict = None,
    ):
        self.solver = solver
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_state_change = max_state_change
        self.trust_region = trust_region
        self.grad_clip_norm = grad_clip_norm
        self.solver_options = solver_options or {}

    def resolve_trust_region(self) -> float:
        """The direction-space cap Δ actually applied: √(δᵀSδ) ≤ Δ, or None (off).

        ``trust_region`` (direction units) wins when set; otherwise derived from the
        physical knob as ``max_state_change / learning_rate``.
        """
        if self.trust_region is not None:
            return self.trust_region
        if self.max_state_change is None:
            return None
        if callable(self.learning_rate):
            raise ValueError(
                "QGTConfig.max_state_change cannot be combined with a learning-rate "
                "schedule (the direction cap is max_state_change/learning_rate). "
                "Set QGTConfig.trust_region explicitly (direction units) instead."
            )
        return self.max_state_change / self.learning_rate

    def to_dict(self):
        return {
            "solver": self.solver,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "max_state_change": self.max_state_change,
            "trust_region": self.trust_region,
            "grad_clip_norm": self.grad_clip_norm,
            "solver_options": self.solver_options,
        }


def resolve_qgt_solver(solver: str, n_params: int, n_samples: int) -> str:
    """Resolve "auto" to a concrete solver: "minsr" when P > M, else "cholesky".

    minSR solves the same regularised natural-gradient step in the M×M sample space,
    which is both cheaper and full-rank in the over-parametrised regime (the sampled
    P×P S has rank ≤ M there). Both sizes are static under jit, so this resolves at
    trace time.
    """
    if solver != "auto":
        return solver
    return "minsr" if n_params > n_samples else "cholesky"


DEFAULT_QGT_CONFIG = QGTConfig(solver="auto", learning_rate=1e-3, regularization=1e-2)
MEMORY_EFFICIENT_QGT_CONFIG = QGTConfig(solver="diagonal", learning_rate=1e-3, regularization=1e-2)
LARGE_SYSTEM_QGT_CONFIG = QGTConfig(
    solver="gmres",
    learning_rate=5e-4,
    regularization=1e-2,
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


def compute_qgt(params, batch, model_apply, regularization: float = 1e-2):
    """Compute the regularised QGT matrix S, shape (n_params, n_params).

    S_{kl} = ⟨O_k O_l⟩ − ⟨O_k⟩⟨O_l⟩  + ε I

    Uses a matmul instead of einsum to avoid the O(B * n_params²) intermediate.
    """
    log_derivs = compute_log_derivatives(params, batch, model_apply)  # (B, P)
    B = log_derivs.shape[0]
    O_mean = jnp.mean(log_derivs, axis=0)  # (P,)
    # Centred build: S = ŌᵀŌ/B with Ō = O − ⟨O⟩. Mathematically identical to
    # ⟨OOᵀ⟩ − ⟨O⟩⟨O⟩ᵀ but PSD by construction up to rounding — the uncentred form
    # subtracts two large nearly-equal matrices (catastrophic cancellation in float32
    # when the O_k have large means, e.g. Jastrow Σlog r terms), which is how S
    # acquires spurious negative eigenvalues and the solve returns non-descent steps.
    O_centered = log_derivs - O_mean  # (B, P)
    S = O_centered.T @ O_centered / B  # (O(B*P²) FLOP, O(P²) memory)
    # Scale-invariant shift (Sorella): ε is a *fraction of the mean diagonal curvature*,
    # not an absolute number. With an absolute shift, models whose O_k are large (e.g. a
    # Jastrow Σlog r term: diag(S) ~ 1e4-1e6) get a relative regularisation below f32
    # rounding and the Cholesky factorisation of S fails outright.
    shift = regularization * (jnp.mean(jnp.diag(S)) + 1e-30)
    S = S + shift * jnp.eye(S.shape[0])
    return 0.5 * (S + S.T), O_mean  # symmetrise to cancel floating-point skew


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

    The solve is Jacobi-preconditioned: S is rescaled to unit diagonal
    (S̃ = D^{-1/2} S D^{-1/2}, D = diag(S)) before the shift ε is added — equivalent
    to Levenberg-Marquardt regularisation S + ε·diag(S). This makes ε per-direction
    scale-invariant AND collapses the condition number, so float32 factorisations stay
    reliable whether the O_k are O(1) (spin nets) or O(1e5) (Jastrow log terms).

    Returns:
        (natural_grad_flat, unravel_fn, info) — ``info`` holds the guard diagnostics
        (``fisher_norm``, ``trust_scale``, ``solve_ok``, ``nat_grad_norm``).
    """
    flat_params, unravel_fn = ravel_pytree(params)
    flat_grads, _ = ravel_pytree(energy_grads)

    S_raw, _ = compute_qgt(
        flat_params,
        batch,
        lambda p, x: model_apply(unravel_fn(p), x),
        regularization=0.0,
    )

    # Jacobi scaling. Dead directions — zero-variance O_k, e.g. the additive-constant
    # bias of log|ψ| that every LogWavefunction has — get d_inv = 0, i.e. an exactly
    # zero step: they don't change the physical state, their exact force is 0, and any
    # nonzero value there is f32 rounding residue that a floor-based scheme would
    # amplify by 1/(ε·floor). This also matches minSR, whose step lives in the row
    # space of Ō and is identically zero along such directions.
    diag = jnp.diag(S_raw)
    cutoff = 1e-9 * jnp.max(diag) + 1e-30
    d_inv = jnp.where(diag > cutoff, 1.0 / jnp.sqrt(diag + cutoff), 0.0)
    S_t = S_raw * d_inv[:, None] * d_inv[None, :] + qgt_config.regularization * jnp.eye(
        S_raw.shape[0]
    )
    g_t = flat_grads * d_inv

    solver = resolve_qgt_solver(qgt_config.solver, flat_params.size, batch.shape[0])
    if solver == "direct":
        y = solve_qgt_direct(S_t, g_t)
    elif solver == "cholesky":
        y = solve_qgt_cholesky(S_t, g_t)
    elif solver == "gmres":
        opts = qgt_config.solver_options
        y = solve_qgt_gmres(
            S_t,
            g_t,
            maxiter=opts.get("maxiter", 1000),
            tol=opts.get("tolerance", 1e-8),
        )
    elif solver == "diagonal":
        y = solve_qgt_diagonal(S_t, g_t)
    else:
        raise ValueError(f"Unknown QGT solver: {solver!r}")
    natural_grad = y * d_inv

    # Trust-region metric = the solved (LM-regularised) metric: S + ε·diag(S).
    dSd = natural_grad @ (S_raw @ natural_grad) + qgt_config.regularization * jnp.sum(
        diag * natural_grad**2
    )
    natural_grad, info = _apply_trust_region(natural_grad, dSd, qgt_config)
    # Euclidean norm of the step handed to the optimizer — compare against
    # grad_clip_norm to see whether the clip (applied in the optax chain) binds.
    info["nat_grad_norm"] = jnp.linalg.norm(natural_grad)
    return natural_grad, unravel_fn, info


def _apply_trust_region(natural_grad, dSd, qgt_config: QGTConfig):
    """Rescale delta so its Fisher norm sqrt(delta^T S delta) is within the trust region.

    No-op when the resolved cap is None. A failed solve (NaN/inf, or dSd < 0 from a
    non-PSD S) returns a **zero step** rather than poisoning the parameters: the epoch
    is wasted and the next batch retries, which beats a NaN checkpoint.

    Returns ``(natural_grad, info)``, where info says which constraint shaped the step:
    ``fisher_norm`` (pre-rescale), ``trust_scale`` (1.0 = not binding) and ``solve_ok``
    (0.0 = failed solve). See docs/explainers/stochastic-reconfiguration.md.
    """
    delta = qgt_config.resolve_trust_region()
    fisher_norm = jnp.sqrt(jnp.maximum(dSd, 0.0))
    if delta is None:
        info = {
            "fisher_norm": fisher_norm,
            "trust_scale": jnp.asarray(1.0),
            "solve_ok": jnp.asarray(1.0),
        }
        return natural_grad, info
    scale = jnp.minimum(1.0, delta / (fisher_norm + 1e-30))
    ok = jnp.isfinite(dSd) & (dSd >= 0.0) & jnp.all(jnp.isfinite(natural_grad))
    info = {
        "fisher_norm": fisher_norm,
        "trust_scale": jnp.where(ok, scale, 0.0),
        "solve_ok": ok.astype(jnp.float32),
    }
    return jnp.where(ok, natural_grad * scale, jnp.zeros_like(natural_grad)), info


def compute_natural_gradient_minsr(params, batch, e_loc, model_apply, qgt_config: QGTConfig):
    """Natural gradient via the M x M Gram dual (minSR).

    Solves in sample space instead of parameter space, which is the right
    formulation when the parameter count P exceeds the sample count M. The
    push-through identity makes this the same regularised step as full SR.

    Args:
        params: parameter pytree.
        batch: samples, ``(M, dof)``.
        model_apply: the (coord-wrapped) log-amplitude callable.
        grads: the Euclidean energy gradient pytree.
        e_loc: local energies, ``(M,)``.
        regularization: epsilon in the preconditioned metric.
        trust_region: cap on sqrt(delta^T S delta); None disables it.

    Returns:
        ``(natural_gradient_pytree, info)``.
    """
    flat_params, unravel_fn = ravel_pytree(params)
    log_derivs = compute_log_derivatives(
        flat_params, batch, lambda p, x: model_apply(unravel_fn(p), x)
    )
    M = log_derivs.shape[0]
    O_bar = log_derivs - jnp.mean(log_derivs, axis=0)  # centre → (M, P)
    e = e_loc - jnp.mean(e_loc)  # centred residual → (M,)

    # Jacobi scaling with the SAME D = diag(S) = mean(Ō², axis=0) as full SR (see
    # compute_natural_gradient), so the push-through identity — hence exact
    # minSR ≡ full-SR equivalence — holds for the scaled matrices too:
    #     (ÕᵀÕ/M + εI_P)⁻¹Õᵀ = Õᵀ(ÕÕᵀ/M + εI_M)⁻¹,   Õ = Ō·D^{-1/2}.
    # Equivalent to Levenberg-Marquardt regularisation S + ε·diag(S).
    diag = jnp.mean(O_bar**2, axis=0)  # diag(S), (P,) — no P×P build needed
    cutoff = 1e-9 * jnp.max(diag) + 1e-30
    d_inv = jnp.where(diag > cutoff, 1.0 / jnp.sqrt(diag + cutoff), 0.0)
    O_t = O_bar * d_inv[None, :]  # (M, P)

    T = (O_t @ O_t.T) / M + qgt_config.regularization * jnp.eye(M)
    y = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(T), e)  # T⁻¹ e, (M,)
    natural_grad = (2.0 / M) * (O_t.T @ y) * d_inv  # (P,), back in the original basis
    # Trust-region metric = the solved (LM-regularised) metric, without building S:
    # δᵀ(S + ε·diag(S))δ = |Ōδ|²/M + ε·Σ diag_k δ_k².
    dSd = jnp.sum((O_bar @ natural_grad) ** 2) / M + qgt_config.regularization * jnp.sum(
        diag * natural_grad**2
    )
    natural_grad, info = _apply_trust_region(natural_grad, dSd, qgt_config)
    info["nat_grad_norm"] = jnp.linalg.norm(natural_grad)
    return natural_grad, unravel_fn, info
