"""Time-dependent variational Monte Carlo (TDVP) pieces (roadmap §8d, step 9).

TDVP / McLachlan projects the exact evolution onto the variational manifold:

    S θ̇ = -i F   (real time),     S θ̇ = -F   (imaginary time),
    S_kk' = ⟨O_k* O_k'⟩_c,   F_k = ⟨O_k* (E_loc - ⟨E⟩)⟩,   O_k = ∂_θk log ψ.

Key fact (roadmap §8d): **qvarnet already does imaginary-time TDVP** — SR with learning rate η
*is* Euler-integrated imaginary-time TDVP with δτ = η (up to the energy-gradient vs TDVP-force
factor of 2). So ``train(use_qgt=True)`` is imaginary-time t-VMC; this module adds the explicit
force, an imaginary-time step, and the **trajectory-error / convergence residual**.

Real time genuinely needs complex-output models (Im S = Berry curvature ⇒ the symplectic flow;
real log ψ has Im S = 0, GEOMETRY_NOTES.md) plus a stiff-ODE integrator (RK4 / p-tVMC). Those
are the research frontier; the shared ``geometry`` QGT and the extensible MetricsHistory schema
are the groundwork already in place. The functions here are for **real-output** models.
"""

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .qgt import compute_log_derivatives, compute_qgt, solve_qgt_cholesky


def tdvp_force(params, batch, e_loc, model_apply):
    """TDVP force F_k = ⟨O_k (E_loc - ⟨E⟩)⟩ for real models. Returns ``(F, unravel_fn)``.

    (The energy gradient used by SR is 2·F; the factor of 2 is the only difference between the
    SR step and the imaginary-time TDVP step.)
    """
    flat, unravel = ravel_pytree(params)
    log_derivs = compute_log_derivatives(flat, batch, lambda p, x: model_apply(unravel(p), x))
    o_centered = log_derivs - jnp.mean(log_derivs, axis=0)
    e = e_loc - jnp.mean(e_loc)
    F = jnp.mean(o_centered * e[:, None], axis=0)
    return F, unravel


def imaginary_time_step(params, batch, e_loc, model_apply, dt, regularization=1e-6):
    """One Euler step of imaginary-time TDVP: θ ← θ − dt·S⁻¹F (drives toward the ground state).

    Equivalent in direction to the SR step; exposed so dynamics code shares one implementation.
    """
    flat, unravel = ravel_pytree(params)
    S, _ = compute_qgt(flat, batch, lambda p, x: model_apply(unravel(p), x), regularization)
    F, _ = tdvp_force(params, batch, e_loc, model_apply)
    theta_dot = solve_qgt_cholesky(S, F)
    return unravel(flat - dt * theta_dot)


def tdvp_residual(params, batch, e_loc, model_apply, regularization=1e-6):
    """McLachlan imaginary-time residual r² = Var(E_loc) − Fᵀ S⁻¹ F (the analogue of the
    three-referee verdict for dynamics).

    It is the part of the energy fluctuation the tangent space *cannot* represent, so r² ≥ 0
    and r² → 0 at an exact eigenstate (zero-variance) or when the manifold is rich enough.
    Returns ``(residual, var, captured)``.
    """
    flat, unravel = ravel_pytree(params)
    S, _ = compute_qgt(flat, batch, lambda p, x: model_apply(unravel(p), x), regularization)
    F, _ = tdvp_force(params, batch, e_loc, model_apply)
    captured = float(F @ solve_qgt_cholesky(S, F))
    var = float(jnp.var(e_loc))
    return max(var - captured, 0.0), var, captured
