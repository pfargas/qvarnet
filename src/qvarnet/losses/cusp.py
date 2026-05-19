import jax
import jax.numpy as jnp

from .base import AuxiliaryLoss


class CuspLoss(AuxiliaryLoss):
    """Cusp-condition penalty: enforces ∂log|ψ|/∂r_ij → C_n / ε^(n/2) as r_ij → 0.

    Residual: (ε^(n/2) * ∂log|ψ|/∂r_ij - C_n)²  evaluated at pre-sampled near-coalescence
    points. The gradient w.r.t. r_ij is (1/2)(∂/∂x_i - ∂/∂x_j).
    """

    def __init__(self, cusp_configs, pair_i, pair_j, alpha, epsilon, n, C_n, **_):
        self.cusp_configs = cusp_configs  # (n_cusp, dof)
        self.pair_i = pair_i             # (n_cusp,) int
        self.pair_j = pair_j             # (n_cusp,) int
        self.alpha = alpha
        self.epsilon = epsilon
        self.n = n
        self.C_n = C_n

    def __call__(self, params, model_apply, batch):
        def log_psi_single(pos):
            return model_apply(params, pos[None]).squeeze()

        grad_log_psi = jax.vmap(jax.grad(log_psi_single))(self.cusp_configs)
        n_cusp = self.cusp_configs.shape[0]
        idx = jnp.arange(n_cusp)
        grad_rij = 0.5 * (grad_log_psi[idx, self.pair_i] - grad_log_psi[idx, self.pair_j])
        residuals = (self.epsilon ** (self.n / 2.0) * grad_rij - self.C_n) ** 2
        return self.alpha * jnp.mean(residuals)
