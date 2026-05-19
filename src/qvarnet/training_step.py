"""Training step computation for Variational Monte Carlo."""

from functools import partial
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .vmc_state import VMCState
from .qgt import compute_natural_gradient, DEFAULT_QGT_CONFIG


def compute_local_energy(
    hamiltonian, params, samples: jnp.ndarray, model_apply: Callable
) -> jnp.ndarray:
    """Compute E_loc(x) = Ĥψ(x)/ψ(x) for all samples.

    Returns:
        Local energies, shape (batch,)
    """
    return hamiltonian.local_energy(params, samples, model_apply).squeeze()


@partial(jax.jit, static_argnames=["model_apply"])
def energy_fn(
    hamiltonian,
    params,
    batch: jnp.ndarray,
    model_apply: Callable,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute energy expectation value and standard error.

    Returns:
        E: Energy expectation value, scalar
        E_loc: Local energies per sample, shape (batch,)
        sigma_e: Standard error of energy, scalar
    """
    E_loc = compute_local_energy(hamiltonian, params, batch, model_apply)
    E = jnp.mean(E_loc)
    sigma_e = jnp.std(E_loc)
    return E, E_loc, sigma_e




@partial(jax.jit, static_argnames=["model_apply", "use_cusp_condition"])
def energy_and_grads(
    hamiltonian,
    params,
    batch: jnp.ndarray,
    model_apply: Callable,
    use_cusp_condition: bool = False,
    cusp_configs: jnp.ndarray = None,
    cusp_alpha: float = 0.01,
    cusp_pair_i: jnp.ndarray = None,
    cusp_pair_j: jnp.ndarray = None,
    cusp_epsilon: float = 1e-2,
    cusp_n: float = 2.0,
    cusp_C_n: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """Compute energy and parameter gradients via variance-minimization loss.

    Loss: L(θ) = 2⟨(E_loc - ⟨E⟩) log|ψ_θ|⟩
                + cusp_alpha * mean_{x∈cusp_configs}[ C_ij(x;θ) ]  (if use_cusp_condition)

    Cusp residual: C_ij(x;θ) = (ε^(n/2) * ∂log|ψ_θ|/∂r_ij - C_n)²
        where r_ij = x_i - x_j, so ∂/∂r_ij = (1/2)(∂/∂x_i - ∂/∂x_j).
    """
    E, E_loc, sigma_e = energy_fn(hamiltonian, params, batch, model_apply)

    def loss(p):
        log_psi = model_apply(p, batch).squeeze()
        vmc = 2 * jnp.mean(jax.lax.stop_gradient(E_loc - E) * log_psi)
        if use_cusp_condition:
            def log_psi_single(pos):
                return model_apply(p, pos[None]).squeeze()
            grad_log_psi = jax.vmap(jax.grad(log_psi_single))(cusp_configs)  # (n_cusp, N)
            n_cusp = cusp_configs.shape[0]
            idx = jnp.arange(n_cusp)
            grad_rij = 0.5 * (grad_log_psi[idx, cusp_pair_i] - grad_log_psi[idx, cusp_pair_j])
            cusp_residuals = (cusp_epsilon ** (cusp_n / 2.0) * grad_rij - cusp_C_n) ** 2
            return vmc + cusp_alpha * jnp.mean(cusp_residuals)
        return vmc

    grads = jax.grad(loss)(params)
    return E, sigma_e, grads


@partial(jax.jit, static_argnames=["use_qgt", "use_cusp_condition"])
def compute_step(
    state: VMCState,
    batch: jnp.ndarray,
    hamiltonian,
    use_qgt: bool = False,
    qgt_config: dict = None,
    use_cusp_condition: bool = False,
    cusp_configs: jnp.ndarray = None,
    cusp_alpha: float = 0.01,
    cusp_pair_i: jnp.ndarray = None,
    cusp_pair_j: jnp.ndarray = None,
    cusp_epsilon: float = 1e-2,
    cusp_n: float = 2.0,
    cusp_C_n: float = 1.0,
) -> Tuple[VMCState, jnp.ndarray, jnp.ndarray]:
    """Perform one training step: compute energy/gradients and update parameters."""
    if qgt_config is None:
        qgt_config = DEFAULT_QGT_CONFIG.to_dict()

    E, sigma_e, grads = energy_and_grads(
        hamiltonian, state.params, batch, state.apply_fn,
        use_cusp_condition=use_cusp_condition,
        cusp_configs=cusp_configs,
        cusp_alpha=cusp_alpha,
        cusp_pair_i=cusp_pair_i,
        cusp_pair_j=cusp_pair_j,
        cusp_epsilon=cusp_epsilon,
        cusp_n=cusp_n,
        cusp_C_n=cusp_C_n,
    )

    if not use_qgt:
        new_state = state.apply_gradients(grads=grads)
    else:
        new_state = _apply_natural_gradient_step(state, grads, batch, qgt_config)

    return new_state, E, sigma_e, grads


def _apply_natural_gradient_step(
    state: VMCState, grads: dict, batch: jnp.ndarray, qgt_config: dict
) -> VMCState:
    """Apply natural gradient update: θ ← θ - η S⁻¹(θ) ∇E(θ)."""
    natural_grad_flat, unravel_fn = compute_natural_gradient(
        state.params, batch, state.apply_fn, grads, qgt_config
    )
    learning_rate = qgt_config.get("learning_rate", 1e-3)
    params_flat = ravel_pytree(state.params)[0]
    new_params = unravel_fn(params_flat - learning_rate * natural_grad_flat)
    return state.replace(params=new_params)
