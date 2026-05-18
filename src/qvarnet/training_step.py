"""Training step computation for Variational Monte Carlo."""

from functools import partial
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .vmc_state import VMCState
from .qgt import compute_natural_gradient, DEFAULT_QGT_CONFIG


def compute_local_energy(
    hamiltonian, params, samples: jnp.ndarray, model_apply: Callable, is_log_model: bool
) -> jnp.ndarray:
    """Compute E_loc(x) = Ĥψ(x)/ψ(x) for all samples.

    Returns:
        Local energies, shape (batch,)
    """
    return hamiltonian.local_energy(
        params, samples, model_apply, is_log_model=is_log_model
    ).squeeze()


@partial(jax.jit, static_argnames=["model_apply", "is_log_model"])
def energy_fn(
    hamiltonian,
    params,
    batch: jnp.ndarray,
    model_apply: Callable,
    is_log_model: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute energy expectation value and standard error.

    Returns:
        E: Energy expectation value, scalar
        E_loc: Local energies per sample, shape (batch,)
        sigma_e: Standard error of energy, scalar
    """
    E_loc = compute_local_energy(hamiltonian, params, batch, model_apply, is_log_model)
    E = jnp.mean(E_loc)
    sigma_e = jnp.std(E_loc)
    return E, E_loc, sigma_e




@partial(jax.jit, static_argnames=["model_apply", "is_log_model", "use_cusp_condition"])
def energy_and_grads(
    hamiltonian,
    params,
    batch: jnp.ndarray,
    model_apply: Callable,
    is_log_model: bool,
    use_cusp_condition: bool = False,
    cusp_configs: jnp.ndarray = None,
    cusp_alpha: float = 0.01,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """Compute energy and parameter gradients via variance-minimization loss.

    Loss: L(θ) = 2⟨(E_loc - ⟨E⟩) log|ψ_θ|⟩
                + cusp_alpha * mean_{x∈cusp_configs}[ E_L(x;θ)² ]  (if use_cusp_condition)
    """
    E, E_loc, sigma_e = energy_fn(hamiltonian, params, batch, model_apply, is_log_model)

    def _log_psi(p):
        out = model_apply(p, batch).squeeze()
        return out if is_log_model else jnp.log(jnp.abs(out))

    def loss(p):
        vmc = 2 * jnp.mean(jax.lax.stop_gradient(E_loc - E) * _log_psi(p))
        if use_cusp_condition:
            E_cusp = hamiltonian.local_energy(p, cusp_configs, model_apply, is_log_model)
            # Normalize by stop-gradient RMS so the cusp loss is O(1) regardless of
            # epsilon. Without this, E_cusp ~ L(L-1)/epsilon^2 ~ 10^4 at the cusp,
            # making the gradient ~10^8 and blowing up the model.
            E_cusp_rms = jax.lax.stop_gradient(jnp.sqrt(jnp.mean(E_cusp ** 2)) + 1.0)
            return vmc + cusp_alpha * jnp.mean((E_cusp / E_cusp_rms) ** 2)
        return vmc

    grads = jax.grad(loss)(params)
    return E, sigma_e, grads


@partial(jax.jit, static_argnames=["is_log_model", "use_qgt", "use_cusp_condition"])
def compute_step(
    state: VMCState,
    batch: jnp.ndarray,
    hamiltonian,
    is_log_model: bool = False,
    use_qgt: bool = False,
    qgt_config: dict = None,
    use_cusp_condition: bool = False,
    cusp_configs: jnp.ndarray = None,
    cusp_alpha: float = 0.01,
) -> Tuple[VMCState, jnp.ndarray, jnp.ndarray]:
    """Perform one training step: compute energy/gradients and update parameters."""
    if qgt_config is None:
        qgt_config = DEFAULT_QGT_CONFIG.to_dict()

    E, sigma_e, grads = energy_and_grads(
        hamiltonian, state.params, batch, state.apply_fn,
        is_log_model=is_log_model,
        use_cusp_condition=use_cusp_condition,
        cusp_configs=cusp_configs,
        cusp_alpha=cusp_alpha,
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
