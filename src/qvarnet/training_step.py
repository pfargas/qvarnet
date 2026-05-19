"""Training step computation for Variational Monte Carlo."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .qgt import DEFAULT_QGT_CONFIG, QGTConfig, compute_natural_gradient
from .vmc_state import VMCState


def compute_local_energy(
    hamiltonian, params, samples: jnp.ndarray, model_apply: Callable, key=None
) -> jnp.ndarray:
    """Compute E_loc(x) = Ĥψ(x)/ψ(x) for all samples. Returns shape (batch,)."""
    return hamiltonian.local_energy(params, samples, model_apply, key=key).squeeze()


@partial(jax.jit, static_argnames=["model_apply"])
def energy_fn(
    hamiltonian,
    params,
    batch: jnp.ndarray,
    model_apply: Callable,
    key=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute energy expectation value and standard error.

    Returns:
        E: scalar energy expectation value
        E_loc: local energies per sample, shape (batch,)
        sigma_e: standard deviation of local energies, scalar
    """
    E_loc = compute_local_energy(hamiltonian, params, batch, model_apply, key=key)
    E = jnp.mean(E_loc)
    sigma_e = jnp.std(E_loc)
    return E, E_loc, sigma_e


@partial(jax.jit, static_argnames=["model_apply", "auxiliary_losses"])
def energy_and_grads(
    hamiltonian,
    params,
    batch: jnp.ndarray,
    model_apply: Callable,
    auxiliary_losses: tuple = (),
    key=None,
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    """Compute energy and parameter gradients via the VMC variance-minimisation loss.

    Loss: L(θ) = 2⟨(E_loc - ⟨E⟩) log|ψ_θ|⟩ + Σ aux_loss(θ)
    """
    E, E_loc, sigma_e = energy_fn(hamiltonian, params, batch, model_apply, key=key)

    def loss(p):
        log_psi = model_apply(p, batch).squeeze()
        vmc = 2 * jnp.mean(jax.lax.stop_gradient(E_loc - E) * log_psi)
        if auxiliary_losses:
            aux = sum(aux_loss(p, model_apply, batch) for aux_loss in auxiliary_losses)
            return vmc + aux
        return vmc

    grads = jax.grad(loss)(params)
    return E, sigma_e, grads


@partial(jax.jit, static_argnames=["use_qgt", "qgt_config", "auxiliary_losses"])
def compute_step(
    state: VMCState,
    batch: jnp.ndarray,
    hamiltonian,
    use_qgt: bool = False,
    qgt_config: QGTConfig = None,
    auxiliary_losses: tuple = (),
    key=None,
) -> tuple[VMCState, jnp.ndarray, jnp.ndarray]:
    """Perform one training step: compute energy/gradients and update parameters."""
    if qgt_config is None:
        qgt_config = DEFAULT_QGT_CONFIG

    E, sigma_e, grads = energy_and_grads(
        hamiltonian, state.params, batch, state.apply_fn,
        auxiliary_losses=auxiliary_losses,
        key=key,
    )

    if not use_qgt:
        new_state = state.apply_gradients(grads=grads)
    else:
        new_state = _apply_natural_gradient_step(state, grads, batch, qgt_config)

    return new_state, E, sigma_e, grads


def _apply_natural_gradient_step(
    state: VMCState, grads: dict, batch: jnp.ndarray, qgt_config: QGTConfig
) -> VMCState:
    """Apply natural gradient update: θ ← θ - η S⁻¹(θ) ∇E(θ)."""
    natural_grad_flat, unravel_fn = compute_natural_gradient(
        state.params, batch, state.apply_fn, grads, qgt_config
    )
    params_flat = ravel_pytree(state.params)[0]
    new_params = unravel_fn(params_flat - qgt_config.learning_rate * natural_grad_flat)
    return state.replace(params=new_params)
