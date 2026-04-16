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
    """
    Compute local energy E_loc(x) = Ĥψ(x) / ψ(x) for all samples.

    Args:
        hamiltonian: Energy operator with local_energy method
        params: Model parameters (PyTree)
        samples: Configuration samples, shape (batch, DoF)
        model_apply: Model forward pass function
        is_log_model: If True, model outputs log(ψ)

    Returns:
        Local energies, shape (batch, 1)
    """
    local_energy = hamiltonian.local_energy(
        params, samples, model_apply, is_log_model=is_log_model
    )
    return local_energy.reshape(-1, 1)


@partial(jax.jit, static_argnames=["model_apply"])
def log_psi(x: jnp.ndarray, params, model_apply: Callable) -> jnp.ndarray:
    """
    Compute log|ψ(x)| for direct (non-log) models.

    Args:
        x: Configurations, shape (batch, DoF)
        params: Model parameters
        model_apply: Model function

    Returns:
        Log probabilities, shape (batch,)
    """
    psi = model_apply(params, x)
    return jnp.log(jnp.abs(psi)).squeeze()


@partial(jax.jit, static_argnames=["model_apply", "is_log_model"])
def energy_fn(
    hamiltonian,
    params,
    batch: jnp.ndarray,
    model_apply: Callable,
    is_log_model: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute energy expectation value and standard error.

    Args:
        hamiltonian: Energy operator
        params: Model parameters
        batch: Training batch, shape (batch_size, DoF)
        model_apply: Model forward pass
        is_log_model: If model outputs log(ψ)

    Returns:
        E: Energy expectation value, scalar
        local_energies: E_loc for each sample, shape (batch, 1)
        sigma_e: Standard error of energy, scalar
    """
    local_energy_per_point = compute_local_energy(
        hamiltonian, params, batch, model_apply, is_log_model=is_log_model
    )
    E = jnp.mean(local_energy_per_point)
    sigma_e = jnp.std(local_energy_per_point)
    return E, local_energy_per_point, sigma_e


@partial(jax.jit, static_argnames=["model_apply", "is_log_model"])
def energy_and_grads(
    hamiltonian,
    params,
    batch: jnp.ndarray,
    model_apply: Callable,
    is_log_model: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Compute energy and gradients using variance minimization loss.

    The loss function is:
        L(θ) = 2⟨(E_loc(x) - ⟨E⟩) log|ψ(x)|⟩

    This biases the optimization toward low-energy states.

    Args:
        hamiltonian: Energy operator
        params: Model parameters (PyTree)
        batch: Training samples, shape (batch_size, DoF)
        model_apply: Model forward pass function
        is_log_model: If True, model outputs log(|ψ|)

    Returns:
        E: Energy expectation value
        sigma_e: Standard error of energy
        grads: Gradients w.r.t. parameters (PyTree, same shape as params)
    """
    E, E_loc, sigma_e = energy_fn(
        hamiltonian, params, batch, model_apply, is_log_model=is_log_model
    )

    # Construct loss function based on model type
    if not is_log_model:
        # For direct models: use log|ψ|
        def loss(p):
            log_psi_vals = log_psi(batch, p, model_apply).reshape(-1, 1)
            return 2 * jnp.mean(
                jax.lax.stop_gradient(E_loc - E) * log_psi_vals
            )
    else:
        # For log models: model output is already log|ψ|
        def loss(p):
            log_psi_vals = model_apply(p, batch).reshape(-1, 1)
            return 2 * jnp.mean(
                jax.lax.stop_gradient(E_loc - E) * log_psi_vals
            )

    # Compute gradients
    grads = jax.grad(loss)(params)
    return E, sigma_e, grads


@partial(jax.jit, static_argnames=["is_log_model", "use_qgt"])
def compute_step(
    state: VMCState,
    batch: jnp.ndarray,
    hamiltonian,
    is_log_model: bool = False,
    use_qgt: bool = False,
    qgt_config: dict = None,
) -> Tuple[VMCState, jnp.ndarray, jnp.ndarray]:
    """
    Perform one training step: compute energy/gradients and update parameters.

    Args:
        state: Current VMCState (contains params, optimizer state, etc.)
        batch: Training batch of samples, shape (batch_size, DoF)
        hamiltonian: Energy operator
        is_log_model: If model outputs log(ψ)
        use_qgt: If True, use natural gradient (QGT). Else vanilla gradient descent.
        qgt_config: Configuration dict for QGT solver (if use_qgt=True)

    Returns:
        new_state: Updated VMCState with new parameters
        energy: Energy expectation value at this step
        std: Standard error of energy
    """
    if qgt_config is None:
        qgt_config = DEFAULT_QGT_CONFIG.to_dict()

    # Compute energy and gradients
    E, sigma_e, grads = energy_and_grads(
        hamiltonian, state.params, batch, state.apply_fn, is_log_model=is_log_model
    )

    # Apply parameter update
    if not use_qgt:
        # Standard gradient descent using optimizer
        new_state = state.apply_gradients(grads=grads)
    else:
        # Natural gradient descent using QGT
        new_state = _apply_natural_gradient_step(state, grads, batch, qgt_config)

    return new_state, E, sigma_e, grads


def _apply_natural_gradient_step(
    state: VMCState, grads: dict, batch: jnp.ndarray, qgt_config: dict
) -> VMCState:
    """
    Apply natural gradient update using Quantum Geometric Tensor.

    The natural gradient is δθ = -η S^{-1}(θ) ∇E(θ),
    where S is the QGT and η is the learning rate.

    Args:
        state: Current training state
        grads: Computed gradients w.r.t. parameters
        batch: Training samples (used to compute QGT)
        qgt_config: QGT configuration (learning_rate, solver, etc.)

    Returns:
        Updated VMCState with natural gradient applied
    """
    # Compute QGT and solve for natural gradient
    natural_grad_flat, unravel_fn = compute_natural_gradient(
        state.params, batch, state.apply_fn, grads, qgt_config
    )

    # Extract learning rate from config
    learning_rate = qgt_config.get("learning_rate", 1e-3)

    # Update parameters: θ ← θ - η * natural_grad
    params_flat = ravel_pytree(state.params)[0]
    new_params_flat = params_flat - learning_rate * natural_grad_flat
    new_params = unravel_fn(new_params_flat)

    # Return updated state
    return state.replace(params=new_params)
