"""Training step computation for Variational Monte Carlo."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from ..geometry.qgt import (
    DEFAULT_QGT_CONFIG,
    QGTConfig,
    compute_natural_gradient,
    compute_natural_gradient_minsr,
)
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Compute energy and parameter gradients via the VMC variance-minimisation loss.

    Loss: L(θ) = 2⟨(E_loc - ⟨E⟩) log|ψ_θ|⟩ + Σ aux_loss(θ)

    Returns ``(E, sigma_e, E_loc, grads)`` — ``E_loc`` (per-sample) is surfaced so the
    caller can form per-chain energies for the stationarity diagnostics (step 4).
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
    return E, sigma_e, E_loc, grads


@partial(jax.jit, static_argnames=["n_chains", "use_qgt", "qgt_config", "auxiliary_losses"])
def compute_step(
    state: VMCState,
    batch: jnp.ndarray,
    hamiltonian,
    n_chains: int = 1,
    use_qgt: bool = False,
    qgt_config: QGTConfig = None,
    auxiliary_losses: tuple = (),
    key=None,
) -> tuple[VMCState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Perform one training step: energy/gradients, per-chain energies, param update.

    ``batch`` is the flattened ``(n_chains * n_eff, dof)`` MCMC batch (chain-contiguous,
    see ``samplers/step.py``); reshaping back to ``(n_chains, n_eff)`` recovers the
    per-chain mean local energies ``E_chain`` used by split-R̂ / Geweke.

    Returns ``(new_state, E, sigma_e, E_chain, grads)``.
    """
    if qgt_config is None:
        qgt_config = DEFAULT_QGT_CONFIG

    E, sigma_e, E_loc, grads = energy_and_grads(
        hamiltonian,
        state.params,
        batch,
        state.apply_fn,
        auxiliary_losses=auxiliary_losses,
        key=key,
    )
    E_chain = jnp.mean(E_loc.reshape(n_chains, -1), axis=1)  # (n_chains,)

    if not use_qgt:
        new_state = state.apply_gradients(grads=grads)
    else:
        # Stochastic reconfiguration as a *preconditioner*: precondition the gradient
        # with S⁻¹, then hand it to optax via apply_gradients. The step size η lives in
        # the optimizer (train() sets tx=optax.sgd(qgt_config.learning_rate) when
        # use_qgt), so this both advances state.step and honours optax LR schedules —
        # unlike the old manual state.replace(params=...) which did neither.
        if qgt_config.solver == "minsr":
            # Gram-dual SR — solve the M×M system from the energy residuals directly.
            if auxiliary_losses:
                raise ValueError(
                    "minSR does not support auxiliary losses: it preconditions the "
                    "energy gradient via the M×M Gram dual and has no access to aux "
                    "terms. Use a full-SR solver, or drop auxiliary_losses."
                )
            natural_grad_flat, unravel_fn = compute_natural_gradient_minsr(
                state.params, batch, E_loc, state.apply_fn, qgt_config
            )
        else:
            natural_grad_flat, unravel_fn = compute_natural_gradient(
                state.params, batch, state.apply_fn, grads, qgt_config
            )
        natural_grads = unravel_fn(natural_grad_flat)
        new_state = state.apply_gradients(grads=natural_grads)

    return new_state, E, sigma_e, E_chain, grads
