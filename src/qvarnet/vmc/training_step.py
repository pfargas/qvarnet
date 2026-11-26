"""Training step computation for Variational Monte Carlo."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from qvarnet.ansatz.output import split_log_psi
from qvarnet.optim.qgt import (
    DEFAULT_QGT_CONFIG,
    QGTConfig,
    compute_natural_gradient,
    compute_natural_gradient_minsr,
    resolve_qgt_solver,
)
from qvarnet.vmc.state import VMCState


def clip_local_energy(E_loc: jnp.ndarray, clip_scale: float) -> jnp.ndarray:
    """Clip E_loc to median ± clip_scale·MAD, MAD = mean |E_loc − median| (FermiNet convention).

    Gradient use only: report ⟨E⟩, σ_E and error bars from the unclipped E_loc.
    E_loc: (batch,) → (batch,).
    """
    assert E_loc.ndim == 1, E_loc.shape
    center = jnp.median(E_loc)
    mad = jnp.mean(jnp.abs(E_loc - center))
    return jnp.clip(E_loc, center - clip_scale * mad, center + clip_scale * mad)


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


@partial(jax.jit, static_argnames=["model_apply", "auxiliary_losses", "clip_scale"])
def energy_and_grads(
    hamiltonian,
    params,
    batch: jnp.ndarray,
    model_apply: Callable,
    auxiliary_losses: tuple = (),
    key=None,
    clip_scale: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Compute energy and parameter gradients via the VMC variance-minimisation loss.

    Loss: L(θ) = 2⟨(Ẽ_loc - ⟨Ẽ_loc⟩) log|ψ_θ|⟩ + Σ aux_loss(θ)

    Ẽ_loc = E_loc, or ``clip_local_energy(E_loc, clip_scale)`` when ``clip_scale`` is set.
    The residual is centred at the clipped mean so it sums to zero over the batch.

    Returns ``(E, sigma_e, E_loc, grads)`` — E, sigma_e and ``E_loc`` are always
    *unclipped*; ``E_loc`` (per-sample) is surfaced so the caller can form per-chain
    energies for the stationarity diagnostics (step 4).
    """
    E, E_loc, sigma_e = energy_fn(hamiltonian, params, batch, model_apply, key=key)
    if clip_scale is None:
        residual = E_loc - E
    else:
        E_loc_c = clip_local_energy(E_loc, clip_scale)
        residual = E_loc_c - jnp.mean(E_loc_c)

    def loss(p):
        log_psi = split_log_psi(model_apply(p, batch))[0].squeeze()
        assert log_psi.shape == residual.shape, (log_psi.shape, residual.shape)
        vmc = 2 * jnp.mean(jax.lax.stop_gradient(residual) * log_psi)
        if auxiliary_losses:
            aux = sum(aux_loss(p, model_apply, batch) for aux_loss in auxiliary_losses)
            return vmc + aux
        return vmc

    grads = jax.grad(loss)(params)
    return E, sigma_e, E_loc, grads


@partial(
    jax.jit,
    static_argnames=["n_chains", "use_qgt", "qgt_config", "auxiliary_losses", "clip_scale"],
)
def compute_step(
    state: VMCState,
    batch: jnp.ndarray,
    hamiltonian,
    n_chains: int = 1,
    use_qgt: bool = False,
    qgt_config: QGTConfig = None,
    auxiliary_losses: tuple = (),
    key=None,
    clip_scale: float | None = None,
) -> tuple[VMCState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Perform one training step: energy/gradients, per-chain energies, param update.

    ``batch`` is the flattened ``(n_chains * n_eff, dof)`` MCMC batch (chain-contiguous,
    see ``samplers/step.py``); reshaping back to ``(n_chains, n_eff)`` recovers the
    per-chain mean local energies ``E_chain`` used by split-R̂ / Geweke.

    Returns ``(new_state, E, sigma_e, E_chain, grads, sr_info)`` — ``sr_info`` is the
    guard-diagnostics dict from the natural-gradient solve (``fisher_norm``,
    ``trust_scale``, ``solve_ok``, ``nat_grad_norm``), empty when ``use_qgt=False``
    (``use_qgt`` is static, so the pytree structure is fixed per trace).
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
        clip_scale=clip_scale,
    )
    E_chain = jnp.mean(E_loc.reshape(n_chains, -1), axis=1)  # (n_chains,), unclipped

    sr_info = {}
    if not use_qgt:
        new_state = state.apply_gradients(grads=grads)
    else:
        # Stochastic reconfiguration as a *preconditioner*: precondition the gradient
        # with S⁻¹, then hand it to optax via apply_gradients. The step size η lives in
        # the optimizer (train() sets tx=optax.sgd(qgt_config.learning_rate) when
        # use_qgt), so this both advances state.step and honours optax LR schedules —
        # unlike the old manual state.replace(params=...) which did neither.
        _n_params = ravel_pytree(state.params)[0].size
        _solver = resolve_qgt_solver(qgt_config.solver, _n_params, batch.shape[0])
        if _solver == "minsr":
            # Gram-dual SR — solve the M×M system from the energy residuals directly.
            if auxiliary_losses:
                raise ValueError(
                    "minSR does not support auxiliary losses: it preconditions the "
                    "energy gradient via the M×M Gram dual and has no access to aux "
                    "terms. Use a full-SR solver, or drop auxiliary_losses."
                )
            # minSR builds its RHS from E_loc directly (it re-centres internally),
            # so it gets the same clipped values the loss used.
            e_for_grad = E_loc if clip_scale is None else clip_local_energy(E_loc, clip_scale)
            natural_grad_flat, unravel_fn, sr_info = compute_natural_gradient_minsr(
                state.params, batch, e_for_grad, state.apply_fn, qgt_config
            )
        else:
            natural_grad_flat, unravel_fn, sr_info = compute_natural_gradient(
                state.params, batch, state.apply_fn, grads, qgt_config
            )
        natural_grads = unravel_fn(natural_grad_flat)
        new_state = state.apply_gradients(grads=natural_grads)

    return new_state, E, sigma_e, E_chain, grads, sr_info
