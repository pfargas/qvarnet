"""Full-sum (exact) VMC for small discrete systems (roadmap step 3).

The continuum analogue of summing over a grid: for N ≲ ~14 spins, evaluate ⟨E⟩ by summing
over *all* 2^N configurations weighted by |ψ_θ|² — deterministic, no MCMC, no sampling noise.
This isolates architecture + optimiser from sampling (the notebook's exact-gradient bakeoff)
and validates the discrete Hamiltonian against exact diagonalisation.

It also proves the method/space factorisation: the same model and the same connected-elements
local energy used by the (future) MCMC flip-kernel path run here unchanged.
"""

import jax
import jax.numpy as jnp
import optax


def all_spin_configs(n: int) -> jnp.ndarray:
    """All 2^n spin configs as (2^n, n) of ±1 — matches ``utils.exact_diag`` ordering."""
    a = jnp.arange(2**n)
    bits = (a[:, None] >> jnp.arange(n)[None, :]) & 1
    return (1 - 2 * bits).astype(jnp.float32)


def full_sum_energy(model, params, hamiltonian):
    """Exact ⟨E⟩ = Σ_s |ψ(s)|² E_loc(s) / Σ_s |ψ(s)|² over all 2^N configurations.

    Differentiable in ``params`` → exact (noise-free) gradients for ``jax.grad``.
    """
    configs = all_spin_configs(hamiltonian.n_spins)

    def logpsi_fn(c):
        return model.apply(params, c).squeeze(-1)

    logpsi = logpsi_fn(configs)  # (2^N,)
    e_loc = hamiltonian.local_energy_logpsi(logpsi_fn, configs)  # (2^N,)
    # |ψ|² weights, shifted for numerical stability (overall scale cancels).
    w = jnp.exp(2.0 * (logpsi - jnp.max(logpsi)))
    return jnp.sum(w * e_loc) / jnp.sum(w)


def train_full_sum(model, params, hamiltonian, optimizer, n_steps: int):
    """Deterministic gradient descent on the exact full-sum energy.

    Returns ``(final_params, energies)`` where ``energies`` is the per-step ⟨E⟩ trace.
    """
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        e, grads = jax.value_and_grad(lambda p: full_sum_energy(model, p, hamiltonian))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, e

    energies = []
    for _ in range(n_steps):
        params, opt_state, e = step(params, opt_state)
        energies.append(float(e))
    return params, energies
