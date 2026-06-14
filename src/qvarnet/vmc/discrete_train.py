"""Sampled discrete VMC (roadmap step 3) — single-spin-flip MCMC training.

A lean sibling of ``vmc.train.train`` for discrete systems: it swaps in the spin-flip
sampler and a ``DiscreteHamiltonian`` but reuses ``compute_step`` (VMC gradient + SR/minSR),
``VMCState``, and ``MetricsHistory`` *unchanged* — the §7 method/space factorization in action.

Returns ``(TrainResult, final_state)`` so the trained parameters are recoverable (the discrete
testbed has no checkpoint callbacks).
"""

import time

import jax
import jax.numpy as jnp
import optax
from jax import random

from ..geometry.qgt import DEFAULT_QGT_CONFIG
from ..samplers.discrete import sample_spins
from .metrics_history import MetricsHistory
from .probability import build_prob_fn
from .train_result import TrainResult
from .training_step import compute_step
from .vmc_state import VMCState


def train_discrete(
    model,
    hamiltonian,
    optimizer,
    *,
    n_chains,
    n_epochs,
    rng_seed=0,
    chain_length=200,
    burn_in=50,
    thinning=2,
    use_qgt=False,
    qgt_config=None,
):
    """Train a discrete (spin) NQS by single-spin-flip VMC.

    Args mirror ``train`` where sensible. ``hamiltonian`` must expose ``n_spins`` and the
    standard ``local_energy(params, samples, model_apply, key)`` (e.g. ``TFIMHamiltonian``).
    On the QGT path the optimizer is replaced by ``optax.sgd(qgt lr)`` exactly as in ``train``.
    """
    n_spins = hamiltonian.n_spins
    if use_qgt:
        if qgt_config is None:
            qgt_config = DEFAULT_QGT_CONFIG
        optimizer = optax.sgd(qgt_config.learning_rate)

    key = random.PRNGKey(rng_seed)
    key, init_key, spin_key = random.split(key, 3)
    params = model.init(init_key, jnp.ones((1, n_spins)))
    state = VMCState.create(apply_fn=model.apply, params=params, tx=optimizer)
    prob_fn = build_prob_fn(model.apply)

    # Random ±1 initial spins, one config per chain.
    spins = jnp.where(random.bernoulli(spin_key, 0.5, (n_chains, n_spins)), 1.0, -1.0)

    history = MetricsHistory()
    for step in range(n_epochs):
        t0 = time.perf_counter()
        key, subkey = random.split(key)
        batch, spins, acceptance = sample_spins(
            subkey, prob_fn, state.params, spins, n_chains, n_spins, chain_length, burn_in, thinning
        )
        state, E, sigma_e, E_chain, _ = compute_step(
            state=state,
            batch=batch,
            hamiltonian=hamiltonian,
            n_chains=n_chains,
            use_qgt=use_qgt,
            qgt_config=qgt_config,
        )
        E, sigma_e, E_chain, acceptance = jax.device_get((E, sigma_e, E_chain, acceptance))
        history.append(
            {
                "step": step,
                "energy": float(E),
                "std": float(sigma_e),
                "error_of_mean": float(sigma_e / jnp.sqrt(batch.shape[0])),
                "E_chain": E_chain,
                "acceptance_rate": acceptance,
                "step_size": 0.0,  # single-spin-flip has no step size
                "cm_mean": 0.0,
                "cm_std": 0.0,
                "wall_time": time.perf_counter() - t0,
            }
        )

    return TrainResult(history=history), state
