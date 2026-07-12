"""minSR (Gram-dual SR) guards:

1. With the same regularisation ε, the minSR step equals the full-SR step exactly:
   (S+εI)⁻¹Ōᵀ = Ōᵀ(T+εI)⁻¹ (provable via the SVD of Ō), with S=ŌᵀŌ/M (P×P) and
   T=ŌŌᵀ/M (M×M). Test in the over-parametrised regime P>M where minSR is the point.
2. It trains end-to-end via train(use_qgt=True, solver="minsr"): step increments and
   the energy descends and stays sane (correctness is covered by guard 1).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import make_ho_model

from qvarnet.config.training_setup import TrainingConfig
from qvarnet.geometry.qgt import (
    QGTConfig,
    compute_natural_gradient,
    compute_natural_gradient_minsr,
)
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.train import train
from qvarnet.vmc.training_step import compute_local_energy


def test_minsr_matches_full_sr_same_reg():
    # Small over-parametrised problem: M samples < P params (minSR's home turf).
    model = make_ho_model()  # MLP[16,16] over 3 dof → P ~ a few hundred
    key = jax.random.PRNGKey(0)
    dof = 3
    M = 12
    batch = jax.random.normal(key, (M, dof)) * 0.5
    params = model.init(key, jnp.ones((1, dof)))
    P = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert P > M, f"expected over-parametrised P>M, got P={P}, M={M}"

    ham = HarmonicOscillatorHamiltonian(omega=1.0)
    e_loc = compute_local_energy(ham, params, batch, model.apply)

    # autodiff force F = ∇θ 2⟨(E_loc-Ē) logψ⟩ — what the full-SR path solves S δ = F with.
    def loss(p):
        log_psi = model.apply(p, batch).squeeze()
        return 2 * jnp.mean(jax.lax.stop_gradient(e_loc - jnp.mean(e_loc)) * log_psi)

    grads = jax.grad(loss)(params)

    cfg = QGTConfig(solver="cholesky", regularization=1e-2)
    full_flat, _, _ = compute_natural_gradient(params, batch, model.apply, grads, cfg)
    minsr_flat, _, _ = compute_natural_gradient_minsr(params, batch, e_loc, model.apply, cfg)

    diff = float(jnp.max(jnp.abs(minsr_flat - full_flat)))
    scale = float(jnp.max(jnp.abs(full_flat))) + 1e-12
    # Exact identity in infinite precision (same scale-invariant shift eps*tr/P on both
    # sides). The residual here is float32 round-off on two differently-conditioned
    # solves: cond ~ P/eps, so relative error ~ (P/eps)*2^-23 ~ 5e-3 at eps=1e-2
    # (vs O(1) for a genuine mismatch).
    assert diff / scale < 2e-2, f"minSR vs full-SR (same reg) relative diff {diff / scale:.2e}"


def test_minsr_trains_end_to_end(tmp_path):
    result = train(
        shape=(64, 1),
        model=make_ho_model(),
        optimizer=optax.sgd(0.02),  # classic SR update rule; keep in sync with qgt lr
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(
            # 300 epochs: recalibrated for the Jacobi/LM-regularised SR solve (the
            # per-direction ε damps the stiff envelope direction more than the old
            # absolute shift, so early progress is slower but numerically safe).
            n_epochs=300,
            rng_seed=0,
            use_qgt=True,
            checkpoint_path=str(tmp_path),
            # calibrated for cold-restart sampling (engine default is now True)
            warm_walkers=False,
        ),
        qgt_config=QGTConfig(solver="minsr", learning_rate=0.02, regularization=1e-2),
        sampler_params={
            "step_size": 0.6,
            "chain_length": 200,
            "thermalization_steps": 50,
            "thinning_factor": 2,
        },
    )
    energies = np.array([s.energy for s in result.history])
    tail = float(np.mean(energies[-20:]))
    assert energies[0] > tail, "minSR did not reduce the energy"
    assert tail < 5.0 and np.isfinite(tail), f"minSR energy not sane: tail={tail}"


def test_minsr_rejects_aux_losses(tmp_path):
    from qvarnet.config.training_setup import CuspConfig

    with pytest.raises(ValueError, match="minSR does not support auxiliary losses"):
        train(
            shape=(32, 2),
            model=make_ho_model(),
            optimizer=optax.sgd(1e-2),
            hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
            training_config=TrainingConfig(
                n_epochs=1,
                rng_seed=0,
                use_qgt=True,
                checkpoint_path=str(tmp_path),
                cusp=CuspConfig(L=1.0),
            ),
            qgt_config=QGTConfig(solver="minsr"),
            sampler_params={
                "step_size": 0.5,
                "chain_length": 100,
                "thermalization_steps": 20,
                "thinning_factor": 2,
            },
        )
