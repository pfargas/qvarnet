"""Golden tests for the stabilised SR stack (centred QGT, auto solver, trust region,
recipes). Pins the 2026-07 numerical fixes so config refactors stay safe."""

import jax.numpy as jnp
import numpy as np

from qvarnet.boundaries import NoBoundary
from qvarnet.config.coord_mode import LabCoords
from qvarnet.geometry.qgt import (
    QGTConfig,
    compute_natural_gradient,
    compute_qgt,
    resolve_qgt_solver,
)
from qvarnet.hamiltonian.continuous import CalogeroSutherlandHamiltonian
from qvarnet.models.compose import LogWavefunction
from qvarnet.models.envelopes import GaussianEnvelope
from qvarnet.models.jastrow import LogJastrow
from qvarnet.models.mlp import MLP
from qvarnet.recipes import adam_train, sr_train
from qvarnet.train import train


def _linear_apply(p, x):
    # log|psi| = x @ p  ->  O(x) = x, so S = Cov(x) exactly.
    return x @ p


def test_centered_qgt_matches_covariance_and_stays_psd():
    """Centred build == Cov(O) + eps*I, and stays PSD in float32 even when the
    log-derivatives have large means (the catastrophic-cancellation regime that made
    the old uncentred build produce negative eigenvalues and non-descent steps)."""
    rng = np.random.default_rng(0)
    # large mean (1e3) with small fluctuations: worst case for uncentred f32 build
    x = jnp.asarray(rng.normal(1e3, 1.0, size=(256, 8)), dtype=jnp.float32)
    p = jnp.zeros(8, dtype=jnp.float32)
    eps = 1e-2
    S, O_mean = compute_qgt(p, x, _linear_apply, eps)

    cov = np.cov(np.asarray(x, dtype=np.float64).T, bias=True)
    # scale-invariant regularisation: the shift is eps * mean(diag(cov)), not eps
    shift = eps * float(np.mean(np.diag(cov)))
    assert np.allclose(np.asarray(S) - shift * np.eye(8), cov, rtol=1e-2, atol=1e-2)
    eigs = np.linalg.eigvalsh(np.asarray(S))
    assert eigs.min() > 0, f"S must stay PSD in float32, got min eig {eigs.min()}"


def test_resolve_auto_solver():
    assert resolve_qgt_solver("auto", n_params=100, n_samples=50) == "minsr"
    assert resolve_qgt_solver("auto", n_params=50, n_samples=100) == "cholesky"
    assert resolve_qgt_solver("direct", n_params=100, n_samples=50) == "direct"


def test_trust_region_units_resolve():
    """max_state_change is physical (state change per step): Δ = msc/lr at any lr —
    the units trap that let QGTConfig(learning_rate=0.02) run 20× the validated
    budget when the default cap was in direction units."""
    assert QGTConfig(learning_rate=1e-3, max_state_change=0.1).resolve_trust_region() == 100.0
    assert QGTConfig(learning_rate=0.02, max_state_change=0.1).resolve_trust_region() == 5.0
    # direction-space override wins
    assert QGTConfig(learning_rate=0.02, trust_region=7.0).resolve_trust_region() == 7.0
    # fully off
    assert QGTConfig(max_state_change=None).resolve_trust_region() is None
    # schedules cannot be divided by — require the explicit override
    import pytest

    with pytest.raises(ValueError, match="schedule"):
        QGTConfig(learning_rate=lambda step: 1e-3).resolve_trust_region()


def test_trust_region_caps_fisher_norm():
    """With trust_region set, the returned step has sqrt(d^T S d) <= trust_region."""
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(0.0, 1.0, size=(128, 6)), dtype=jnp.float32)
    params = {"w": jnp.zeros(6)}
    grads = {"w": jnp.asarray(rng.normal(0.0, 1e4, size=6), dtype=jnp.float32)}

    def apply_fn(p, xx):
        return xx @ p["w"]

    delta = 0.05
    cfg = QGTConfig(solver="cholesky", regularization=1e-2, trust_region=delta)
    d, unravel, info = compute_natural_gradient(params, x, apply_fn, grads, cfg)
    assert float(info["trust_scale"]) < 1.0  # the cap reports itself as binding
    S, _ = compute_qgt(jnp.zeros(6), x, _linear_apply, 1e-2)
    fisher_norm = float(jnp.sqrt(d @ (S @ d)))
    assert fisher_norm <= delta * 1.01, f"trust region violated: {fisher_norm} > {delta}"

    # fully off: max_state_change would otherwise derive a cap of 0.1/lr
    cfg_off = QGTConfig(
        solver="cholesky", regularization=1e-2, trust_region=None, max_state_change=None
    )
    d_off, _, _ = compute_natural_gradient(params, x, apply_fn, grads, cfg_off)
    assert float(jnp.sqrt(d_off @ (S @ d_off))) > delta  # cap was actually binding


# ── golden end-to-end: tiny CS system through the recipes ────────────────────────

N_P, CS_L = 4, 0.8
SHAPE = (64, N_P)
E_EXACT = N_P * (1 + CS_L * (N_P - 1))  # 13.6


def _model():
    return LogWavefunction(
        transform=NoBoundary(),
        network=MLP(hidden=[8]),
        envelope=GaussianEnvelope(),
        jastrow=LogJastrow(n_particles=N_P, lambda_init=CS_L),  # cusp-exact init
    )


def _ham():
    return CalogeroSutherlandHamiltonian(L=CS_L, epsilon=1e-12)


def _quiet(kw):
    import dataclasses

    kw["training_config"] = dataclasses.replace(kw["training_config"], print_summary=False)
    return kw


def test_golden_cs_adam(tmp_path):
    """adam_train runs finite on a tiny CS system and lands at a sane energy scale."""
    kw = _quiet(
        adam_train(n_epochs=60, learning_rate=1e-2, checkpoint_path=str(tmp_path), warmup_steps=100)
    )
    result = train(shape=SHAPE, model=_model(), hamiltonian=_ham(), coord_mode=LabCoords(), **kw)
    energies = np.array([float(s.energy) for s in result.history])
    assert np.all(np.isfinite(energies))
    assert energies[-1] < 10 * E_EXACT, f"adam: E diverged to {energies[-1]}"
    assert result.final_positions.shape == SHAPE
    assert result.final_step_size > 0


def test_golden_cs_adam_then_sr(tmp_path):
    """The intended workflow: Adam explore → SR finetune via prev_result.

    Checks the full carry-over (params, walkers, adapted step size), that the SR run
    stays finite (no spike blow-up), and that it does not undo Adam's convergence.
    (SR *from a cold wide init* is intentionally slow — the trust region throttles the
    state change per step — so the golden test exercises the warm-started path.)
    """
    kw1 = _quiet(
        adam_train(
            n_epochs=60, learning_rate=1e-2, checkpoint_path=str(tmp_path / "a"), warmup_steps=100
        )
    )
    r1 = train(shape=SHAPE, model=_model(), hamiltonian=_ham(), coord_mode=LabCoords(), **kw1)

    kw2 = sr_train(n_epochs=40, prev_result=r1, checkpoint_path=str(tmp_path / "b"))
    assert kw2["init_params"] is not None
    assert not isinstance(kw2["initial_chain_config"].init_positions, str)
    assert kw2["sampler_params"]["step_size"] == r1.final_step_size
    r2 = train(
        shape=SHAPE, model=_model(), hamiltonian=_ham(), coord_mode=LabCoords(), **_quiet(kw2)
    )

    e1 = np.array([float(s.energy) for s in r1.history])
    e2 = np.array([float(s.energy) for s in r2.history])
    assert np.all(np.isfinite(e2)), "SR finetune produced non-finite energies"
    # SR must not blow up what Adam converged: stay within a loose factor of the
    # Adam endpoint (covers MC noise while catching any spike-runaway regression).
    assert e2[-1] < max(3 * abs(e1[-1]), 10 * E_EXACT), (
        f"SR finetune diverged: adam ended at {e1[-1]}, sr at {e2[-1]}"
    )


def _optimizer_state_names(opt_state):
    """Flatten an optax state tree into the set of state-class names."""
    names = {opt_state.__class__.__name__}
    if isinstance(opt_state, (tuple, list)):
        for s in opt_state:
            names |= _optimizer_state_names(s)
    return names


def test_use_qgt_honours_passed_optimizer(tmp_path):
    """SR is a preconditioner, the optimizer is the update rule: passing Adam under
    use_qgt must run Adam on the natural gradient (the old silent SGD override bit
    twice — notebook cells ran sgd(1e-3) while claiming adam)."""
    import optax

    from qvarnet.config.training_setup import TrainingConfig

    captured = {}

    class _Grab:
        def on_step_end(self, step, state, metrics):
            captured["opt_state"] = state.opt_state
            return False

        def on_train_end(self, state, history):
            pass

    train(
        shape=SHAPE,
        model=_model(),
        optimizer=optax.adam(1e-3),
        hamiltonian=_ham(),
        coord_mode=LabCoords(),
        training_config=TrainingConfig(
            n_epochs=2,
            rng_seed=0,
            use_qgt=True,
            checkpoint_path=str(tmp_path),
            print_summary=False,
        ),
        callbacks=[_Grab()],
    )
    names = _optimizer_state_names(captured["opt_state"])
    assert "ScaleByAdamState" in names, f"passed Adam was not used; opt state: {names}"
