"""Step 4 follow-up: optimization diagnostics (grad norms/SNR, θ-step, QGT spectrum)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from conftest import make_ho_model

from qvarnet.config.training_setup import TrainingConfig
from qvarnet.diagnostics import (
    d_eff,
    d_part,
    dead_fraction,
    global_grad_norm,
    global_theta_ratio,
    gradient_snr,
    per_layer_grad_norms,
    qgt_eigenvalues,
    theta_ratios,
)
from qvarnet.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.train import train


def test_grad_norm_matches_manual():
    grads = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([[0.0, 0.0]])}
    assert global_grad_norm(grads) == 5.0  # sqrt(9+16)
    per = per_layer_grad_norms(grads)
    assert len(per) == 2 and any(abs(v - 5.0) < 1e-6 for v in per.values())


def test_gradient_snr_shape_and_finiteness():
    rng = np.random.default_rng(0)
    M, P = 200, 12
    o_mat = rng.standard_normal((M, P))
    e_loc = rng.standard_normal(M)
    snr = gradient_snr(o_mat, e_loc)
    assert snr.shape == (P,) and np.all(np.isfinite(snr)) and np.all(snr >= 0)


def test_theta_ratios_and_dead_fraction():
    old = {"w": jnp.array([1.0, 2.0, 4.0])}
    new = {"w": jnp.array([1.0, 2.0, 4.0])}  # no change
    assert global_theta_ratio(old, new) == 0.0
    assert dead_fraction(old, new) == 1.0
    new2 = {"w": jnp.array([2.0, 2.0, 4.0])}  # one param moved a lot
    assert global_theta_ratio(old, new2) > 0.0
    assert set(theta_ratios(old, new2).keys()) == {
        jax.tree_util.keystr((jax.tree_util.DictKey("w"),))
    }


def test_qgt_spectrum_d_eff_d_part():
    # diagonal spectrum: 5 large eigenvalues, rest tiny
    eigs = np.array([1e-9, 1e-9, 1e-9, 0.3, 0.5, 0.7, 0.9, 1.0])
    assert d_eff(eigs, eps=1e-3) == 5
    # uniform spectrum → participation ratio == count
    assert abs(d_part(np.ones(10)) - 10.0) < 1e-6


def test_qgt_eigenvalues_real_psd():
    model = make_ho_model()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
    batch = jax.random.normal(jax.random.PRNGKey(1), (64, 2)) * 0.5
    eigs = qgt_eigenvalues(params, batch, model.apply, regularization=1e-6)
    P = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert eigs.shape == (P,)
    assert np.all(eigs > -1e-5)  # PSD up to round-off


def test_training_records_opt_diagnostics(tmp_path):
    result = train(
        shape=(32, 2),
        model=make_ho_model(),
        optimizer=optax.adam(1e-2),
        hamiltonian=HarmonicOscillatorHamiltonian(omega=1.0),
        training_config=TrainingConfig(n_epochs=5, rng_seed=0, checkpoint_path=str(tmp_path)),
        sampler_params={
            "step_size": 0.5,
            "chain_length": 100,
            "thermalization_steps": 20,
            "thinning_factor": 2,
        },
    )
    gn = result.history.get("grad_norm")
    tr = result.history.get("theta_ratio")
    assert gn.shape == (5,) and np.all(np.isfinite(gn)) and np.all(gn >= 0)
    assert tr.shape == (5,) and np.all(np.isfinite(tr)) and np.all(tr >= 0)
