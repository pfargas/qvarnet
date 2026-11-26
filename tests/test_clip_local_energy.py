"""E_loc clipping (TrainingConfig.clip_local_energy / clip_scale) guards:

1. clip_local_energy caps outliers at median ± scale·MAD and leaves inliers untouched.
2. Off, or with a window wider than every residual, the gradient is the unclipped one.
3. With an outlier: reported E is unclipped, the gradient changes, and the plain-gradient
   (full SR) and minSR paths see the same clipped values (minSR ≡ full SR still holds).
"""

import jax
import jax.numpy as jnp
from conftest import make_ho_model
from flax import struct

from qvarnet import TrainingConfig
from qvarnet.optim.qgt import QGTConfig, compute_natural_gradient, compute_natural_gradient_minsr
from qvarnet.physics.hamiltonian.continuous import HarmonicOscillatorHamiltonian
from qvarnet.vmc.training_step import clip_local_energy, energy_and_grads

SPIKE = 1e3


@struct.dataclass
class SpikedHamiltonian:
    """HO local energy with +SPIKE on sample 0: a controlled outlier."""

    inner: HarmonicOscillatorHamiltonian

    def local_energy(self, params, samples, model_apply, key=None):
        e = self.inner.local_energy(params, samples, model_apply, key=key).reshape(-1)
        return e.at[0].add(SPIKE)


def _setup(M=12, dof=3):
    model = make_ho_model()
    key = jax.random.PRNGKey(0)
    batch = jax.random.normal(key, (M, dof)) * 0.5
    params = model.init(key, jnp.ones((1, dof)))
    return model, params, batch


def _flat(tree):
    return jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(tree)])


def test_clip_function():
    e = jnp.array([1.0, 1.1, 0.9, 1.05, 0.95, 100.0])
    c = clip_local_energy(e, 5.0)
    assert c.shape == e.shape
    med = jnp.median(e)
    mad = jnp.mean(jnp.abs(e - med))
    assert jnp.isclose(c[-1], med + 5.0 * mad)
    assert jnp.allclose(c[:-1], e[:-1])


def test_wide_window_equals_unclipped():
    model, params, batch = _setup()
    ham = HarmonicOscillatorHamiltonian(omega=1.0)
    E0, s0, el0, g0 = energy_and_grads(ham, params, batch, model.apply)
    E1, s1, el1, g1 = energy_and_grads(ham, params, batch, model.apply, clip_scale=1e6)
    assert jnp.allclose(E0, E1) and jnp.allclose(s0, s1) and jnp.allclose(el0, el1)
    assert jnp.allclose(_flat(g0), _flat(g1), rtol=1e-5, atol=1e-7)


def test_outlier_clipped_in_gradient_only():
    model, params, batch = _setup()
    ham = SpikedHamiltonian(HarmonicOscillatorHamiltonian(omega=1.0))
    E0, s0, el0, g0 = energy_and_grads(ham, params, batch, model.apply)
    E1, s1, el1, g1 = energy_and_grads(ham, params, batch, model.apply, clip_scale=5.0)

    # Reported quantities are unclipped and identical.
    assert jnp.allclose(E0, E1) and jnp.allclose(s0, s1) and jnp.allclose(el0, el1)
    assert float(jnp.max(el1)) > SPIKE / 2
    # The outlier dominates the unclipped gradient; clipping must change it.
    assert not jnp.allclose(_flat(g0), _flat(g1), rtol=1e-2)

    # minSR fed the clipped E_loc reproduces full SR on the clipped-loss gradient.
    cfg = QGTConfig(solver="cholesky", regularization=1e-2)
    full, _, _ = compute_natural_gradient(params, batch, model.apply, g1, cfg)
    minsr, _, _ = compute_natural_gradient_minsr(
        params, batch, clip_local_energy(el1, 5.0), model.apply, cfg
    )
    rel = float(jnp.max(jnp.abs(minsr - full)) / (jnp.max(jnp.abs(full)) + 1e-12))
    assert rel < 2e-2, rel


def test_config_flag():
    assert TrainingConfig(n_epochs=1).clip_local_energy is False
    assert TrainingConfig(n_epochs=1).clip_scale == 5.0
    TrainingConfig(n_epochs=1, clip_local_energy=True, clip_scale=3.0)
    try:
        TrainingConfig(n_epochs=1, clip_scale=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("clip_scale=0 should raise")
