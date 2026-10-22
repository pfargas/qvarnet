"""Output convention (log_abs, phase) and the slogdet FermionicMLP."""

import jax
import jax.numpy as jnp
import pytest

from qvarnet.ansatz import FermionicMLP, GaussianEnvelope, LogWavefunction
from qvarnet.ansatz.output import split_log_psi


def _fermion_model(n, d):
    return LogWavefunction(
        network=FermionicMLP(architecture=[16], n_fermions=n, n_dim=d), envelope=GaussianEnvelope()
    )


@pytest.mark.parametrize("n,d", [(3, 1), (3, 2)])
def test_swap_flips_sign_keeps_amplitude(n, d):
    model = _fermion_model(n, d)
    x = jax.random.normal(jax.random.PRNGKey(0), (64, n * d))
    params = model.init(jax.random.PRNGKey(1), x)

    p = x.reshape(64, n, d)
    x_swapped = p.at[:, [0, 1]].set(p[:, [1, 0]]).reshape(64, n * d)

    la, ph = model.apply(params, x)
    la_s, ph_s = model.apply(params, x_swapped)
    assert la.shape == ph.shape == (64,)
    assert jnp.allclose(la, la_s, atol=1e-5)
    assert jnp.allclose(jnp.cos(ph - ph_s), -1.0)  # phases differ by π
    assert jnp.all((ph == 0) | (ph == jnp.pi))


def test_single_output_is_bosonic_and_warns():
    with pytest.warns(UserWarning, match="bosonic"):
        log_abs, phase = split_log_psi(jnp.ones((5, 1)))
    assert log_abs.shape == phase.shape == (5,)
    assert jnp.all(phase == 0)
