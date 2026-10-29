import jax
import jax.numpy as jnp
import pytest

from qvarnet.ansatz import BSplineJastrow

jax.config.update("jax_numpy_rank_promotion", "raise")


def _u(model, params, r):
    """Single-pair u(r) via a 2-particle configuration."""
    return model.apply(params, jnp.array([0.0, r]))[0]


@pytest.mark.parametrize("cusp", [0.0, 0.7])
def test_constraints(cusp):
    m = BSplineJastrow(n_particles=2, r_cut=2.0, n_knots=6, cusp=cusp)
    p = m.init(jax.random.PRNGKey(0), jnp.zeros(2))
    p = jax.tree.map(lambda a: jax.random.normal(jax.random.PRNGKey(1), a.shape), p)

    def u(r):
        return _u(m, p, r)

    assert jnp.allclose(jax.grad(u)(1e-6), cusp, atol=1e-4)
    for d in (u, jax.grad(u), jax.grad(jax.grad(u))):
        assert jnp.abs(d(2.0 - 1e-6)) < 1e-4
    assert u(2.5) == 0.0


def test_periodic_and_shape():
    L = 5.0
    m = BSplineJastrow(n_particles=4, r_cut=2.0, L=L)
    p = m.init(jax.random.PRNGKey(0), jnp.zeros(4))
    p = jax.tree.map(lambda a: jax.random.normal(jax.random.PRNGKey(1), a.shape), p)
    x = jax.random.uniform(jax.random.PRNGKey(2), (3, 4), maxval=L)
    out = m.apply(p, x)
    assert out.shape == (3, 1)
    assert jnp.allclose(out, m.apply(p, x.at[:, 0].add(L)), atol=1e-5)
