import jax
import jax.numpy as jnp
import numpy as np
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


def _dense_reference(model, params, x):
    """The pre-2026-11-24 evaluation: every pair distance against all M+3 B-splines."""
    from qvarnet.ansatz.jastrow import _cubic_bspline

    M, dr = model.n_knots, model.r_cut / model.n_knots
    c_free = params["params"]["coeffs"]
    c_m1 = c_free[1] - 2.0 * dr * model.cusp
    c = jnp.concatenate([c_m1[None], c_free, jnp.zeros(3)])
    k = jnp.arange(-1, M + 2).astype(x.dtype)
    dx = x[..., :, None] - x[..., None, :]
    if model.L is not None:
        dx = dx - model.L * jnp.round(dx / model.L)
    r = jnp.abs(dx)
    u = _cubic_bspline(r[..., None] / dr - jnp.broadcast_to(k, r.shape + k.shape)) @ c
    mask = jnp.triu(jnp.ones((model.n_particles, model.n_particles), bool), k=1)
    return jnp.sum(jnp.where(mask, u, 0.0), axis=(-2, -1))[..., None]


@pytest.mark.parametrize("L", [None, 7.0])
def test_local_evaluation_matches_dense(L):
    """Local (4-coefficient) evaluation == dense: value, ∇_x, ∇²_x and ∂/∂c, incl. r beyond r_cut (open)."""
    n = 5
    model = BSplineJastrow(n_particles=n, r_cut=3.5, n_knots=12, cusp=0.7, L=L)
    x = jax.random.uniform(jax.random.PRNGKey(0), (64, n), minval=0.0, maxval=10.0 if L is None else L)
    params = model.init(jax.random.PRNGKey(1), x)
    params = {"params": {"coeffs": jax.random.normal(jax.random.PRNGKey(2), params["params"]["coeffs"].shape)}}
    new = lambda p, xi: model.apply(p, xi[None])[0, 0]  # noqa: E731
    ref = lambda p, xi: _dense_reference(model, p, xi[None])[0, 0]  # noqa: E731
    lap = lambda f: jax.vmap(lambda xi: jnp.trace(jax.hessian(lambda y: f(params, y))(xi)))  # noqa: E731
    for f_new, f_ref in [
        (jax.vmap(lambda xi: new(params, xi)), jax.vmap(lambda xi: ref(params, xi))),
        (jax.vmap(jax.grad(lambda xi: new(params, xi))), jax.vmap(jax.grad(lambda xi: ref(params, xi)))),
        (lap(new), lap(ref)),
        (jax.vmap(lambda xi: jax.grad(new)(params, xi)["params"]["coeffs"]),
         jax.vmap(lambda xi: jax.grad(ref)(params, xi)["params"]["coeffs"])),
    ]:
        np.testing.assert_allclose(f_new(x), f_ref(x), rtol=1e-4, atol=1e-4)
