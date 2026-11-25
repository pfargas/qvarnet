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


@pytest.mark.parametrize("L", [None, 7.0])
def test_derivs_match_autodiff(L):
    """Closed-form ∂_a log J and ∂²_a log J == jax.grad / Hessian diagonal (float64), incl. r beyond r_cut."""
    n = 6
    model = BSplineJastrow(n_particles=n, r_cut=3.5, n_knots=12, cusp=0.7, L=L)
    with jax.enable_x64(True):
        x = jax.random.uniform(jax.random.PRNGKey(0), (32, n), jnp.float64, maxval=10.0 if L is None else L)
        x = x.at[0, 1].set(x[0, 0] + 1e-3)  # a pair near contact
        c = jax.random.normal(jax.random.PRNGKey(2), (model.n_knots - 1,), jnp.float64)
        params = {"params": {"coeffs": c}}
        g, d2 = model.apply(params, x, method="derivs")
        assert g.shape == d2.shape == x.shape
        f = lambda xi: model.apply(params, xi[None])[0, 0]  # noqa: E731
        np.testing.assert_allclose(g, jax.vmap(jax.grad(f))(x), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d2, jax.vmap(lambda xi: jnp.diag(jax.hessian(f)(xi)))(x), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("method", ["forward_ad", "folx"])
def test_analytic_kinetic_matches_autodiff(method):
    """E_loc with the Jastrow via derivs (auto) == full Hessian of everything: MLP + Jastrow, masses, PBC, f64.

    folx alone is ~1e-6 off the full Hessian on this model (also without the Jastrow), hence its tolerance."""
    from qvarnet.ansatz import LogWavefunction
    from qvarnet.ansatz.mlp import MLP
    from qvarnet.physics.boundaries import PeriodicBoundary
    from qvarnet.physics.hamiltonian.continuous import HarmonicOscillatorHamiltonian
    from qvarnet.physics.particles import Particles

    n, L = 5, 6.0
    model = LogWavefunction(network=MLP(hidden=[8]), transform=PeriodicBoundary(L),
                            jastrow=BSplineJastrow(n_particles=n, r_cut=L / 2, n_knots=10, cusp=0.5, L=L))
    promo = "allow" if method == "folx" else "raise"  # folx's own internals broadcast implicitly
    with jax.enable_x64(True), jax.numpy_rank_promotion(promo):
        x = jax.random.uniform(jax.random.PRNGKey(0), (16, n), jnp.float64, maxval=L)
        params = model.init(jax.random.PRNGKey(1), x)
        params = jax.tree.map(lambda a: jax.random.normal(jax.random.PRNGKey(2), a.shape, jnp.float64), params)
        if method == "folx":
            pytest.importorskip("folx")
        pt = Particles(n=n, masses=(1.0, 2.0, 0.5, 1.0, 3.0))
        assert model.analytic_parts()  # so model.apply takes the closed-form path
        e_an = HarmonicOscillatorHamiltonian(laplacian_method=method, particles=pt).local_energy(params, x, model.apply)
        e_ref = HarmonicOscillatorHamiltonian(laplacian_method="full_hessian", particles=pt).local_energy(
            params, x, lambda p, y: model.apply(p, y))  # a plain function has no __self__: all AD
        assert e_an.shape == e_ref.shape == (16,)
        np.testing.assert_allclose(e_an, e_ref, rtol=1e-11 if method == "forward_ad" else 1e-5)


def test_derivs_float32_accuracy():
    """float32 derivs vs float64 truth: catches reduced-precision (TF32) matmuls on GPU (gave ~1e-3)."""
    n = 40
    model = BSplineJastrow(n_particles=n, r_cut=20.0, n_knots=200, cusp=0.7, L=40.0)
    rng = np.random.default_rng(0)
    x = np.mod(np.arange(n) + 0.5 + 0.15 * rng.standard_normal((64, n)), 40.0).astype(np.float32)
    c = np.cumsum(rng.standard_normal(model.n_knots - 1)).astype(np.float32) * 0.05  # smooth, like a trained u
    g32, d32 = model.apply({"params": {"coeffs": jnp.asarray(c)}}, jnp.asarray(x), method="derivs")
    with jax.enable_x64(True):
        g64, d64 = model.apply({"params": {"coeffs": jnp.asarray(c, jnp.float64)}}, jnp.asarray(x, jnp.float64),
                               method="derivs")
    for a, b in [(g32, g64), (d32, d64)]:
        assert float(np.max(np.abs(np.asarray(a, np.float64) - b)) / np.max(np.abs(b))) < 1e-4  # roundoff ~1e-5
