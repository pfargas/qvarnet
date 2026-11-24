from flax import linen as nn
from jax import numpy as jnp


class LogJastrow(nn.Module):
    """Bosonic Jastrow factor in log space, with one learnable exponent lambda.

    Open boundary (``L is None``, the default):
        log J = lambda * sum_{i<j} log|x_i - x_j|

    Periodic box of side ``L``:
        log J = lambda * sum_{i<j} log|sin(pi (x_i - x_j) / L)|

    The periodic form is the exactly L-periodic analogue, smooth everywhere except
    the physical coincidence cusp; see docs/explainers/periodic-systems.md for why
    a minimum-image log|x_i-x_j| is not a substitute.

    Applied to raw coordinates; combine with a network via ``LogWavefunction``.

    Args:
        n_particles: used to build the upper-triangle pair mask.
        lambda_init: initial value of the learnable exponent.
        L: box length. None (default) means open boundary.
    """

    n_particles: int
    lambda_init: float = 1.0
    L: float | None = None

    @nn.compact
    def __call__(self, x):
        # x: (..., n_particles)  — 1D coordinates (raw, pre-transform)
        lam = self.param("lambda", nn.initializers.constant(self.lambda_init), ())
        xi = x[..., :, None]  # (..., n, 1)
        xj = x[..., None, :]  # (..., 1, n)
        dx = xi - xj
        if self.L is None:
            r = jnp.abs(dx)
        else:
            r = jnp.abs(jnp.sin(jnp.pi * dx / self.L))  # exactly L-periodic
        mask = jnp.triu(jnp.ones((self.n_particles, self.n_particles), bool), k=1)
        log_r = jnp.where(mask, jnp.log(r + 1e-10), 0.0)
        return (lam * jnp.sum(log_r, axis=(-2, -1)))[..., None]


class OrderedLinear1DJastrow(nn.Module):
    n_particles: int
    a_rod: float = 1.0
    barrier: float = 1e4  # slope-1 linear penalty past the wall

    @nn.compact
    def __call__(self, x):
        xi = x[..., :, None]
        xj = x[..., None, :]
        idx = jnp.arange(self.n_particles)
        min_gap = self.a_rod * (idx[None, :] - idx[:, None])  # [i,j] = (j-i)*a_rod
        margin = (xj - xi) - min_gap  # [i,j]; i<j entries meaningful

        legal = margin > 0
        safe_margin = jnp.where(legal, margin, 1.0)  # never hand log() a non-positive value
        log_legal = jnp.log(safe_margin + 1e-10)
        log_illegal = (
            -self.barrier + margin
        )  # smooth, monotone falloff — not flat -> should change to exp maybe
        log_term = jnp.where(legal, log_legal, log_illegal)

        mask = jnp.triu(jnp.ones((self.n_particles, self.n_particles), bool), k=1)
        log_term = jnp.where(mask, log_term, 0.0)
        return (jnp.sum(log_term, axis=(-2, -1)))[..., None]


def _cubic_bspline(t):
    """Cardinal cubic B-spline centred at 0, support [-2, 2], C^2."""
    a = jnp.abs(t)
    inner = (4.0 - 6.0 * a**2 + 3.0 * a**3) / 6.0
    outer = jnp.clip(2.0 - a, 0.0) ** 3 / 6.0
    return jnp.where(a < 1.0, inner, outer)


class BSplineJastrow(nn.Module):
    """Pair Jastrow in log space with a learnable cubic B-spline u(r):

        log J = sum_{i<j} u(|x_i - x_j|),   u(r) = sum_k c_k B(r/dr - k),  dr = r_cut / n_knots

    Constraints (built in, not learned):
        u(r_cut) = u'(r_cut) = u''(r_cut) = 0, and u = 0 beyond r_cut  (c_{M-1..M+1} = 0)
        u'(0) = cusp                                                    (c_{-1} = c_1 - 2 dr cusp)

    cusp=0 (default) keeps log J smooth at coincidence; set it for contact
    interactions whose cusp you know. Cannot represent a log-singular core — use
    LogJastrow for that. Periodic box ``L``: minimum-image distance, needs
    r_cut <= L/2 (then log J is C^2 and exactly L-periodic).

    Args:
        n_particles: used to build the upper-triangle pair mask.
        r_cut: cutoff radius.
        n_knots: number of intervals M on [0, r_cut]; M - 1 free coefficients.
        cusp: imposed u'(0).
        L: box length. None (default) means open boundary.
    """

    n_particles: int
    r_cut: float
    n_knots: int = 8
    cusp: float = 0.0
    L: float | None = None

    @nn.compact
    def __call__(self, x):
        # x: (..., n_particles)  — 1D coordinates (raw, pre-transform)
        if self.L is not None and self.r_cut > self.L / 2:
            raise ValueError(
                f"r_cut={self.r_cut} > L/2={self.L / 2}: periodic u(r) would not be smooth."
            )
        M = self.n_knots
        dr = self.r_cut / M
        c_free = self.param("coeffs", nn.initializers.zeros, (M - 1,))  # c_0 .. c_{M-2}
        c_m1 = c_free[1] - 2.0 * dr * self.cusp if M > 2 else -2.0 * dr * self.cusp
        c = jnp.concatenate([c_m1[None], c_free, jnp.zeros(3)])  # c_{-1} .. c_{M+1}

        dx = x[..., :, None] - x[..., None, :]
        if self.L is not None:
            dx = dx - self.L * jnp.round(dx / self.L)
        r = jnp.abs(dx)
        # Local evaluation: at t = r/dr only the 4 B-splines k = i-1 .. i+2 (i = floor t) are nonzero,
        # so gather those 4 coefficients instead of evaluating all M+3 (memory (..., n, n, 4), not M+3).
        t = r / dr
        i = jnp.clip(jnp.floor(t), 0, M).astype(jnp.int32)
        j = jnp.arange(4).reshape((1,) * t.ndim + (4,))
        w = _cubic_bspline((t - i)[..., None] + 1.0 - j)  # B(t - k), k = i - 1 + j
        c_pad = jnp.concatenate([c, jnp.zeros(1, c.dtype)])  # c_{-1} .. c_{M+2}: c_pad[i + j] = c_k
        # Read from G copies of c (copy = position mod G): the gradient w.r.t. c is a scatter-add, and with one
        # copy all B·n²·4 additions collide on M+4 slots (GPU, N = 84, 10^4 samples: 1.2 s → 35 ms with G = 64). G small: SR keeps a
        # per-sample gradient of all G copies (G = 1024 needed 16 GiB there).
        G = 64
        g = (jnp.arange(t.size, dtype=jnp.int32).reshape(t.shape) % G)[..., None]
        c_rep = jnp.broadcast_to(c_pad, (G, M + 4))
        u = jnp.sum(w * c_rep[g, i[..., None] + j], axis=-1)  # (..., n, n)
        u = jnp.where(t < M, u, 0.0)  # u = 0 beyond r_cut (only reached without PBC)
        mask = jnp.triu(jnp.ones((self.n_particles, self.n_particles), bool), k=1)
        return jnp.sum(jnp.where(mask, u, 0.0), axis=(-2, -1))[..., None]
