r"""Analytic two-body soft-core scattering Jastrow (3D, periodic box).

Implements the spec in ``PhD/comments/softcore_jastrow_spec.md``: the log of the zero-energy,
s-wave (l=0) scattering solution of the soft-core (square-barrier) pair potential, summed over
distinct pairs and evaluated with the minimum-image convention. This is the short-range
correlation factor a one-body DeepSet cannot represent (the r_ij-dependent correlation hole);
it is *piped into the engine* via ``LogWavefunction(jastrow=...)`` — no qvarnet change needed —
so the full ansatz is ``log Ψ = Σ_{i<j} j(r_ij) + log Ψ_NN`` (spec §9).

Lives in the soft-sphere layer (not the engine) because it is problem-specific: paper units
(a_s = 1), 3D, and parameters fixed by the ``Potential`` (``α = K₀ = √(V₀/2)``, ``Rc = R`` — see
``dilute_gas.softcore_jastrow_params``).

Pair log-factor j(r) = log u(r) − log r, with u the zero-energy radial solution (spec §2–3):

    r < Rc : log(sinh(α r)) − log r − log α − log(cosh(α Rc))
    r ≥ Rc : log(1 − 1/r)        (node at r = a_s = 1; valid only for r > 1, and Rc > 1)

Branch on **Rc** (never on 1): the sinh branch covers all r < Rc, so the outside form is never
hit below Rc > 1. Both branches are guarded so the dead branch can't poison reverse-mode
gradients (spec §5).

First pass uses the bare outside tail (not strictly periodic at L/2, spec §8) and a **frozen**
α/Rc (the bare two-body value). Promoting α to a variational ``self.param`` (spec §9) is a later
refinement; the NN term absorbs medium/long-range and the PBC tail meanwhile.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


def _log_sinhc(z):
    """Stable log(sinh(z)/z) for z ≥ 0 — finite at z→0 (→0), no overflow at large z.

    The inside branch of j(r) is ``log(sinh(αr)) − log r − log α``; the log-r terms cancel
    analytically into ``log(sinh(αr)/(αr))``, which removes the r→0 (coincident-pair) divergence
    that breaks the naive ``log(sinh) − log(r)`` form (``exp(-2z)`` rounds to 1 at tiny z → −inf).
    Series for small z, exp form for large z; double-``where`` so the dead branch can't poison grads.
    """
    small = 1e-3
    z_safe = jnp.where(z < small, 1.0, z)  # keep the large-z branch finite at z→0
    big = (z_safe + jnp.log1p(-jnp.exp(-2.0 * z_safe)) - jnp.log(2.0)) - jnp.log(z_safe)
    series = jnp.log1p(
        z * z / 6.0 + z**4 / 120.0
    )  # log(sinh z / z) = z²/6 + ... near 0
    return jnp.where(z < small, series, big)


def _log_cosh(x):
    """Stable log(cosh(x)) (no cosh overflow when α Rc is large, i.e. near the hard-sphere limit)."""
    ax = jnp.abs(x)
    return ax + jnp.log1p(jnp.exp(-2.0 * ax)) - jnp.log(2.0)


def pair_log_jastrow(r, alpha: float, Rc: float):
    """Per-pair log-factor j(r) on already-min-imaged distances ``r`` (paper units, a_s = 1).

    NaN-safe: the inside form is the cancelled ``log(sinh(αr)/(αr)) − log cosh(αRc)`` (finite at
    r=0); the outside argument is clamped > 1 where its branch is dead, so the masked-out branch
    (evaluated anyway by ``jnp.where``) stays finite and its gradient is clean.
    """
    inside = _log_sinhc(alpha * r) - _log_cosh(alpha * Rc)  # r < Rc; finite even at r→0
    r_out = jnp.where(r >= Rc, r, 2.0)  # r ≥ Rc: keep arg > 1 where dead
    outside = jnp.log1p(-1.0 / r_out)
    return jnp.where(r < Rc, inside, outside)


class SoftCoreJastrow(nn.Module):
    """log J = Σ_{i<j} j(r_ij) for N bosons in a cubic PBC box of side ``L`` (3D, paper units).

    Applied by ``LogWavefunction`` to the **raw** coordinates ``x`` of shape ``(..., N*n_dim)``;
    returns ``(..., 1)`` to be summed into log|ψ|. ``alpha``/``Rc`` come from
    ``dilute_gas.softcore_jastrow_params``; both are frozen scalars (no parameters), so this module
    has an empty parameter pytree and adds zero optimiser cost.
    """

    n_particles: int
    n_dim: int
    L: float
    alpha: float
    Rc: float
    eps_reg: float = 1e-6  # guard against r_ij=0 (coincident pair) in the gradient

    @nn.compact
    def __call__(self, x):
        # Compute ONLY the N(N-1)/2 distinct i<j pairs (triu gather), not the full N×N matrix:
        # half the FLOPs/memory and no diagonal/mask. The (i,j) index arrays are static (built from
        # the static n_particles at trace time). Net win grows with N — at N=256 this ~halves both
        # the Laplacian time and the (batch,pairs,d) intermediate vs the dense N×N broadcast; at
        # small N the dense op is marginally faster (gather overhead) but negligible vs the epoch.
        pos = x.reshape(*x.shape[:-1], self.n_particles, self.n_dim)  # (..., N, d)
        i, j = jnp.triu_indices(self.n_particles, k=1)                # each (N(N-1)/2,)
        d = pos[..., i, :] - pos[..., j, :]                           # (..., P, d), P=N(N-1)/2
        d = d - self.L * jnp.round(d / self.L)                        # minimum image
        r = jnp.sqrt(jnp.sum(d**2, axis=-1) + self.eps_reg)           # (..., P); eps: 0-grad guard
        # C¹ cutoff at r_c = L/2 so log J is smooth *and periodic* on the torus (spec §8). Subtract
        # the raw 2-body j's value AND slope at r_c, then zero it beyond ⇒ j_cut(r_c)=j_cut'(r_c)=0.
        # This leaves the short-range 2-body physics (hole/cusp at r∼a) intact up to a tiny linear
        # tilt; only the long-range tail (pairs already ~uncorrelated) is healed. WITHOUT it the
        # non-periodic tail breaks the ⟨T⟩=½⟨|∇logψ|²⟩≥0 identity ⇒ negative kinetic, E/N below 4πx.
        # r_c = L/2 > Rc (guaranteed by box_fits_interaction), so r_c is in the outside branch:
        # j(r_c)=log(1−1/r_c), j'(r_c)=1/(r_c²−r_c) — analytic, no nested AD inside the Laplacian.
        rc = 0.5 * self.L
        j_rc = jnp.log1p(-1.0 / rc)
        jp_rc = 1.0 / (rc * rc - rc)
        jr = pair_log_jastrow(r, self.alpha, self.Rc) - j_rc - (r - rc) * jp_rc  # (..., P)
        jr = jnp.where(r < rc, jr, 0.0)                              # heal to 0 beyond L/2
        return jnp.sum(jr, axis=-1)[..., None]                        # (..., 1)
