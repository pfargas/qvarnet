r"""Dilute Bose-gas parametrization for the hard-/soft-sphere study.

Reproduces the control-parameter layer of Mazzanti, Polls & Fabrocini,
*"Energy and structure of dilute hard- and soft-sphere Bose gases"*
(arXiv:cond-mat/0305502). Units: :math:`\hbar = m = 1`, so the kinetic term is
:math:`-\tfrac12\nabla^2` (matching :class:`PenetrableSphereHamiltonian`).

**Three dimensions only.** Every relation here is 3D-specific: the s-wave
scattering length Eq. (10) comes from the ``l=0`` radial equation in 3D, the gas
parameter carries ``a^3`` because ``[rho] = L^-3``, and the Lee-Yang expansion is
the 3D dilute-Bose result (the 1D analogue is Lieb-Liniger, 2D is Schick's
logarithmic form). The :class:`PenetrableSphereHamiltonian` itself is
dimension-agnostic; only this mapping-to-``x`` layer assumes ``n_dim == 3``.

The physics of a dilute gas is controlled **not** by ``(V0, R, rho)`` directly
but by the *gas parameter*

.. math::  x = \rho\, a^3,

where ``a`` is the s-wave scattering length. For the soft-sphere barrier
(Eq. 9: ``V = V0`` for ``r < R``, else 0) the scattering length is (Eq. 10)

.. math::  a = R\left[1 - \frac{\tanh(K_0 R)}{K_0 R}\right], \qquad K_0 = \sqrt{V_0}.

This module maps ``(V0, R)`` -> ``a`` and a target ``x`` -> box side ``L`` for
``N`` particles, plus the Lee-Yang low-density benchmark (Eq. 1) used to validate
the VMC energies at small ``x``.
"""

from __future__ import annotations

import math

# -- Gas constant --

def get_x(V0: float, R: float, L: float, N: int) -> float:
    """Gas parameter ``x = rho a^3`` for soft-sphere barrier (V0, R) at density rho = N/L^3."""
    a = soft_sphere_scattering_length(V0, R)
    rho = N / L**3
    return rho * a**3

# ── scattering length ─────────────────────────────────────────────────────────


def soft_sphere_scattering_length(V0: float, R: float = 1.0) -> float:
    r"""s-wave scattering length of the soft (penetrable) sphere, Eq. (10).

    ``a = R[1 - tanh(K0 R)/(K0 R)]`` with ``K0 = sqrt(V0)`` (hbar = m = 1).

    Limits (both handled without catastrophic cancellation):
        * ``V0 -> inf`` (``K0 R -> inf``): ``a -> R``  — recovers the hard sphere.
        * ``V0 -> 0``   (``K0 R -> 0``):   ``a -> 0``  — non-interacting.
    """
    if V0 < 0:
        raise ValueError("soft-sphere barrier requires V0 >= 0 (repulsive)")
    if V0 == 0.0:
        return 0.0
    k0r = math.sqrt(V0) * R
    if k0r < 1e-3:
        # tanh(z)/z = 1 - z^2/3 + 2 z^4/15 - ...  ->  a = R(z^2/3 - 2 z^4/15 + ...)
        a_over_R = k0r**2 / 3.0 - 2.0 * k0r**4 / 15.0
    else:
        a_over_R = 1.0 - math.tanh(k0r) / k0r
    return R * a_over_R


def soft_sphere_V0_for_scattering_length(a: float, R: float = 1.0,
                                         tol: float = 1e-12) -> float:
    r"""Invert Eq. (10): barrier height ``V0`` giving scattering length ``a`` at range ``R``.

    Monotone in ``V0`` (``a`` grows from 0 toward ``R`` as ``V0`` grows), so a
    bisection on ``K0 R`` is robust. Requires ``0 < a < R``.
    """
    if not (0.0 < a < R):
        raise ValueError(f"need 0 < a < R; got a={a}, R={R} (a=R is the hard-sphere limit)")
    target = a / R  # = 1 - tanh(z)/z, monotone increasing in z = K0 R

    def f(z: float) -> float:
        return (1.0 - math.tanh(z) / z) - target

    lo, hi = 1e-6, 1.0
    while f(hi) < 0.0:        # grow upper bracket until it straddles the root
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    z = 0.5 * (lo + hi)
    return (z / R) ** 2  # V0 = K0^2 = (z/R)^2


# ── density / box geometry ─────────────────────────────────────────────────────


def density_from_gas_parameter(x: float, a: float) -> float:
    """Number density ``rho = x / a^3`` for gas parameter ``x = rho a^3``."""
    return x / a**3


def box_side_for_gas_parameter(x: float, a: float, N: int) -> float:
    """Cubic-box side ``L`` holding ``N`` particles at gas parameter ``x``.

    ``rho = N / L^3 = x / a^3``  =>  ``L = (N a^3 / x)^{1/3}``.
    """
    rho = density_from_gas_parameter(x, a)
    return (N / rho) ** (1.0 / 3.0)


# ── low-density benchmark ──────────────────────────────────────────────────────


def lee_yang_energy_per_particle(x: float, a: float = 1.0) -> float:
    r"""Lee-Yang expansion of ``E/N`` (Eq. 1), in units of ``hbar^2/(2 m a^2) = 1/a^2``.

    .. math::  E/N = \frac{\hbar^2}{2 m a^2}\,4\pi x\left[1 + \frac{128}{15}\sqrt{x/\pi}\right].

    Universal (depends only on ``x`` and ``a``), so any short-range potential with
    the same ``a`` must converge to this as ``x -> 0`` — the validation target.
    """
    return (1.0 / a**2) * 4.0 * math.pi * x * (1.0 + (128.0 / 15.0) * math.sqrt(x / math.pi))
