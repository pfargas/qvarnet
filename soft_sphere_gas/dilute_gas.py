r"""Dilute Bose-gas parametrization for the hard-/soft-sphere study.

Reproduces the control-parameter layer of Mazzanti, Polls & Fabrocini,
*"Energy and structure of dilute hard- and soft-sphere Bose gases"*
(arXiv:cond-mat/0305502).

Units and conventions
---------------------
This module speaks the **paper's convention** so every number drops directly onto
Mazzanti's tables and figures:

* :math:`\hbar^2 / 2m = 1`  (the kinetic operator is :math:`-\nabla^2`),
* lengths in units of the scattering length ``a``  (i.e. ``a = 1``),
* energies in units of :math:`\hbar^2 / 2m a^2`.

In this convention Eq. (10) reads :math:`K_0^2 = V_0\,m/\hbar^2 = V_0/2` (constant
``HBAR2_OVER_M`` = :math:`\hbar^2/m` = 2). The paper's published barriers
``V0(SS10)=0.00681670`` (R=10) and ``V0(SS5)=0.06308561`` (R=5) then both give
``a = 1`` -- see ``test_paper_published_potentials``.

Bridge to the qvarnet engine
-----------------------------
:class:`PenetrableSphereHamiltonian` (and the rest of the engine) works in
:math:`\hbar = m = 1`, i.e. its kinetic operator is :math:`-\tfrac12\nabla^2` --
a factor of two from the paper's :math:`-\nabla^2`. The translation lives in
exactly two helpers and nowhere else:

* build the Hamiltonian with ``V0_engine = engine_V0(V0_paper)``  (= V0_paper / 2),
* report ``E_paper = to_paper_energy(E_engine)``                  (= 2 * E_engine).

With both applied, ``E_paper = <-grad^2> + <V0_paper * theta(R - r)>`` is exactly
the paper's Hamiltonian expectation. Lengths (``R``, box ``L``) are identical in
both conventions; only ``V0`` and the energy carry the factor of two.

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

.. math::  a = R\left[1 - \frac{\tanh(K_0 R)}{K_0 R}\right], \qquad K_0 = \sqrt{V_0/2}.

This module maps ``(V0, R)`` -> ``a`` and a target ``x`` -> box side ``L`` for
``N`` particles, plus the Lee-Yang low-density benchmark (Eq. 1) used to validate
the VMC energies at small ``x``.
"""

from __future__ import annotations

import math

# Paper convention: hbar^2 / 2m = 1  =>  hbar^2 / m = 2. Eq. (10) needs m/hbar^2,
# so K0^2 = V0 * m/hbar^2 = V0 / HBAR2_OVER_M. This same constant carries the
# factor of two of the engine bridge (engine_V0 / to_paper_energy below).
HBAR2_OVER_M = 2.0


# -- bridge to the qvarnet engine (hbar = m = 1, kinetic -1/2 grad^2) --


def engine_V0(V0_paper: float) -> float:
    """Barrier height to hand to ``PenetrableSphereHamiltonian`` (hbar = m = 1).

    The engine's kinetic operator is ``-1/2 grad^2`` whereas the paper uses
    ``-grad^2``; matching the same physics halves the barrier. Pair with
    :func:`to_paper_energy` on the energy coming back out.
    """
    return V0_paper / HBAR2_OVER_M


def to_paper_energy(E_engine: float) -> float:
    """Convert an engine energy (hbar = m = 1) to paper units (hbar^2/2m = 1)."""
    return HBAR2_OVER_M * E_engine


# -- Gas parameter --

def get_x(V0: float, R: float, L: float, N: int) -> float:
    """Gas parameter ``x = rho a^3`` for soft-sphere barrier (V0, R) at density rho = N/L^3.

    ``V0`` is in paper units (hbar^2/2m = 1); see module docstring.
    """
    a = soft_sphere_scattering_length(V0, R)
    rho = N / L**3
    return rho * a**3

# ── scattering length ─────────────────────────────────────────────────────────


def soft_sphere_scattering_length(V0: float, R: float = 1.0) -> float:
    r"""s-wave scattering length of the soft (penetrable) sphere, Eq. (10).

    ``a = R[1 - tanh(K0 R)/(K0 R)]`` with ``K0 = sqrt(V0/2)`` -- paper units
    (hbar^2/2m = 1, so K0^2 = V0 m/hbar^2 = V0/2). ``V0`` is the barrier height in
    units of hbar^2/2ma^2.

    Limits (both handled without catastrophic cancellation):
        * ``V0 -> inf`` (``K0 R -> inf``): ``a -> R``  — recovers the hard sphere.
        * ``V0 -> 0``   (``K0 R -> 0``):   ``a -> 0``  — non-interacting.
    """
    if V0 < 0:
        raise ValueError("soft-sphere barrier requires V0 >= 0 (repulsive)")
    if V0 == 0.0:
        return 0.0
    k0r = math.sqrt(V0 / HBAR2_OVER_M) * R
    if k0r < 1e-3:
        # tanh(z)/z = 1 - z^2/3 + 2 z^4/15 - ...  ->  a = R(z^2/3 - 2 z^4/15 + ...)
        a_over_R = k0r**2 / 3.0 - 2.0 * k0r**4 / 15.0
    else:
        a_over_R = 1.0 - math.tanh(k0r) / k0r
    return R * a_over_R


def soft_sphere_V0_for_scattering_length(a: float, R: float = 1.0,
                                         tol: float = 1e-12) -> float:
    r"""Invert Eq. (10): barrier height ``V0`` giving scattering length ``a`` at range ``R``.

    Returns ``V0`` in paper units (hbar^2/2m = 1). Monotone in ``V0`` (``a`` grows
    from 0 toward ``R`` as ``V0`` grows), so a bisection on ``K0 R`` is robust.
    Requires ``0 < a < R``.
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
    return HBAR2_OVER_M * (z / R) ** 2  # V0 = K0^2 * hbar^2/m = HBAR2_OVER_M * (z/R)^2


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
    r"""Lee-Yang expansion of ``E/N`` (Eq. 1), paper units (hbar^2/2m = 1).

    .. math::  E/N = \frac{\hbar^2}{2 m a^2}\,4\pi x\left[1 + \frac{128}{15}\sqrt{x/\pi}\right].

    With ``a = 1`` (lengths in units of a) this returns the scaled ``E/N`` plotted in
    the paper (Fig. 1, Tables I/III); the ``1/a^2`` prefactor only matters if you want
    absolute energy at ``a != 1``. Universal (depends only on ``x`` and ``a``), so any
    short-range potential with the same ``a`` must converge to this as ``x -> 0`` -- the
    low-density validation target. The leading term ``4 pi x`` alone is a rigorous lower
    bound to the exact energy (Lieb-Yngvason).
    """
    return (1.0 / a**2) * 4.0 * math.pi * x * (1.0 + (128.0 / 15.0) * math.sqrt(x / math.pi))


def first_order_energy_upper_bound(x: float, V0: float, R: float) -> float:
    r"""First-order perturbation energy per particle, Eq. (31): a variational upper bound.

    .. math::  E_1/N = \tfrac12 \rho V_0 \tfrac{4\pi}{3} R^3 = \tfrac12 \tilde V(0),

    the mean potential energy of the *uncorrelated* (constant) wavefunction. In paper
    units with ``a = 1`` the density is ``rho = x``, so ``E1/N = 1/2 x V0 (4pi/3) R^3``
    (``V0``, ``R``, ``x`` all in paper units).

    Cheap zero-training sanity check: an untrained VMC run must give a potential energy
    per particle equal to this, and the converged total energy must lie below it.
    """
    return 0.5 * x * V0 * (4.0 / 3.0) * math.pi * R**3
