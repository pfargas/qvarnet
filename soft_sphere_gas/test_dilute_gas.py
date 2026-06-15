"""Parametrization layer for the hard-/soft-sphere dilute Bose gas (Mazzanti 2003)."""

import math

import pytest

from dilute_gas import (
    box_side_for_gas_parameter,
    density_from_gas_parameter,
    lee_yang_energy_per_particle,
    soft_sphere_scattering_length,
    soft_sphere_V0_for_scattering_length,
)


# ── scattering length, Eq. (10) ────────────────────────────────────────────────


def test_hard_sphere_limit():
    # V0 -> inf  =>  a -> R  (hard sphere)
    a = soft_sphere_scattering_length(V0=1e8, R=1.0)
    assert a == pytest.approx(1.0, abs=1e-3)
    assert a < 1.0  # always strictly below R for finite barrier


def test_noninteracting_limit():
    # V0 -> 0  =>  a -> 0
    assert soft_sphere_scattering_length(V0=0.0, R=1.0) == 0.0
    assert soft_sphere_scattering_length(V0=1e-8, R=1.0) == pytest.approx(0.0, abs=1e-8)


def test_small_V0_series_matches_closed_form():
    # In the small-K0R branch the truncated series must agree with tanh form
    # evaluated just above the switch point.
    V0, R = (1.1e-3) ** 2, 1.0  # K0 R = 1.1e-3, just above 1e-3 switch
    z = math.sqrt(V0) * R
    closed = R * (1.0 - math.tanh(z) / z)
    assert soft_sphere_scattering_length(V0, R) == pytest.approx(closed, rel=1e-6)


def test_monotonic_in_V0():
    vals = [soft_sphere_scattering_length(V0, R=1.0) for V0 in (0.1, 1.0, 10.0, 100.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_scales_with_R():
    # a is homogeneous of degree 1 in R at fixed K0 R, i.e. fixed V0 R^2.
    a1 = soft_sphere_scattering_length(V0=4.0, R=1.0)
    a2 = soft_sphere_scattering_length(V0=1.0, R=2.0)  # same K0 R = 2
    assert a2 == pytest.approx(2.0 * a1, rel=1e-12)


def test_negative_V0_rejected():
    with pytest.raises(ValueError):
        soft_sphere_scattering_length(V0=-1.0)


# ── inversion ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("a_target", [0.05, 0.2, 0.5, 0.8, 0.95])
def test_invert_scattering_length_roundtrip(a_target):
    V0 = soft_sphere_V0_for_scattering_length(a_target, R=1.0)
    assert soft_sphere_scattering_length(V0, R=1.0) == pytest.approx(a_target, rel=1e-9)


def test_invert_rejects_out_of_range():
    with pytest.raises(ValueError):
        soft_sphere_V0_for_scattering_length(1.0, R=1.0)  # a == R unreachable


# ── density / box geometry ─────────────────────────────────────────────────────


def test_box_reproduces_gas_parameter():
    a, N, x = 0.3, 16, 1e-3
    L = box_side_for_gas_parameter(x, a, N)
    rho = N / L**3
    assert rho * a**3 == pytest.approx(x, rel=1e-12)


def test_density_from_gas_parameter():
    assert density_from_gas_parameter(x=1e-3, a=0.5) == pytest.approx(1e-3 / 0.125)


# ── Lee-Yang benchmark, Eq. (1) ────────────────────────────────────────────────


def test_lee_yang_leading_order():
    # As x -> 0, E/N -> 4 pi x / a^2 (mean-field), correction term vanishes.
    x, a = 1e-6, 1.0
    e = lee_yang_energy_per_particle(x, a)
    assert e == pytest.approx(4 * math.pi * x, rel=2e-2)
    assert e > 4 * math.pi * x  # positive LHY correction


def test_lee_yang_units_scale_as_inv_a_squared():
    x = 1e-3
    assert lee_yang_energy_per_particle(x, a=2.0) == pytest.approx(
        lee_yang_energy_per_particle(x, a=1.0) / 4.0, rel=1e-12
    )
