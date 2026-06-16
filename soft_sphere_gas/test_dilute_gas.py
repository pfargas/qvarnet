"""Parametrization layer for the hard-/soft-sphere dilute Bose gas (Mazzanti 2003)."""

import math

import pytest
from dilute_gas import (
    HBAR2_OVER_M,
    box_side_for_gas_parameter,
    density_from_gas_parameter,
    engine_V0,
    first_order_energy_upper_bound,
    lee_yang_energy_per_particle,
    soft_sphere_scattering_length,
    soft_sphere_V0_for_scattering_length,
    to_paper_energy,
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
    # evaluated just above the switch point. K0 R = sqrt(V0/HBAR2_OVER_M)*R.
    R = 1.0
    k0r = 1.1e-3  # just above the 1e-3 series/closed-form switch
    V0 = HBAR2_OVER_M * (k0r / R) ** 2
    closed = R * (1.0 - math.tanh(k0r) / k0r)
    assert soft_sphere_scattering_length(V0, R) == pytest.approx(closed, rel=1e-6)


def test_monotonic_in_V0():
    vals = [soft_sphere_scattering_length(V0, R=1.0) for V0 in (0.1, 1.0, 10.0, 100.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_scales_with_R():
    # a is homogeneous of degree 1 in R at fixed K0 R, i.e. fixed V0 R^2.
    a1 = soft_sphere_scattering_length(V0=4.0, R=1.0)
    a2 = soft_sphere_scattering_length(V0=1.0, R=2.0)  # same K0 R (= sqrt(2))
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


def test_lee_yang_matches_paper_values():
    # Paper text (Sec. V): E_LD/N = 0.001317 (x=1e-4), 0.1862 (x=1e-2), a = 1.
    assert lee_yang_energy_per_particle(1e-4) == pytest.approx(0.001317, abs=1e-6)
    assert lee_yang_energy_per_particle(1e-2) == pytest.approx(0.1862, abs=1e-4)


# ── paper convention: published potentials reproduce a = 1 ─────────────────────


@pytest.mark.parametrize(
    "R, V0",
    [
        (10.0, 0.00681670),  # SS10
        (5.0, 0.06308561),   # SS5
    ],
)
def test_paper_published_potentials(R, V0):
    # The whole point of the convention: the paper's tabulated (R, V0) give a = 1.
    assert soft_sphere_scattering_length(V0, R) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize(
    "R, x, expected",
    [
        (10.0, 1e-4, 0.0014277),  # SS10
        (10.0, 1e-2, 0.14277),
        (5.0, 1e-4, 0.0016516),   # SS5
        (5.0, 1e-2, 0.16516),
    ],
)
def test_first_order_upper_bound_matches_paper(R, x, expected):
    # Eq. (31) numbers reported in Sec. V, with V0 set so that a = 1 at this R.
    V0 = soft_sphere_V0_for_scattering_length(1.0, R)
    assert first_order_energy_upper_bound(x, V0, R) == pytest.approx(expected, rel=2e-3)


# ── engine bridge (paper hbar^2/2m=1  <->  engine hbar=m=1) ─────────────────────


def test_engine_bridge_is_a_factor_of_two():
    assert engine_V0(0.00681670) == pytest.approx(0.00681670 / 2.0)
    assert to_paper_energy(0.5) == pytest.approx(1.0)
    # round trip through the same constant
    assert to_paper_energy(engine_V0(HBAR2_OVER_M)) == pytest.approx(HBAR2_OVER_M)
