"""Unit tests for the surface deposition/aggregation ODE system, Eqs. (C.13)-(C.20)."""

from __future__ import annotations

import numpy as np
import pytest

from thrombus_bench.mechanistic.surface_ode import (
    SurfaceState,
    chemical_flux_activated,
    chemical_flux_resting,
    mechanical_flux_activated,
    mechanical_flux_resting,
    saturation_term,
    step_surface_state,
    surface_ode_rhs,
)


def test_surface_state_zero_initial_condition():
    wall_dofs = np.arange(10)
    state = SurfaceState.zeros_like(wall_dofs)
    assert np.allclose(state.M, 0.0)
    assert np.allclose(state.M_r, 0.0)
    assert np.allclose(state.M_as, 0.0)
    assert np.allclose(state.M_at, 0.0)


def test_saturation_term_bounds():
    M_inf = 7.0e6
    M = np.array([0.0, 0.5 * M_inf, M_inf])
    S = saturation_term(M, M_inf)
    assert S[0] == pytest.approx(1.0)
    assert S[1] == pytest.approx(0.5)
    assert S[2] == pytest.approx(0.0)


def test_mechanical_flux_zero_for_nonnegative_gradient():
    gamma_w_ref = 1000.0
    d_shear_rate_dx = np.array([1.0, 0.0])  # both non-negative
    saturation = np.array([1.0, 1.0])
    resting_platelet_conc = np.array([3.5e8, 3.5e8])
    flux = mechanical_flux_resting(
        gamma_w_ref, d_shear_rate_dx, saturation, resting_platelet_conc, k_rs_cm_s=0.0037, L_m=1.0
    )
    assert np.allclose(flux, 0.0)


def test_mechanical_flux_positive_for_negative_gradient():
    gamma_w_ref = 1000.0
    d_shear_rate_dx = np.array([-500.0])
    saturation = np.array([1.0])
    resting_platelet_conc = np.array([3.5e8])
    flux = mechanical_flux_resting(
        gamma_w_ref, d_shear_rate_dx, saturation, resting_platelet_conc, k_rs_cm_s=0.0037, L_m=1.0
    )
    assert flux[0] > 0.0


def test_mechanical_flux_activated_includes_aggregation_term():
    # With M/M_inf = 1 (full coverage), the k_aa aggregation term dominates
    # since the k_as adhesion term is scaled by saturation S=0.
    gamma_w_ref = 1000.0
    d_shear_rate_dx = np.array([-500.0])
    saturation = np.array([0.0])
    M = np.array([7.0e6])
    flux = mechanical_flux_activated(
        gamma_w_ref, d_shear_rate_dx, saturation, M, M_inf_plt_cm2=7.0e6,
        activated_platelet_conc=np.array([1.75e7]), k_as_cm_s=0.045, k_aa_cm_s=0.045, L_m=1.0,
    )
    assert flux[0] > 0.0


def test_chemical_flux_resting_scales_with_saturation():
    conc = np.array([3.5e8])
    flux_full = chemical_flux_resting(np.array([1.0]), conc, k_rs_cm_s=0.0037)
    flux_half = chemical_flux_resting(np.array([0.5]), conc, k_rs_cm_s=0.0037)
    assert flux_full[0] == pytest.approx(2.0 * flux_half[0])


def test_chemical_flux_activated_zero_saturation_and_zero_coverage_gives_zero():
    flux = chemical_flux_activated(
        np.array([0.0]), np.array([0.0]), M_inf_plt_cm2=7.0e6,
        activated_platelet_conc=np.array([1.75e7]), k_as_cm_s=0.045, k_aa_cm_s=0.045,
    )
    assert flux[0] == pytest.approx(0.0)


def test_surface_ode_rhs_theta1_keeps_M_r_zero():
    wall_dofs = np.arange(5)
    state = SurfaceState.zeros_like(wall_dofs)
    rhs = surface_ode_rhs(
        state,
        gamma_w_ref=1000.0,
        d_shear_rate_dx=np.full(5, -100.0),
        resting_platelet_conc=np.full(5, 3.5e8),
        activated_platelet_conc=np.full(5, 1.75e7),
        k_rs_cm_s=0.0037,
        k_as_cm_s=0.045,
        k_aa_cm_s=0.045,
        M_inf_plt_cm2=7.0e6,
        L_m=1.0,
        D_a=1.0,
    )
    assert np.allclose(rhs.M_r, 0.0)
    assert np.all(rhs.M > 0.0)
    assert np.all(rhs.M_at >= rhs.M_as)  # M_at includes the extra aggregation term


def test_surface_ode_growth_saturates_below_capacity():
    """Repeated stepping should keep M bounded near M_inf (saturation term
    drives growth to zero as M approaches M_inf), never exceeding it by more
    than a small explicit-Euler overshoot tolerance."""

    wall_dofs = np.arange(3)
    state = SurfaceState.zeros_like(wall_dofs)
    M_inf = 7.0e6
    dt = 0.05
    for _ in range(4000):
        rhs = surface_ode_rhs(
            state,
            gamma_w_ref=1000.0,
            d_shear_rate_dx=np.full(3, -100.0),
            resting_platelet_conc=np.full(3, 3.5e8),
            activated_platelet_conc=np.full(3, 1.75e7),
            k_rs_cm_s=0.0037,
            k_as_cm_s=0.045,
            k_aa_cm_s=0.045,
            M_inf_plt_cm2=M_inf,
            L_m=1.0,
            D_a=1.0,
        )
        state = step_surface_state(state, rhs, dt)
    assert np.all(state.M < 1.1 * M_inf)
    assert np.all(state.M > 0.5 * M_inf)  # should have grown substantially
