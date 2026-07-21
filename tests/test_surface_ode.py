"""Unit tests for the surface deposition/aggregation ODE system, Eqs. (C.13)-(C.20).

Depends on `mechanistic/surface_ode.py`, which is currently a scaffolding
stub -- these tests are placeholders documenting the intended coverage and
are marked `xfail` until that module is implemented:

* `saturation_term`: S=1 at M=0, S=0 at M=M_inf, monotonically decreasing.
* Simplified ODE system (Eqs. C.17-C.20, theta=1): M_r stays at 0 (all
  resting-platelet adhesion converts immediately to activated surface
  coverage under theta=1); dM/dt and dM_as/dt reduce to the same expression
  (Eq. C.19); dM_at/dt >= dM_as/dt always (M_at accumulates aggregation on
  top of M_as, Eq. C.20).
* Mechanical flux terms (Eqs. 6-7) are exactly zero wherever the local
  shear-rate gradient is non-negative (the model's defining zero-order
  condition for shear-gradient-induced aggregation).
"""

from __future__ import annotations

import numpy as np
import pytest

from thrombus_bench.mechanistic.surface_ode import SurfaceState, saturation_term


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


@pytest.mark.xfail(reason="surface_ode.py mechanical/chemical flux + ODE RHS not yet implemented", strict=False)
def test_mechanical_flux_zero_for_nonnegative_gradient():
    from thrombus_bench.mechanistic.surface_ode import mechanical_flux_resting

    wall_shear_rate = np.array([1000.0, 1000.0])
    d_shear_rate_dx = np.array([1.0, 0.0])  # both non-negative
    saturation = np.array([1.0, 1.0])
    resting_platelet_conc = np.array([3.5e8, 3.5e8])
    flux = mechanical_flux_resting(
        wall_shear_rate, d_shear_rate_dx, saturation, resting_platelet_conc, k_rs_cm_s=0.0037, L_cm=1.0
    )
    assert np.allclose(flux, 0.0)


@pytest.mark.xfail(reason="surface_ode.py ODE RHS not yet implemented", strict=False)
def test_surface_ode_rhs_simplified_theta_1_keeps_M_r_zero():
    from thrombus_bench.mechanistic.surface_ode import surface_ode_rhs

    wall_dofs = np.arange(5)
    state = SurfaceState.zeros_like(wall_dofs)
    d_state = surface_ode_rhs(state, fluxes={}, rates={}, D_a=1.0)
    assert np.allclose(d_state.M_r, 0.0)
