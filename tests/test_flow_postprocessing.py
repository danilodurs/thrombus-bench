"""Unit tests for `flow_solver.py`'s new visualization/postprocessing
quantities (Stage 1 of the visualization-quality work): `vorticity`/
`FlowSolution.vorticity_at_quadrature` and `wall_traction`.

Uses the same plain rectangular channel + fully-developed-Poiseuille
reasoning as `test_mechanistic_conservation.py`: away from the entrance,
`u_y ~= 0` and `u_x` depends only on `y`, so both new quantities have a
known qualitative (antisymmetric-about-the-centerline) sign structure that
does not depend on the exact parabolic shape -- a robust check even on a
coarse mesh.
"""

from __future__ import annotations

import numpy as np
import pytest

from thrombus_bench.mechanistic.flow_solver import CarreauParams, solve_steady_flow, wall_traction
from thrombus_bench.mechanistic.mesh import build_channel_mesh

CARREAU = CarreauParams(mu_inf_pa_s=0.0035, mu_0_pa_s=0.056, lambda_s=3.313, n=0.3568)


@pytest.fixture
def channel_flow():
    channel = build_channel_mesh(length_mm=50.0, diameter_mm=4.0, target_num_elements=600)
    return solve_steady_flow(channel, inlet_velocity_m_s=0.47, carreau=CARREAU)


def test_vorticity_zero_for_uniform_plug_flow():
    """A pure uniform (plug) velocity field -- no shear anywhere -- has zero
    vorticity everywhere, independent of any solve (a synthetic DOF vector
    exercises `vorticity`/`curl` directly without depending on how well a
    coarse mesh reproduces a physical channel flow)."""

    channel = build_channel_mesh(length_mm=50.0, diameter_mm=4.0, target_num_elements=200)
    flow = solve_steady_flow(channel, inlet_velocity_m_s=0.47, carreau=CARREAU)
    u_uniform = np.zeros_like(flow.u)
    u_uniform[0::2] = 0.47  # ux = const, uy = 0 everywhere
    vort = flow.basis_u.interpolate(u_uniform)
    from thrombus_bench.mechanistic.flow_solver import vorticity

    omega = vorticity(vort)
    assert np.allclose(omega, 0.0, atol=1e-9)


def test_vorticity_antisymmetric_about_centerline(channel_flow):
    """Fully-developed channel flow's vorticity omega = -du_x/dy should be
    positive above the centerline (velocity decreasing with y, so
    du_x/dy < 0) and negative below it (by symmetry), with comparable
    magnitude on each side -- checked near the outlet, where the profile
    has had the whole channel length to develop."""

    omega = channel_flow.vorticity_at_quadrature()
    qp_x = np.asarray(channel_flow.basis_u.global_coordinates()[0])
    qp_y = np.asarray(channel_flow.basis_u.global_coordinates()[1])

    diameter_m = 4.0e-3
    near_outlet = np.abs(qp_x - 0.9 * 0.05) < 0.003
    top = near_outlet & (qp_y > 0.6 * diameter_m)
    bottom = near_outlet & (qp_y < 0.4 * diameter_m)
    assert top.any() and bottom.any()

    omega_top = omega[top].mean()
    omega_bottom = omega[bottom].mean()
    assert omega_top > 0.0
    assert omega_bottom < 0.0
    # Antisymmetric about the centerline: comparable magnitude, opposite sign.
    assert omega_top == pytest.approx(-omega_bottom, rel=0.3)


def test_wall_traction_shape_and_finite(channel_flow):
    result = wall_traction(channel_flow, CARREAU, "wall_vessel")
    n_qp = result["points"].shape[0]
    assert result["traction"].shape == (n_qp, 2)
    assert result["magnitude"].shape == (n_qp,)
    assert np.all(np.isfinite(result["points"]))
    assert np.all(np.isfinite(result["traction"]))
    assert result["magnitude"].min() >= 0.0


def test_wall_traction_tangential_component_opposes_flow(channel_flow):
    """Near the outlet (developed flow), the wall-tangential (x) traction
    component should be negative (drag opposing the +x flow direction) at
    both the top and bottom walls -- `t_x = mu * du_x/dy * n_y`, and
    `du_x/dy` and `n_y` have opposite signs at each wall by construction, so
    the product is negative at both regardless of which wall."""

    result = wall_traction(channel_flow, CARREAU, "wall_vessel")
    points, traction = result["points"], result["traction"]
    diameter_m = 4.0e-3

    near_outlet = np.abs(points[:, 0] - 0.9 * 0.05) < 0.003
    bottom_wall = near_outlet & (points[:, 1] < 0.1 * diameter_m)
    top_wall = near_outlet & (points[:, 1] > 0.9 * diameter_m)
    assert bottom_wall.any() and top_wall.any()

    assert traction[bottom_wall, 0].mean() < 0.0
    assert traction[top_wall, 0].mean() < 0.0


def test_wall_traction_normal_component_antisymmetric(channel_flow):
    """The wall-normal (y) traction component (pressure-dominated) should be
    antisymmetric between the bottom (`n = (0, -1)`) and top (`n = (0, 1)`)
    walls at the same axial position, since `t_y ~= -p * n_y` and pressure
    itself is (to leading order) uniform across the channel height at fixed
    x for this thin, low-Reynolds-number channel."""

    result = wall_traction(channel_flow, CARREAU, "wall_vessel")
    points, traction = result["points"], result["traction"]
    diameter_m = 4.0e-3

    near_outlet = np.abs(points[:, 0] - 0.9 * 0.05) < 0.003
    bottom_wall = near_outlet & (points[:, 1] < 0.1 * diameter_m)
    top_wall = near_outlet & (points[:, 1] > 0.9 * diameter_m)
    assert bottom_wall.any() and top_wall.any()

    ty_bottom = traction[bottom_wall, 1].mean()
    ty_top = traction[top_wall, 1].mean()
    assert ty_bottom * ty_top < 0.0
    assert ty_bottom == pytest.approx(-ty_top, rel=0.3)


def test_wall_traction_thrombus_multiplier_increases_magnitude(channel_flow):
    """Passing `thrombus_fields` with M_at above its critical threshold
    everywhere should scale up the viscosity (Eq. 18) and hence the
    traction magnitude relative to the no-thrombus case, at the same
    converged velocity/pressure field."""

    from thrombus_bench.mechanistic.flow_solver import ThrombusViscosityFields

    n = channel_flow.basis_p.N
    thrombus_fields = ThrombusViscosityFields(
        M_at_nodal=np.full(n, 1.0e8),
        FI_nodal=np.zeros(n),
        M_at_critical_plt_cm2=2.0e7,
        fibrin_critical_uM=0.6,
        steepness_theta=20.0,
        multiplier_max=80.0,
    )
    baseline = wall_traction(channel_flow, CARREAU, "wall_vessel")
    boosted = wall_traction(channel_flow, CARREAU, "wall_vessel", thrombus_fields=thrombus_fields)
    assert boosted["magnitude"].mean() > baseline["magnitude"].mean()
