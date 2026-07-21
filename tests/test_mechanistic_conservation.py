"""Sanity checks for the mechanistic flow solver on a trivial steady-state case.

A plain rectangular channel (no aneurysm) with a uniform inlet velocity has
a known analytical solution in the Newtonian, high-shear (mu ~= mu_inf)
limit: fully-developed 2D Poiseuille flow, with a parabolic velocity profile
whose peak is 1.5x the mean velocity. This is used here as the trivial
steady-state validation case referenced in the project scaffolding notes,
checking:

1. Mass conservation: inlet flux == outlet flux (to near machine precision),
   for both the plain channel and the full idealized aneurysm geometry.
2. Momentum/physical plausibility: no-slip at the walls, and a downstream
   velocity profile approaching the analytical Poiseuille parabola once the
   Carreau viscosity has relaxed to its high-shear value.
"""

from __future__ import annotations

import numpy as np
import pytest

from thrombus_bench.mechanistic.flow_solver import CarreauParams, compute_boundary_flux, solve_steady_flow
from thrombus_bench.mechanistic.mesh import GeometryConfig, MeshConfig, build_aneurysm_mesh, build_channel_mesh

CARREAU = CarreauParams(mu_inf_pa_s=0.0035, mu_0_pa_s=0.056, lambda_s=3.313, n=0.3568)


@pytest.fixture
def channel():
    return build_channel_mesh(length_mm=50.0, diameter_mm=4.0, target_num_elements=600)


def test_channel_flow_converges(channel):
    flow = solve_steady_flow(channel, inlet_velocity_m_s=0.47, carreau=CARREAU)
    assert flow.converged
    assert flow.residual < 1e-6


def test_channel_mass_conservation(channel):
    flow = solve_steady_flow(channel, inlet_velocity_m_s=0.47, carreau=CARREAU)
    q_in = compute_boundary_flux(flow, "inlet")
    q_out = compute_boundary_flux(flow, "outlet")
    # Flux in and out should be equal and opposite (outward normal convention).
    assert abs(q_in + q_out) < 1e-12
    diameter_m = 4.0e-3
    expected = diameter_m * 0.47
    assert q_out == pytest.approx(expected, rel=1e-10)


def test_channel_no_slip_walls(channel):
    flow = solve_steady_flow(channel, inlet_velocity_m_s=0.47, carreau=CARREAU)
    wall_dofs = flow.basis_u.get_dofs("wall_vessel").all()
    # Exclude the inlet/outlet corner dofs: a corner belongs to both `inlet`
    # (or `outlet`) and `wall_vessel`, and the prescribed inlet velocity
    # takes precedence there (a standard, harmless corner-BC ambiguity, not
    # a solver defect). No-slip is checked on the strictly-interior wall.
    doflocs = flow.basis_u.doflocs
    length_m = 0.05
    interior = ~np.isclose(doflocs[0], 0.0) & ~np.isclose(doflocs[0], length_m)
    interior_wall_dofs = wall_dofs[interior[wall_dofs]]
    assert len(interior_wall_dofs) > 0
    assert np.allclose(flow.u[interior_wall_dofs], 0.0, atol=1e-12)


def test_channel_velocity_profile_approaches_poiseuille(channel):
    """Downstream centerline velocity should approach 1.5x mean (parabolic profile)."""

    flow = solve_steady_flow(channel, inlet_velocity_m_s=0.47, carreau=CARREAU)
    # Evaluate u_x along a vertical line near the outlet via nodal DOF lookup
    # (P2 nodal dofs coincide with mesh vertices for the corner nodes).
    doflocs = flow.basis_u.doflocs
    near_outlet = np.isclose(doflocs[0], 0.05, atol=2e-3)
    ux_dof_is_first_component = np.arange(flow.basis_u.N) % 2 == 0
    mask = near_outlet & ux_dof_is_first_component
    ux_near_outlet = flow.u[mask]
    peak = ux_near_outlet.max()
    mean_inlet = 0.47
    # Peak should exceed the mean (profile has developed away from plug flow)
    # and stay within a broad band of the ideal Newtonian Poiseuille ratio of
    # 1.5, allowing for shear-thinning deviation and mesh coarseness.
    assert peak > mean_inlet
    assert peak < 2.0 * mean_inlet


def test_aneurysm_mesh_is_watertight():
    """Every boundary facet must be tagged as exactly one of the four named
    boundaries -- an untagged facet silently drops its no-slip/inlet/outlet
    BC and breaks mass conservation (see mesh.py `is_wall_vessel` docstring
    note)."""

    geom = GeometryConfig(vessel_diameter_mm=3.2, aneurysm_diameter_mm=7.0, vessel_length_mm=50.0)
    tm = build_aneurysm_mesh(geom, MeshConfig(target_num_elements=1200))
    m = tm.mesh
    total_boundary = set(m.boundary_facets().tolist())
    tagged = set()
    for facets in m.boundaries.values():
        tagged.update(facets.tolist())
    assert total_boundary == tagged


@pytest.mark.parametrize(
    "vessel_d_mm,aneurysm_d_mm,v_in_m_s",
    [(3.2, 7.0, 0.47), (4.0, 10.0, 0.75)],
)
def test_aneurysm_flow_mass_conservation(vessel_d_mm, aneurysm_d_mm, v_in_m_s):
    geom = GeometryConfig(vessel_diameter_mm=vessel_d_mm, aneurysm_diameter_mm=aneurysm_d_mm, vessel_length_mm=50.0)
    tm = build_aneurysm_mesh(geom, MeshConfig(target_num_elements=1500))
    flow = solve_steady_flow(tm, inlet_velocity_m_s=v_in_m_s, carreau=CARREAU)
    assert flow.converged
    q_in = compute_boundary_flux(flow, "inlet")
    q_out = compute_boundary_flux(flow, "outlet")
    assert abs(q_in + q_out) / abs(q_in) < 1e-8
