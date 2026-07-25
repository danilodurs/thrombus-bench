"""Smoke tests for `viz/showcase_plots.py`: every function should run to
completion without error on a small fixture and return a plottable
artist/mappable. These are presentation-only functions (no new numerical
logic beyond what `flow_solver.py`/`field_interpolation.py` already
unit-test), so a smoke test is the appropriate coverage level per the
Stage 1 testing strategy.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from thrombus_bench.mechanistic.flow_solver import CarreauParams, solve_steady_flow, wall_traction
from thrombus_bench.mechanistic.mesh import GeometryConfig, MeshConfig, build_aneurysm_mesh
from thrombus_bench.viz import showcase_plots
from thrombus_bench.viz.field_interpolation import interpolate_velocity_to_grid

CARREAU = CarreauParams(mu_inf_pa_s=0.0035, mu_0_pa_s=0.056, lambda_s=3.313, n=0.3568)


@pytest.fixture(scope="module")
def aneurysm_fixture():
    geom = GeometryConfig(vessel_diameter_mm=3.2, aneurysm_diameter_mm=7.0, vessel_length_mm=50.0)
    tm = build_aneurysm_mesh(geom, MeshConfig(target_num_elements=600))
    flow = solve_steady_flow(tm, inlet_velocity_m_s=0.47, carreau=CARREAU)
    n_vertices = tm.mesh.p.shape[1]
    ux = flow.u[0 : 2 * n_vertices : 2]
    uy = flow.u[1 : 2 * n_vertices : 2]
    grid = interpolate_velocity_to_grid(tm.mesh.p, ux, uy, tm.mesh.t, grid_size=(20, 60))
    traction = wall_traction(flow, CARREAU, "wall_vessel")
    return {"geom": geom, "tm": tm, "flow": flow, "ux": ux, "uy": uy, "grid": grid, "traction": traction}


def test_plot_velocity_quiver(aneurysm_fixture):
    fig, ax = plt.subplots()
    result = showcase_plots.plot_velocity_quiver(aneurysm_fixture["tm"].mesh.p, aneurysm_fixture["ux"], aneurysm_fixture["uy"], ax=ax)
    assert result is not None
    assert ax.get_aspect() in ("equal", 1.0)
    plt.close(fig)


def test_plot_wall_traction_map(aneurysm_fixture):
    fig, ax = plt.subplots()
    result = showcase_plots.plot_wall_traction_map(aneurysm_fixture["traction"], ax=ax)
    assert result is not None
    plt.close(fig)


def test_plot_streamlines(aneurysm_fixture):
    fig, ax = plt.subplots()
    showcase_plots.plot_streamlines(aneurysm_fixture["grid"], ax=ax)
    plt.close(fig)


def test_plot_vorticity_field(aneurysm_fixture):
    fig, ax = plt.subplots()
    result = showcase_plots.plot_vorticity_field(aneurysm_fixture["grid"], ax=ax)
    assert result is not None
    plt.close(fig)


def test_plot_pressure_field(aneurysm_fixture):
    fig, ax = plt.subplots()
    flow = aneurysm_fixture["flow"]
    result = showcase_plots.plot_pressure_field(aneurysm_fixture["tm"].mesh.p, aneurysm_fixture["tm"].mesh.t, flow.p, ax=ax)
    assert result is not None
    plt.close(fig)


def test_plot_bulk_field(aneurysm_fixture):
    fig, ax = plt.subplots()
    n_vertices = aneurysm_fixture["tm"].mesh.p.shape[1]
    fake_fibrin = np.random.default_rng(0).uniform(0, 1, size=n_vertices)
    result = showcase_plots.plot_bulk_field(
        aneurysm_fixture["tm"].mesh.p, aneurysm_fixture["tm"].mesh.t, fake_fibrin, ax=ax, title="Fibrin [uM] (bulk)"
    )
    assert result is not None
    plt.close(fig)


def test_plot_wall_band_M_at(aneurysm_fixture):
    fig, ax = plt.subplots()
    wall_dofs = aneurysm_fixture["flow"].basis_u.get_dofs("wall_vessel").nodal["u^1"]
    wall_coords = aneurysm_fixture["flow"].basis_u.doflocs[:, wall_dofs]
    fake_m_at = np.random.default_rng(0).uniform(0, 2e7, size=wall_coords.shape[1])
    result = showcase_plots.plot_wall_band_M_at(wall_coords, fake_m_at, ax=ax)
    assert result is not None
    plt.close(fig)


def test_plot_mesh_quality_with_inset(aneurysm_fixture):
    fig, ax = plt.subplots()
    result = showcase_plots.plot_mesh_quality(
        aneurysm_fixture["tm"].mesh.p, aneurysm_fixture["tm"].mesh.t, ax=ax, geom=aneurysm_fixture["geom"], inset=True
    )
    assert result is not None
    plt.close(fig)


def test_plot_mesh_quality_without_geom(aneurysm_fixture):
    """`geom=None` should skip the inset without erroring."""

    fig, ax = plt.subplots()
    showcase_plots.plot_mesh_quality(aneurysm_fixture["tm"].mesh.p, aneurysm_fixture["tm"].mesh.t, ax=ax, geom=None)
    plt.close(fig)


def test_compute_shared_vlim_basic():
    a = np.array([1.0, 2.0, np.nan])
    b = np.array([-3.0, 5.0])
    assert showcase_plots.compute_shared_vlim(a, b) == (-3.0, 5.0)


def test_compute_shared_vlim_symmetric():
    a = np.array([-2.0, 4.0])
    vmin, vmax = showcase_plots.compute_shared_vlim(a, symmetric=True)
    assert vmin == -4.0
    assert vmax == 4.0


def test_showcase_plots_accept_pointcloud_shape_convention(aneurysm_fixture):
    """node_coords/triangles in the (n_nodes, 2)/(n_triangles, 3) point-cloud
    convention should work identically to the raw mesh.p/mesh.t convention
    (see field_interpolation.py's shape-canonicalization docstring)."""

    fig, ax = plt.subplots()
    tm, flow = aneurysm_fixture["tm"], aneurysm_fixture["flow"]
    node_coords_t = tm.mesh.p.T
    triangles_t = tm.mesh.t.T
    showcase_plots.plot_pressure_field(node_coords_t, triangles_t, flow.p, ax=ax)
    plt.close(fig)
