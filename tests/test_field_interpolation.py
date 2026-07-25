"""Unit tests for `viz/field_interpolation.py` (Stage 1 visualization
postprocessing: grid-interpolated velocity + mesh-quality)."""

from __future__ import annotations

import numpy as np
import pytest

from thrombus_bench.mechanistic.flow_solver import CarreauParams, solve_steady_flow
from thrombus_bench.mechanistic.mesh import GeometryConfig, MeshConfig, build_aneurysm_mesh, build_channel_mesh
from thrombus_bench.viz.field_interpolation import (
    interpolate_velocity_to_grid,
    rasterize_reference_on_grid,
    triangle_min_angles_deg,
)

CARREAU = CarreauParams(mu_inf_pa_s=0.0035, mu_0_pa_s=0.056, lambda_s=3.313, n=0.3568)


@pytest.fixture
def aneurysm_flow():
    geom = GeometryConfig(vessel_diameter_mm=3.2, aneurysm_diameter_mm=7.0, vessel_length_mm=50.0)
    tm = build_aneurysm_mesh(geom, MeshConfig(target_num_elements=800))
    flow = solve_steady_flow(tm, inlet_velocity_m_s=0.47, carreau=CARREAU)
    return tm, flow


def test_interpolate_velocity_to_grid_shapes_and_masking(aneurysm_flow):
    tm, flow = aneurysm_flow
    node_coords = tm.mesh.p  # (2, n_nodes)
    n_vertices = node_coords.shape[1]
    ux = flow.u[0 : 2 * n_vertices : 2]
    uy = flow.u[1 : 2 * n_vertices : 2]

    grid = interpolate_velocity_to_grid(node_coords, ux, uy, tm.mesh.t, grid_size=(20, 50))
    assert grid["u"].shape == (20, 50)
    assert grid["v"].shape == (20, 50)
    assert grid["mask"].shape == (20, 50)
    assert grid["mask"].dtype == bool
    # Inside the fluid domain, values must be finite (no NaN leaking in from
    # the convex-hull fallback); outside, NaN by construction.
    assert np.all(np.isfinite(grid["u"][grid["mask"]]))
    assert np.all(np.isfinite(grid["v"][grid["mask"]]))
    assert np.all(np.isnan(grid["u"][~grid["mask"]]))
    assert np.all(np.isnan(grid["v"][~grid["mask"]]))
    # x/y must be strictly increasing (a valid regular grid for streamplot).
    assert np.all(np.diff(grid["x"]) > 0)
    assert np.all(np.diff(grid["y"]) > 0)


def test_interpolate_velocity_to_grid_accepts_pointcloud_shape_convention(aneurysm_flow):
    """The point-cloud `.npz` schema stores `node_coords`/`triangles` as
    `(n_nodes, 2)`/`(n_triangles, 3)` -- the transpose of `mesh.p`/`mesh.t`
    -- and this function must accept either without the caller transposing
    first (see module docstring)."""

    tm, flow = aneurysm_flow
    node_coords_t = tm.mesh.p.T  # (n_nodes, 2)
    triangles_t = tm.mesh.t.T  # (n_triangles, 3)
    n_vertices = node_coords_t.shape[0]
    ux = flow.u[0 : 2 * n_vertices : 2]
    uy = flow.u[1 : 2 * n_vertices : 2]

    grid_a = interpolate_velocity_to_grid(tm.mesh.p, ux, uy, tm.mesh.t, grid_size=(15, 30))
    grid_b = interpolate_velocity_to_grid(node_coords_t, ux, uy, triangles_t, grid_size=(15, 30))
    assert np.array_equal(grid_a["mask"], grid_b["mask"])
    np.testing.assert_allclose(np.nan_to_num(grid_a["u"]), np.nan_to_num(grid_b["u"]))


def test_triangle_min_angles_equilateral_reference():
    """A single equilateral triangle has all three interior angles == 60
    degrees, so its min angle should be exactly (up to float error) 60."""

    node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    triangles = np.array([[0, 1, 2]])
    min_angles = triangle_min_angles_deg(node_coords, triangles)
    assert min_angles.shape == (1,)
    assert min_angles[0] == pytest.approx(60.0, abs=1e-6)


def test_triangle_min_angles_right_triangle_reference():
    """A right isoceles triangle (legs along the axes) has angles
    45/45/90, so the min angle should be 45 degrees."""

    node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    triangles = np.array([[0, 1, 2]])
    min_angles = triangle_min_angles_deg(node_coords, triangles)
    assert min_angles[0] == pytest.approx(45.0, abs=1e-6)


def test_triangle_min_angles_on_real_mesh_in_valid_range(aneurysm_flow):
    """Every triangle's min angle must be strictly between 0 and 60 degrees
    (60 is the equilateral maximum a triangle's *minimum* angle can reach);
    a value outside that range would indicate a computation bug, not just
    mesh coarseness."""

    tm, _ = aneurysm_flow
    min_angles = triangle_min_angles_deg(tm.mesh.p, tm.mesh.t)
    assert min_angles.shape == (tm.mesh.t.shape[1],)
    assert np.all(min_angles > 0.0)
    assert np.all(min_angles <= 60.0 + 1e-6)


def test_interpolate_velocity_to_grid_channel_matches_uniform_inlet():
    """On a plain channel with a converged plug-inlet flow, the interpolated
    grid's velocity near the inlet should be close to the prescribed inlet
    velocity (a coarse end-to-end sanity check, not a precision test)."""

    channel = build_channel_mesh(length_mm=50.0, diameter_mm=4.0, target_num_elements=400)
    flow = solve_steady_flow(channel, inlet_velocity_m_s=0.47, carreau=CARREAU)
    n_vertices = channel.mesh.p.shape[1]
    ux = flow.u[0 : 2 * n_vertices : 2]
    uy = flow.u[1 : 2 * n_vertices : 2]

    grid = interpolate_velocity_to_grid(channel.mesh.p, ux, uy, channel.mesh.t, grid_size=(20, 100))
    near_inlet_col = 1  # skip the very first column (right at the Dirichlet BC edge)
    inlet_u = grid["u"][:, near_inlet_col]
    valid = ~np.isnan(inlet_u)
    assert valid.any()
    assert np.nanmean(inlet_u[valid]) == pytest.approx(0.47, rel=0.3)


def test_rasterize_reference_on_grid_matches_explicit_grid_shape():
    node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    xs = np.linspace(0.0, 1.0, 5)
    ys = np.linspace(0.0, 1.0, 7)

    grid = rasterize_reference_on_grid(node_coords, values, xs, ys)
    assert grid.shape == (7, 5)


def test_rasterize_reference_on_grid_nearest_neighbor_exact_at_nodes():
    """At the exact node coordinates, nearest-neighbor interpolation should
    reproduce the original value exactly."""

    node_coords = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    values = np.array([10.0, 20.0, 30.0, 40.0])
    xs = np.array([0.0, 2.0])
    ys = np.array([0.0, 2.0])

    grid = rasterize_reference_on_grid(node_coords, values, xs, ys)
    # grid[row, col] <-> (ys[row], xs[col])
    assert grid[0, 0] == pytest.approx(10.0)
    assert grid[0, 1] == pytest.approx(20.0)
    assert grid[1, 0] == pytest.approx(30.0)
    assert grid[1, 1] == pytest.approx(40.0)


def test_rasterize_reference_on_grid_aligns_pixel_for_pixel_with_given_axes(aneurysm_flow):
    """Two independent calls sharing the same explicit xs/ys must produce
    pixel-aligned rasters (the whole point of this function, vs.
    `data/generate_dataset._rasterize`'s self-derived bounding box)."""

    tm, flow = aneurysm_flow
    n_vertices = tm.mesh.p.shape[1]
    ux = flow.u[0 : 2 * n_vertices : 2]
    xs = np.linspace(0.0, 0.05, 40)
    ys = np.linspace(0.0, 0.007, 20)

    grid_a = rasterize_reference_on_grid(tm.mesh.p, ux, xs, ys)
    grid_b = rasterize_reference_on_grid(tm.mesh.p.T, ux, xs, ys)
    assert grid_a.shape == (20, 40)
    np.testing.assert_array_equal(grid_a, grid_b)
