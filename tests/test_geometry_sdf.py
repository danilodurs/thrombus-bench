"""Tests for `mechanistic/geometry_sdf.signed_distance_to_wall`.

Sign convention under test (see module docstring): positive inside the
fluid domain, negative outside, zero on the wall.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from scipy.spatial import cKDTree
from skfem import Basis, ElementTriP1

from thrombus_bench.mechanistic.geometry_sdf import signed_distance_to_wall
from thrombus_bench.mechanistic.mesh import GeometryConfig, build_aneurysm_mesh

GEOMETRY_PATH = "configs/geometry.yaml"


@pytest.fixture
def aneurysm_7mm_geom() -> GeometryConfig:
    with open(GEOMETRY_PATH) as f:
        presets = yaml.safe_load(f)["presets"]
    return GeometryConfig.from_preset(presets["aneurysm_7mm"])


def test_vessel_centerline_point_is_inside_with_half_diameter_magnitude(aneurysm_7mm_geom):
    D = aneurysm_7mm_geom.vessel_diameter_mm * 1e-3
    # x=5mm, well clear of both the neck and the inlet -- nearest wall is
    # straight up/down at distance D/2.
    x, y = 0.005, D / 2.0
    d = signed_distance_to_wall(x, y, aneurysm_7mm_geom)
    assert d > 0.0
    assert d == pytest.approx(D / 2.0, abs=1e-9)


def test_point_on_bottom_wall_is_zero(aneurysm_7mm_geom):
    d = signed_distance_to_wall(0.01, 0.0, aneurysm_7mm_geom)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_point_well_outside_domain_is_negative(aneurysm_7mm_geom):
    D = aneurysm_7mm_geom.vessel_diameter_mm * 1e-3
    R = aneurysm_7mm_geom.aneurysm_diameter_mm * 1e-3 / 2.0
    L = aneurysm_7mm_geom.vessel_length_mm * 1e-3
    xc = L / 2.0
    # Directly above the sac center, well above its apex -- nearest wall is
    # the top of the arc at (xc, D+R).
    x, y = xc, 0.05
    d = signed_distance_to_wall(x, y, aneurysm_7mm_geom)
    assert d < 0.0
    assert d == pytest.approx(-(y - (D + R)), abs=1e-9)


def test_point_inside_aneurysm_sac_center_is_positive(aneurysm_7mm_geom):
    D = aneurysm_7mm_geom.vessel_diameter_mm * 1e-3
    R = aneurysm_7mm_geom.aneurysm_diameter_mm * 1e-3 / 2.0
    L = aneurysm_7mm_geom.vessel_length_mm * 1e-3
    xc = L / 2.0
    x, y = xc, D + 0.5 * R
    d = signed_distance_to_wall(x, y, aneurysm_7mm_geom)
    assert d > 0.0
    assert d == pytest.approx(0.5 * R, abs=1e-9)


def test_neck_junction_point_uses_arc_not_naive_min_max(aneurysm_7mm_geom):
    """A point just past the right neck, slightly inside the sac footprint
    at a height just above the vessel top wall -- if the SDF were computed
    as a naive min/max of two independent whole-shape SDFs (full rectangle,
    full circle) rather than the actual union boundary, it would report the
    wrong nearest point/distance right at this transition. Cross-checked
    here against a direct brute-force distance to a dense polygon
    approximation of the true boundary."""

    D = aneurysm_7mm_geom.vessel_diameter_mm * 1e-3
    R = aneurysm_7mm_geom.aneurysm_diameter_mm * 1e-3 / 2.0
    L = aneurysm_7mm_geom.vessel_length_mm * 1e-3
    xc = L / 2.0
    xr = xc + R
    x, y = xr - 0.2 * R, D + 0.05 * R

    d = signed_distance_to_wall(x, y, aneurysm_7mm_geom)

    # Brute-force reference: dense sample of the true analytic boundary
    # (straight edges + exact circular arc), minimum Euclidean distance.
    theta = np.linspace(0.0, np.pi, 20000)
    arc = np.column_stack([xc + R * np.cos(theta), D + R * np.sin(theta)])
    xl = xc - R
    edges = np.vstack(
        [
            np.column_stack([np.linspace(0.0, L, 5000), np.zeros(5000)]),
            np.column_stack([np.full(5000, L), np.linspace(0.0, D, 5000)]),
            np.column_stack([np.linspace(xr, L, 2000), np.full(2000, D)]),
            np.column_stack([np.linspace(0.0, xl, 2000), np.full(2000, D)]),
            np.column_stack([np.zeros(5000), np.linspace(0.0, D, 5000)]),
        ]
    )
    boundary = np.vstack([arc, edges])
    ref_dist = np.min(np.hypot(boundary[:, 0] - x, boundary[:, 1] - y))

    assert abs(d) == pytest.approx(ref_dist, abs=1e-6)


def test_signed_distance_supports_array_input(aneurysm_7mm_geom):
    xs = np.array([0.005, 0.005, 0.1])
    ys = np.array([0.0016, -0.01, 0.0016])
    d = signed_distance_to_wall(xs, ys, aneurysm_7mm_geom)
    assert d.shape == (3,)
    assert d[0] > 0.0  # inside
    assert d[1] < 0.0  # below the vessel, outside
    assert d[2] < 0.0  # past the outlet, outside


def test_sdf_matches_mesh_boundary_nodes(aneurysm_7mm_geom):
    """Integration sanity check: the actual FEM mesh's own boundary nodes
    (built from the identical analytic outline, see mesh.py's
    `_build_boundary_polygon`) should sit almost exactly on the SDF's
    zero-level set."""

    tagged = build_aneurysm_mesh(aneurysm_7mm_geom, {"target_num_elements": 800})
    basis = Basis(tagged.mesh, ElementTriP1())
    boundary_names = ("inlet", "outlet", "wall_vessel", "wall_sac")
    dofs = np.unique(
        np.concatenate([basis.get_dofs(name).all() for name in boundary_names if len(tagged.mesh.boundaries[name])])
    )
    coords = basis.doflocs[:, dofs]

    d = signed_distance_to_wall(coords[0], coords[1], aneurysm_7mm_geom)
    assert np.max(np.abs(d)) < 1e-8, "mesh boundary nodes should lie essentially exactly on the analytic outline"


def test_sdf_matches_nearest_mesh_boundary_node_distance(aneurysm_7mm_geom):
    """Broader cross-check across interior/exterior sample points: |SDF|
    should approximately match the distance to the nearest actual mesh
    boundary node, within the mesh's own discretization spacing."""

    tagged = build_aneurysm_mesh(aneurysm_7mm_geom, {"target_num_elements": 800})
    basis = Basis(tagged.mesh, ElementTriP1())
    boundary_names = ("inlet", "outlet", "wall_vessel", "wall_sac")
    dofs = np.unique(
        np.concatenate([basis.get_dofs(name).all() for name in boundary_names if len(tagged.mesh.boundaries[name])])
    )
    boundary_coords = basis.doflocs[:, dofs].T
    tree = cKDTree(boundary_coords)

    D = aneurysm_7mm_geom.vessel_diameter_mm * 1e-3
    R = aneurysm_7mm_geom.aneurysm_diameter_mm * 1e-3 / 2.0
    L = aneurysm_7mm_geom.vessel_length_mm * 1e-3
    rng = np.random.default_rng(0)
    xs = rng.uniform(-0.005, L + 0.005, size=200)
    ys = rng.uniform(-0.005, D + R + 0.005, size=200)

    d_sdf = np.abs(signed_distance_to_wall(xs, ys, aneurysm_7mm_geom))
    d_nearest_node, _ = tree.query(np.column_stack([xs, ys]))

    # Nearest-*node* distance is a discrete approximation of the true
    # nearest-*boundary* distance, so it is systematically >= the analytic
    # value; bound the gap by a few mesh-spacing lengths (h ~ 0.7mm at this
    # resolution).
    h_approx = 7.5e-4
    assert np.all(d_nearest_node >= d_sdf - 1e-9)
    assert np.max(d_nearest_node - d_sdf) < 3.0 * h_approx
