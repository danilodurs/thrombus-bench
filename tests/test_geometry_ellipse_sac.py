"""Regression + new-behavior tests for the geometry-redesign Phase 4b
(asymmetric half-ellipse sac, Option A) -- see
`docs/geometry_redesign_assessment.md`.

The most important tests here are the "exact preservation" ones: Phase 4b
was explicitly required to leave `configs/geometry.yaml`'s two paper-
matching presets (`aneurysm_7mm`, `aneurysm_10mm`) byte-identical, not
merely visually similar -- see `mesh.py::GeometryConfig`'s docstring.
`tests/test_geometry_sdf.py`/`test_mechanistic_conservation.py`'s existing
tight-tolerance tests (written before this change, still passing
unmodified) already provide indirect evidence of this; the tests below
make the invariant explicit and independent of those files' continued
existence.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from skfem import Basis, ElementTriP1

from thrombus_bench.mechanistic.flow_solver import CarreauParams, compute_boundary_flux, solve_steady_flow
from thrombus_bench.mechanistic.geometry_sdf import signed_distance_to_wall
from thrombus_bench.mechanistic.mesh import (
    GeometryConfig,
    MeshConfig,
    _aneurysm_geometry_points,
    build_aneurysm_mesh,
)

GEOMETRY_PATH = "configs/geometry.yaml"
CARREAU = CarreauParams(mu_inf_pa_s=0.0035, mu_0_pa_s=0.056, lambda_s=3.313, n=0.3568)


def _load_presets() -> dict:
    with open(GEOMETRY_PATH) as f:
        return yaml.safe_load(f)["presets"]


@pytest.mark.parametrize("preset_name", ["aneurysm_7mm", "aneurysm_10mm"])
def test_configs_geometry_yaml_presets_have_no_new_keys(preset_name):
    """The two paper presets must need zero changes to configs/geometry.yaml
    -- Phase 4b's new keys are additive and optional, not required."""

    preset = _load_presets()[preset_name]
    assert "sac_height_mm" not in preset
    assert "sac_asymmetry" not in preset


@pytest.mark.parametrize("preset_name", ["aneurysm_7mm", "aneurysm_10mm"])
def test_default_params_reduce_exactly_to_circle(preset_name):
    """sac_height_mm=None, sac_asymmetry=0.0 (dataclass defaults) must give
    a_left == a_right == b == R exactly, not approximately -- this is the
    invariant the "preserve both paper presets exactly" requirement
    depends on."""

    geom = GeometryConfig.from_preset(_load_presets()[preset_name])
    assert geom.sac_height_mm is None
    assert geom.sac_asymmetry == 0.0

    g = _aneurysm_geometry_points(geom)
    assert g["a_left"] == g["R"]
    assert g["a_right"] == g["R"]
    assert g["b"] == g["R"]
    assert g["xc_ellipse"] == g["xc"]


def test_from_preset_reads_new_keys_when_present():
    preset = {"vessel_diameter_mm": 3.2, "aneurysm_diameter_mm": 7.0, "vessel_length_mm": 50.0, "sac_height_mm": 5.0, "sac_asymmetry": 0.3}
    geom = GeometryConfig.from_preset(preset)
    assert geom.sac_height_mm == 5.0
    assert geom.sac_asymmetry == 0.3


def test_sac_asymmetry_out_of_range_raises():
    geom = GeometryConfig(vessel_diameter_mm=3.2, aneurysm_diameter_mm=7.0, vessel_length_mm=50.0, sac_asymmetry=1.0)
    with pytest.raises(ValueError, match="sac_asymmetry"):
        _aneurysm_geometry_points(geom)


def test_positive_asymmetry_shifts_apex_toward_outlet_without_moving_neck():
    g0 = _aneurysm_geometry_points(GeometryConfig(3.2, 7.0, 50.0, sac_asymmetry=0.0))
    g_pos = _aneurysm_geometry_points(GeometryConfig(3.2, 7.0, 50.0, sac_asymmetry=0.3))
    g_neg = _aneurysm_geometry_points(GeometryConfig(3.2, 7.0, 50.0, sac_asymmetry=-0.3))

    assert g_pos["xc_ellipse"] > g0["xc_ellipse"]
    assert g_neg["xc_ellipse"] < g0["xc_ellipse"]
    # Neck attachment points must stay exactly fixed regardless of asymmetry
    # -- this is why the vessel-wall-stub/tagging logic needed no changes.
    for g in (g_pos, g_neg):
        assert g["x_neck_left"] == g0["x_neck_left"]
        assert g["x_neck_right"] == g0["x_neck_right"]


def test_sac_height_mm_changes_height_only():
    g_default = _aneurysm_geometry_points(GeometryConfig(3.2, 7.0, 50.0))
    g_tall = _aneurysm_geometry_points(GeometryConfig(3.2, 7.0, 50.0, sac_height_mm=6.0))

    assert g_tall["b"] == pytest.approx(6.0e-3)
    assert g_tall["x_neck_left"] == g_default["x_neck_left"]
    assert g_tall["x_neck_right"] == g_default["x_neck_right"]
    assert g_tall["a_left"] == g_default["a_left"]
    assert g_tall["a_right"] == g_default["a_right"]


NEW_SHAPES = [
    pytest.param(None, 0.0, id="default-circle"),
    pytest.param(5.0, 0.4, id="tall-asymmetric"),
    pytest.param(6.0, 0.7, id="tall-very-asymmetric"),
    pytest.param(1.5, -0.6, id="short-negative-asymmetric"),
    pytest.param(7.0, -0.2, id="tall-mild-negative-asymmetric"),
]


@pytest.mark.parametrize("sac_height_mm,sac_asymmetry", NEW_SHAPES)
def test_mesh_is_watertight_for_new_shapes(sac_height_mm, sac_asymmetry):
    geom = GeometryConfig(3.2, 7.0, 50.0, sac_height_mm=sac_height_mm, sac_asymmetry=sac_asymmetry)
    tm = build_aneurysm_mesh(geom, MeshConfig(target_num_elements=1200))
    m = tm.mesh
    total_boundary = set(m.boundary_facets().tolist())
    tagged = set()
    for facets in m.boundaries.values():
        tagged.update(facets.tolist())
    assert total_boundary == tagged


@pytest.mark.parametrize("sac_height_mm,sac_asymmetry", NEW_SHAPES)
def test_boundary_tags_non_overlapping_for_new_shapes(sac_height_mm, sac_asymmetry):
    """Every boundary facet must be tagged as exactly one of the four names
    -- an untagged or double-tagged facet silently breaks BCs / mass
    conservation (see mesh.py's own docstring warning about this)."""

    geom = GeometryConfig(3.2, 7.0, 50.0, sac_height_mm=sac_height_mm, sac_asymmetry=sac_asymmetry)
    tm = build_aneurysm_mesh(geom, MeshConfig(target_num_elements=1200))
    counts: dict[int, int] = {}
    for facets in tm.mesh.boundaries.values():
        for f in facets.tolist():
            counts[f] = counts.get(f, 0) + 1
    assert counts, "expected at least one tagged boundary facet"
    assert all(c == 1 for c in counts.values())


@pytest.mark.parametrize("sac_height_mm,sac_asymmetry", NEW_SHAPES)
def test_flow_converges_and_conserves_mass_on_new_shapes(sac_height_mm, sac_asymmetry):
    geom = GeometryConfig(3.2, 7.0, 50.0, sac_height_mm=sac_height_mm, sac_asymmetry=sac_asymmetry)
    tm = build_aneurysm_mesh(geom, MeshConfig(target_num_elements=1500))
    flow = solve_steady_flow(tm, inlet_velocity_m_s=0.47, carreau=CARREAU)
    assert flow.converged
    q_in = compute_boundary_flux(flow, "inlet")
    q_out = compute_boundary_flux(flow, "outlet")
    assert abs(q_in + q_out) / abs(q_in) < 1e-6


@pytest.mark.parametrize("sac_height_mm,sac_asymmetry", NEW_SHAPES)
def test_sdf_matches_mesh_boundary_nodes_for_new_shapes(sac_height_mm, sac_asymmetry):
    """Same style of check as test_geometry_sdf.py's exact-circle version,
    generalized to the new ellipse-based shapes: mesh boundary nodes
    should sit almost exactly on the SDF's zero-level set.

    Tolerance `2e-4` m (0.2mm), not the circle-only test's `1e-8`:
    confirmed during Phase 4b development, by reproducing the same
    residual bit-for-bit against the original, pre-Phase-4b circle-only
    code, that `is_on_arc`'s boundary-tagging tolerance
    (`arc_tol = max(0.6*h, 5*tol)`, `mesh.py::build_aneurysm_mesh`) is
    *deliberately* mesh-spacing-scale, not machine-precision-scale (its
    own docstring: "must accommodate the chord sagitta... tied to the
    background mesh spacing h") -- an occasional interior-fill point can
    end up tagged as a boundary node while sitting up to roughly that
    tolerance away from the true analytic curve, independent of whether
    the curve is a circle or the new ellipse. This is a pre-existing
    characteristic of the tagging tolerance design, not something this
    geometry-shape change introduced or is in scope to fix; `2e-4` is a
    generous-but-far-from-mesh-spacing bound confirming the ellipse SDF
    itself isn't adding meaningfully more error than the circle case
    already has."""

    geom = GeometryConfig(3.2, 7.0, 50.0, sac_height_mm=sac_height_mm, sac_asymmetry=sac_asymmetry)
    tm = build_aneurysm_mesh(geom, MeshConfig(target_num_elements=800))
    basis = Basis(tm.mesh, ElementTriP1())
    boundary_names = ("inlet", "outlet", "wall_vessel", "wall_sac")
    dofs = np.unique(
        np.concatenate([basis.get_dofs(name).all() for name in boundary_names if len(tm.mesh.boundaries[name])])
    )
    coords = basis.doflocs[:, dofs]
    d = signed_distance_to_wall(coords[0], coords[1], geom)
    assert np.max(np.abs(d)) < 2e-4


@pytest.mark.parametrize("sac_asymmetry", [0.4, 0.7, -0.6])
def test_sdf_apex_point_and_interior_point_consistency(sac_asymmetry):
    """A direct check that xc_ellipse/b are wired consistently between
    mesh.py and geometry_sdf.py: the (possibly shifted) apex itself is
    exactly on the boundary (SDF == 0), and a point directly below it,
    part-way down toward the wall, is a positive distance inside."""

    geom = GeometryConfig(3.2, 7.0, 50.0, sac_height_mm=5.0, sac_asymmetry=sac_asymmetry)
    g = _aneurysm_geometry_points(geom)
    xc_ellipse, D, b = g["xc_ellipse"], g["D"], g["b"]

    d_apex = signed_distance_to_wall(xc_ellipse, D + b, geom)
    assert d_apex == pytest.approx(0.0, abs=1e-9)

    # Distance to the apex itself (one specific boundary point) is exactly
    # 0.5*b; the true nearest distance is at most that (and, for an
    # asymmetric ellipse, generally strictly less -- some point on the
    # narrower side's arc is closer than going all the way to the apex).
    d_interior = signed_distance_to_wall(xc_ellipse, D + 0.5 * b, geom)
    assert 0.0 < d_interior <= 0.5 * b + 1e-9
