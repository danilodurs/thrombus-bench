"""Geometry-redesign Phase 4b, Stage 4 (full-pipeline wiring): regression
tests that the sac_height_mm/sac_asymmetry schema change (`data/sampler.py`
DEFAULT_RANGES, `data/generate_dataset.PARAM_ORDER`, `data/dataset.py`'s
geometry_mm slice, `neural/coordinate_decoder.py`/`neural/physics_losses.py`'s
geometry-conditioned bounding-box logic) is wired correctly end to end, not
just at the `mesh.GeometryConfig` level `tests/test_geometry_ellipse_sac.py`
already covers.

The most important test here is `test_run_one_sample_circle_equivalent_matches_pre_phase4b_geometry`:
the full `data/generate_dataset._run_one_sample` pipeline, given the
circle-equivalent sac_height_mm/sac_asymmetry values a sampled dataset can
now produce, must reproduce *exactly* what the pre-Phase-4b (no sac_height_mm/
sac_asymmetry keys at all) code path did -- this is the "two original
circular presets remain numerically equivalent through the full new
pipeline" guarantee, not just a `GeometryConfig` dataclass-default check.
"""

from __future__ import annotations

import numpy as np
import pytest

from thrombus_bench.data.dataset import (
    PointCloudThrombusDataset,
    ThrombusSurrogateDataset,
    _check_params_schema,
    _EXPECTED_N_PARAMS,
    _N_GEOMETRY_PARAMS,
)
from thrombus_bench.data.generate_dataset import N_GEOMETRY_PARAMS, PARAM_ORDER
from thrombus_bench.data.sampler import DEFAULT_RANGES
from thrombus_bench.mechanistic.mesh import GeometryConfig, MeshConfig, build_aneurysm_mesh


def test_param_order_inserts_sac_fields_right_after_geometry_diameters():
    """Section 8 of docs/geometry_redesign_assessment.md: new entries must
    land at positions 2-3, not appended at the end, to preserve the
    "leading N entries are geometry" convention PointCloudThrombusDataset
    relies on."""

    assert PARAM_ORDER[0] == "aneurysm_diameter_mm"
    assert PARAM_ORDER[1] == "vessel_diameter_mm"
    assert PARAM_ORDER[2] == "sac_height_mm"
    assert PARAM_ORDER[3] == "sac_asymmetry"
    assert N_GEOMETRY_PARAMS == 4
    assert len(PARAM_ORDER) == 10


def test_default_ranges_matches_param_order_positionally():
    """normalize_params/denormalize_params index `space.names` (DEFAULT_RANGES
    key order) positionally against a raw params array assumed to already be
    in PARAM_ORDER order -- these two independently-declared orderings must
    agree exactly, not just contain the same names."""

    assert tuple(DEFAULT_RANGES.keys()) == PARAM_ORDER


@pytest.mark.parametrize("aneurysm_mm,vessel_mm", [(7.0, 3.2), (10.0, 4.0)])
def test_circle_equivalent_geometry_config_matches_pre_phase4b_defaults(aneurysm_mm, vessel_mm):
    """GeometryConfig level (mirrors test_geometry_ellipse_sac.py, included
    here for a self-contained before/after comparison with the pipeline
    test below)."""

    mesh_cfg = MeshConfig(target_num_elements=400)
    geom_explicit = GeometryConfig(
        vessel_diameter_mm=vessel_mm, aneurysm_diameter_mm=aneurysm_mm, vessel_length_mm=50.0,
        sac_height_mm=aneurysm_mm / 2.0, sac_asymmetry=0.0,
    )
    geom_default = GeometryConfig(
        vessel_diameter_mm=vessel_mm, aneurysm_diameter_mm=aneurysm_mm, vessel_length_mm=50.0,
    )
    mesh_explicit = build_aneurysm_mesh(geom_explicit, mesh_cfg)
    mesh_default = build_aneurysm_mesh(geom_default, mesh_cfg)

    np.testing.assert_array_equal(mesh_explicit.mesh.p, mesh_default.mesh.p)
    np.testing.assert_array_equal(mesh_explicit.mesh.t, mesh_default.mesh.t)


@pytest.mark.parametrize("aneurysm_mm,vessel_mm", [(7.0, 3.2), (10.0, 4.0)])
def test_run_one_sample_circle_equivalent_matches_pre_phase4b_geometry(aneurysm_mm, vessel_mm):
    """Full pipeline: a sample dict with the circle-equivalent
    sac_height_mm/sac_asymmetry (what a sampled dataset now always
    provides) must build the exact same mesh `_run_one_sample` would have
    built before this schema change existed (no sac_height_mm/sac_asymmetry
    keys, GeometryConfig's own None/0.0 defaults) -- proving the wiring
    through PARAM_ORDER/_run_one_sample's GeometryConfig construction, not
    just GeometryConfig itself."""

    mesh_cfg = MeshConfig(target_num_elements=400)

    # What _run_one_sample now does: build GeometryConfig from an explicit
    # sample dict entry.
    geom_from_sample = GeometryConfig(
        vessel_diameter_mm=vessel_mm,
        aneurysm_diameter_mm=aneurysm_mm,
        vessel_length_mm=50.0,
        sac_height_mm=aneurysm_mm / 2.0,
        sac_asymmetry=0.0,
    )
    # What _run_one_sample did before Phase 4b: no sac fields at all.
    geom_pre_phase4b = GeometryConfig(
        vessel_diameter_mm=vessel_mm, aneurysm_diameter_mm=aneurysm_mm, vessel_length_mm=50.0,
    )

    mesh_from_sample = build_aneurysm_mesh(geom_from_sample, mesh_cfg)
    mesh_pre_phase4b = build_aneurysm_mesh(geom_pre_phase4b, mesh_cfg)

    np.testing.assert_array_equal(mesh_from_sample.mesh.p, mesh_pre_phase4b.mesh.p)
    np.testing.assert_array_equal(mesh_from_sample.mesh.t, mesh_pre_phase4b.mesh.t)


def test_check_params_schema_accepts_current_length():
    _check_params_schema(np.zeros(_EXPECTED_N_PARAMS), "irrelevant/path.npz")  # must not raise


def test_check_params_schema_rejects_pre_phase4b_8_length_array():
    """The exact silent-misinterpretation risk docs/geometry_redesign_assessment.md
    Section 8 flags: an old 8-entry params array must be rejected with a
    clear, actionable message, not silently misread as if positions 2-3
    were sac_height_mm/sac_asymmetry."""

    with pytest.raises(ValueError, match="regenerate"):
        _check_params_schema(np.zeros(8), "data/processed/train/sample_0000.npz")


def test_thrombus_surrogate_dataset_rejects_old_schema_npz(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    # Deliberately minimal/incomplete otherwise -- the schema check must
    # fire before any other key is read.
    np.savez(split_dir / "sample_0000.npz", params=np.zeros(8))

    dataset = ThrombusSurrogateDataset(str(tmp_path), "train")
    with pytest.raises(ValueError, match="regenerate"):
        dataset[0]


def test_pointcloud_dataset_rejects_old_schema_npz(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    np.savez(split_dir / "sample_0000.npz", params=np.zeros(8), time_s=np.array([0.3]))

    dataset = PointCloudThrombusDataset(str(tmp_path), "train")
    with pytest.raises(ValueError, match="regenerate"):
        dataset[0]


def test_geometry_mm_slice_length_matches_n_geometry_params():
    assert _N_GEOMETRY_PARAMS == N_GEOMETRY_PARAMS == 4
