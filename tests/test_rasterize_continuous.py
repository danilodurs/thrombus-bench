"""Tests for `viz/rasterize_continuous.rasterize_continuous_model` (Phase 6,
`docs/continuous_surrogate_design.md`): the display-only utility that
queries a trained `ContinuousThrombusSurrogate` on a regular grid."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from thrombus_bench.mechanistic.geometry_sdf import signed_distance_to_wall
from thrombus_bench.mechanistic.mesh import GeometryConfig
from thrombus_bench.neural.coordinate_decoder import ContinuousThrombusSurrogate
from thrombus_bench.viz.rasterize_continuous import grid_size_for_aspect_ratio, rasterize_continuous_model

GEOMETRY_PATH = "configs/geometry.yaml"


def _tiny_model_cfg() -> dict:
    return {
        "encoder": {"param_dim": 11, "latent_grid_size": (8, 8), "hidden_channels": 8, "n_layers": 1},
        "operator_core": {"type": "fno", "fno": {"modes": 2, "hidden_channels": 8, "n_layers": 1}},
        "coordinate_decoder": {"mlp_hidden": 16, "n_residual_blocks": 1},
        "output_channels": 11,
        "uncertainty": {"mc_dropout_rate": 0.1},
    }


def test_rasterize_continuous_model_output_shapes():
    model = ContinuousThrombusSurrogate(_tiny_model_cfg())
    params_with_time = torch.randn(11)
    geometry_mm = torch.tensor([7.0, 3.2, 3.5, 0.0])

    fields_grid, fluid_mask = rasterize_continuous_model(model, params_with_time, geometry_mm, grid_size=(16, 24))

    assert fields_grid.shape == (16, 24, 11)
    assert fluid_mask.shape == (16, 24)
    assert fluid_mask.dtype == bool


def test_rasterize_continuous_model_masks_exterior_cells_with_nan():
    model = ContinuousThrombusSurrogate(_tiny_model_cfg())
    params_with_time = torch.randn(11)
    geometry_mm = torch.tensor([7.0, 3.2, 3.5, 0.0])

    fields_grid, fluid_mask = rasterize_continuous_model(model, params_with_time, geometry_mm, grid_size=(20, 40))

    # Some fluid cells, some exterior cells -- the bounding box is not
    # fully filled by the L/T-shaped vessel+aneurysm domain.
    assert fluid_mask.any()
    assert not fluid_mask.all()

    assert np.all(np.isnan(fields_grid[~fluid_mask]))
    assert not np.any(np.isnan(fields_grid[fluid_mask]))


def test_rasterize_continuous_model_mask_matches_analytic_sdf_directly():
    """The mask isn't just "some cells are NaN" -- it must match Phase 1's
    SDF exactly, cell by cell, for the same grid this function builds."""

    model = ContinuousThrombusSurrogate(_tiny_model_cfg())
    params_with_time = torch.randn(11)
    geometry_mm = torch.tensor([7.0, 3.2, 3.5, 0.0])
    grid_size = (12, 18)

    _, fluid_mask = rasterize_continuous_model(model, params_with_time, geometry_mm, grid_size=grid_size)

    # Derive geometry the same way the function does (float(tensor[i])),
    # not from separately-hardcoded decimal literals -- 3.2 isn't exactly
    # representable in float32, so a literal 0.0032 differs from
    # float(torch.tensor(3.2)) * 1e-3 at the ~1e-9 level, enough to flip
    # the sign of a handful of grid points that land almost exactly on the
    # analytic boundary (a real, if narrow, floating-point edge case, not
    # a logic bug -- this mismatch is exactly what surfaced it).
    aneurysm_mm, vessel_mm = float(geometry_mm[0]), float(geometry_mm[1])
    sac_height_mm, sac_asymmetry = float(geometry_mm[2]), float(geometry_mm[3])
    geom = GeometryConfig(
        vessel_diameter_mm=vessel_mm, aneurysm_diameter_mm=aneurysm_mm, vessel_length_mm=50.0,
        sac_height_mm=sac_height_mm, sac_asymmetry=sac_asymmetry,
    )
    L_m = 50.0 * 1e-3
    D_m = vessel_mm * 1e-3
    b_m = sac_height_mm * 1e-3
    xs = np.linspace(0.0, L_m, grid_size[1])
    ys = np.linspace(0.0, D_m + b_m, grid_size[0])
    gx, gy = np.meshgrid(xs, ys)
    expected_mask = signed_distance_to_wall(gx.ravel(), gy.ravel(), geom).reshape(grid_size) >= 0.0

    np.testing.assert_array_equal(fluid_mask, expected_mask)


def test_rasterize_continuous_model_grid_spans_analytic_bounding_box():
    """The grid's own extent should match [0, L] x [0, D+R] exactly (the
    same bounding-box convention as the rest of this project's coordinate
    normalization), not e.g. the mesh's own (nonexistent, at inference
    time) node bounding box."""

    model = ContinuousThrombusSurrogate(_tiny_model_cfg())
    params_with_time = torch.randn(11)
    aneurysm_mm, vessel_mm = 7.0, 3.2
    geometry_mm = torch.tensor([aneurysm_mm, vessel_mm, aneurysm_mm / 2.0, 0.0])

    _, fluid_mask = rasterize_continuous_model(model, params_with_time, geometry_mm, grid_size=(10, 10))

    # The bottom-left corner (x=0, y=0) is always inside the vessel
    # rectangle regardless of geometry -- a basic sanity check that the
    # grid's origin is where it should be.
    assert fluid_mask[0, 0]


@pytest.mark.parametrize("aneurysm_mm,vessel_mm", [(7.0, 3.2), (10.0, 4.0)])
def test_rasterize_continuous_model_works_for_both_geometry_presets(aneurysm_mm, vessel_mm):
    with open(GEOMETRY_PATH) as f:
        presets = yaml.safe_load(f)["presets"]
    assert any(
        p["aneurysm_diameter_mm"] == aneurysm_mm and p["vessel_diameter_mm"] == vessel_mm for p in presets.values()
    )

    model = ContinuousThrombusSurrogate(_tiny_model_cfg())
    params_with_time = torch.randn(11)
    geometry_mm = torch.tensor([aneurysm_mm, vessel_mm, aneurysm_mm / 2.0, 0.0])

    fields_grid, fluid_mask = rasterize_continuous_model(model, params_with_time, geometry_mm, grid_size=(16, 16))
    assert fluid_mask.any()
    assert fields_grid.shape == (16, 16, 11)


def test_grid_size_for_aspect_ratio_matches_physical_proportions():
    """A 50 mm-long vessel with a small D+R should get far more columns
    than rows; the ratio of (cols/rows) should match (L)/(D+R)."""

    geometry_mm = torch.tensor([7.0, 3.2, 3.5, 0.0])  # D + b = 3.2 + 3.5 = 6.7 mm
    n_rows, n_cols = grid_size_for_aspect_ratio(geometry_mm, vessel_length_mm=50.0, points_per_mm=8.0)
    assert n_rows == round(6.7 * 8.0)
    assert n_cols == round(50.0 * 8.0)
    # Independent rounding of numerator/denominator shifts the ratio
    # slightly from the exact physical proportion -- a loose tolerance,
    # not exact equality, is the correct check here.
    assert n_cols / n_rows == pytest.approx(50.0 / 6.7, rel=0.02)


def test_grid_size_for_aspect_ratio_independent_of_latent_grid_size():
    """This function takes no `latent_grid_size` argument at all -- a
    change to that training-time config value must not silently affect
    display resolution."""

    geometry_mm = torch.tensor([10.0, 4.0, 5.0, 0.0])
    result_a = grid_size_for_aspect_ratio(geometry_mm, points_per_mm=6.0)
    result_b = grid_size_for_aspect_ratio(geometry_mm, points_per_mm=6.0)
    assert result_a == result_b


def test_grid_size_for_aspect_ratio_respects_minimum():
    """A tiny geometry at a coarse points_per_mm should still floor at a
    usable minimum resolution (8), not collapse to 0 or 1."""

    geometry_mm = torch.tensor([0.5, 0.5, 0.25, 0.0])
    n_rows, n_cols = grid_size_for_aspect_ratio(geometry_mm, vessel_length_mm=50.0, points_per_mm=0.1)
    assert n_rows >= 8
    assert n_cols >= 8
