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
from thrombus_bench.viz.rasterize_continuous import rasterize_continuous_model

GEOMETRY_PATH = "configs/geometry.yaml"


def _tiny_model_cfg() -> dict:
    return {
        "encoder": {"param_dim": 9, "latent_grid_size": (8, 8), "hidden_channels": 8, "n_layers": 1},
        "operator_core": {"type": "fno", "fno": {"modes": 2, "hidden_channels": 8, "n_layers": 1}},
        "coordinate_decoder": {"mlp_hidden": 16, "n_residual_blocks": 1},
        "output_channels": 11,
        "uncertainty": {"mc_dropout_rate": 0.1},
    }


def test_rasterize_continuous_model_output_shapes():
    model = ContinuousThrombusSurrogate(_tiny_model_cfg())
    params_with_time = torch.randn(9)
    geometry_mm = torch.tensor([7.0, 3.2])

    fields_grid, fluid_mask = rasterize_continuous_model(model, params_with_time, geometry_mm, grid_size=(16, 24))

    assert fields_grid.shape == (16, 24, 11)
    assert fluid_mask.shape == (16, 24)
    assert fluid_mask.dtype == bool


def test_rasterize_continuous_model_masks_exterior_cells_with_nan():
    model = ContinuousThrombusSurrogate(_tiny_model_cfg())
    params_with_time = torch.randn(9)
    geometry_mm = torch.tensor([7.0, 3.2])

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
    params_with_time = torch.randn(9)
    geometry_mm = torch.tensor([7.0, 3.2])
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
    geom = GeometryConfig(vessel_diameter_mm=vessel_mm, aneurysm_diameter_mm=aneurysm_mm, vessel_length_mm=50.0)
    L_m = 50.0 * 1e-3
    D_m = vessel_mm * 1e-3
    R_m = aneurysm_mm * 0.5e-3
    xs = np.linspace(0.0, L_m, grid_size[1])
    ys = np.linspace(0.0, D_m + R_m, grid_size[0])
    gx, gy = np.meshgrid(xs, ys)
    expected_mask = signed_distance_to_wall(gx.ravel(), gy.ravel(), geom).reshape(grid_size) >= 0.0

    np.testing.assert_array_equal(fluid_mask, expected_mask)


def test_rasterize_continuous_model_grid_spans_analytic_bounding_box():
    """The grid's own extent should match [0, L] x [0, D+R] exactly (the
    same bounding-box convention as the rest of this project's coordinate
    normalization), not e.g. the mesh's own (nonexistent, at inference
    time) node bounding box."""

    model = ContinuousThrombusSurrogate(_tiny_model_cfg())
    params_with_time = torch.randn(9)
    aneurysm_mm, vessel_mm = 7.0, 3.2
    geometry_mm = torch.tensor([aneurysm_mm, vessel_mm])

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
    params_with_time = torch.randn(9)
    geometry_mm = torch.tensor([aneurysm_mm, vessel_mm])

    fields_grid, fluid_mask = rasterize_continuous_model(model, params_with_time, geometry_mm, grid_size=(16, 16))
    assert fluid_mask.any()
    assert fields_grid.shape == (16, 16, 11)
