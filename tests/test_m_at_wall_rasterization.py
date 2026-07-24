"""Tests for generate_dataset.py's _rasterize_wall_band (the M_at_wall
raster: surface_ode.SurfaceState.M_at, a wall-only field, rasterized into
a narrow band around the wall on the same fixed grid as the bulk fields)
and its wiring into _run_one_sample / dataset.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from thrombus_bench.data.dataset import ThrombusSurrogateDataset
from thrombus_bench.data.generate_dataset import _rasterize_wall_band, _run_one_sample

PHYSIO_PATH = "configs/physio_params.yaml"


@pytest.fixture
def physio():
    with open(PHYSIO_PATH) as f:
        return yaml.safe_load(f)


def _small_sample() -> dict:
    return {
        "aneurysm_diameter_mm": 7.0,
        "vessel_diameter_mm": 3.2,
        "inlet_velocity_cm_s": 47.0,
        "platelet_conc_plt_ml": 3.5e8,
        "heparin_conc_uM": 2.0,
        "prothrombin_uM": 1.1,
        "antithrombin_uM": 2.844,
        "fibrinogen_uM": 7.0,
    }


def test_wall_adjacent_cell_nonzero_and_far_cell_zero():
    """Synthetic domain: a 10x10 square with a "wall" running along y=0,
    uniform value 7.0. An 11x11 grid (dx=dy=1.0) puts the bottom row of
    grid cells exactly on the wall (distance 0 -> nonzero) and the top row
    10 units away (>> the 1.5-cell threshold -> zero)."""

    node_coords = np.array([[0.0, 10.0, 0.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    wall_x = np.linspace(0.0, 10.0, 11)
    wall_node_coords = np.array([wall_x, np.zeros_like(wall_x)])
    wall_values = np.full_like(wall_x, 7.0)

    raster = _rasterize_wall_band(node_coords, wall_node_coords, wall_values, grid_size=(11, 11))

    assert raster.shape == (11, 11)
    assert raster.dtype == np.float32
    # Bottom row (y=0): on the wall itself -- must be nonzero (== 7.0).
    assert np.allclose(raster[0, :], 7.0)
    # Top row (y=10): 10 units from the wall, far beyond the ~1.5-cell
    # (1.5 units) threshold -- must be exactly zero.
    assert np.allclose(raster[-1, :], 0.0)


def test_band_width_threshold_respected():
    """A cell exactly at the threshold distance should still be included
    (<=); one just past it should not."""

    node_coords = np.array([[0.0, 10.0], [0.0, 10.0]])
    wall_node_coords = np.array([[5.0], [0.0]])
    wall_values = np.array([3.0])
    grid_size = (11, 11)  # dy = 1.0, threshold = 1.5 * 1.0 = 1.5

    raster = _rasterize_wall_band(node_coords, wall_node_coords, wall_values, grid_size, band_width_cells=1.5)

    # Row index 1 is y=1.0 (distance 1.0 <= 1.5) -> nonzero.
    assert raster[1, 5] == pytest.approx(3.0)
    # Row index 2 is y=2.0 (distance 2.0 > 1.5) -> zero.
    assert raster[2, 5] == pytest.approx(0.0)


def test_run_one_sample_m_at_wall_forms_a_band(physio):
    """End-to-end (real mechanistic run): M_at_wall should have some zero
    cells (fluid interior / exterior, away from the wall) and some nonzero
    cells (the wall band) -- not uniformly one or the other."""

    mesh_cfg = {"target_num_elements": 150}
    result = _run_one_sample(_small_sample(), physio, mesh_cfg, end_time_s=0.2, dt_s=0.1, grid_size=(16, 16))

    m_at_wall = result["M_at_wall"]
    assert m_at_wall.shape == (16, 16)
    assert m_at_wall.dtype == np.float32
    assert np.all(m_at_wall >= 0.0)
    assert np.any(m_at_wall == 0.0)
    assert np.any(m_at_wall > 0.0)


def test_dataset_exposes_m_at_wall_log_compressed(physio, tmp_path):
    mesh_cfg = {"target_num_elements": 150}
    result = _run_one_sample(_small_sample(), physio, mesh_cfg, end_time_s=0.2, dt_s=0.1, grid_size=(16, 16))

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    np.savez(split_dir / "sample_0000.npz", **result)

    dataset = ThrombusSurrogateDataset(str(tmp_path), "train")
    item = dataset[0]

    assert "M_at_wall" in item
    assert item["M_at_wall"].shape == (16, 16)
    assert item["M_at_wall"].dtype == torch.float32

    expected_log = np.sign(result["M_at_wall"]) * np.log1p(np.abs(result["M_at_wall"]))
    np.testing.assert_allclose(item["M_at_wall"].numpy(), expected_log, rtol=1e-5, atol=1e-6)
