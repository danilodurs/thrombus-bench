"""Tests that the `thrombin_fibrin_reliable` flag (see mechanistic/
coupled_solver.py's `CoupledSimulationHistory` docstring) is propagated from
a mechanistic run through `generate_dataset.py`'s saved `.npz` sample and
into `dataset.py`'s `ThrombusSurrogateDataset.__getitem__` output."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from thrombus_bench.data.dataset import ThrombusSurrogateDataset
from thrombus_bench.data.generate_dataset import _run_one_sample

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


def test_run_one_sample_includes_thrombin_fibrin_reliable_flag(physio):
    mesh_cfg = {"target_num_elements": 150}
    result = _run_one_sample(_small_sample(), physio, mesh_cfg, end_time_s=0.2, dt_s=0.1, grid_size=(8, 8))
    assert "thrombin_fibrin_reliable" in result
    assert isinstance(result["thrombin_fibrin_reliable"], bool)


def test_dataset_exposes_thrombin_fibrin_reliable_as_bool_tensor(physio, tmp_path):
    mesh_cfg = {"target_num_elements": 150}
    # ThrombusSurrogateDataset needs the legacy raster representation.
    result = _run_one_sample(
        _small_sample(), physio, mesh_cfg, end_time_s=0.2, dt_s=0.1, grid_size=(8, 8), also_save_raster=True
    )

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    np.savez(split_dir / "sample_0000.npz", **result)

    dataset = ThrombusSurrogateDataset(str(tmp_path), "train")
    item = dataset[0]

    assert "thrombin_fibrin_reliable" in item
    assert item["thrombin_fibrin_reliable"].dtype == torch.bool
    assert bool(item["thrombin_fibrin_reliable"].item()) == result["thrombin_fibrin_reliable"]
