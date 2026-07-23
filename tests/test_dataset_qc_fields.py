"""Tests for the QC fields added to generate_dataset.py's saved samples and
dataset.py's ThrombusSurrogateDataset output: flow_n_iterations/flow_residual
(mechanistic/flow_solver.FlowSolution's Picard diagnostics), clip_counts
(per-species cumulative concentration-cap clip events, from
mechanistic/coupled_solver.CoupledSimulationHistory.clip_event_counts), and
conc_min/conc_max (per-species raw nodal field extrema)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from thrombus_bench.data.dataset import _SPECIES_NAMES, ThrombusSurrogateDataset
from thrombus_bench.data.generate_dataset import _ALL_SPECIES, _run_one_sample

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


def test_species_name_order_matches_between_dataset_and_generate_dataset():
    assert _SPECIES_NAMES == _ALL_SPECIES


def test_run_one_sample_includes_qc_fields(physio):
    mesh_cfg = {"target_num_elements": 150}
    result = _run_one_sample(_small_sample(), physio, mesh_cfg, end_time_s=0.2, dt_s=0.1, grid_size=(8, 8))

    assert isinstance(result["flow_n_iterations"], int)
    assert isinstance(result["flow_residual"], float)
    assert np.isfinite(result["flow_residual"])

    for name in _ALL_SPECIES:
        assert isinstance(result[f"clip_count_{name}"], int)
        assert result[f"clip_count_{name}"] >= 0
        assert np.isfinite(result[f"conc_{name}_min"])
        assert np.isfinite(result[f"conc_{name}_max"])
        assert result[f"conc_{name}_min"] <= result[f"conc_{name}_max"]


def test_dataset_exposes_qc_fields_matching_saved_sample(physio, tmp_path):
    mesh_cfg = {"target_num_elements": 150}
    result = _run_one_sample(_small_sample(), physio, mesh_cfg, end_time_s=0.2, dt_s=0.1, grid_size=(8, 8))

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    np.savez(split_dir / "sample_0000.npz", **result)

    dataset = ThrombusSurrogateDataset(str(tmp_path), "train")
    item = dataset[0]

    assert item["flow_n_iterations"].dtype == torch.int64
    assert int(item["flow_n_iterations"]) == result["flow_n_iterations"]
    assert item["flow_residual"].dtype == torch.float32
    assert float(item["flow_residual"]) == pytest.approx(result["flow_residual"], rel=1e-5)

    for key, dtype in (("clip_counts", torch.int64), ("conc_min", torch.float32), ("conc_max", torch.float32)):
        assert item[key].shape == (len(_SPECIES_NAMES),)
        assert item[key].dtype == dtype

    for i, name in enumerate(_SPECIES_NAMES):
        assert int(item["clip_counts"][i]) == result[f"clip_count_{name}"]
        assert float(item["conc_min"][i]) == pytest.approx(result[f"conc_{name}_min"], rel=1e-5)
        assert float(item["conc_max"][i]) == pytest.approx(result[f"conc_{name}_max"], rel=1e-5)
