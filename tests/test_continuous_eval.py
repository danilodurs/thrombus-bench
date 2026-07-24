"""Tests for the continuous-path degradation evaluators added in Phase 7
(`docs/continuous_surrogate_design.md`): `edge_holdout_eval.
evaluate_edge_holdout_degradation_continuous` and `extrapolation_eval.
evaluate_extrapolation_degradation_continuous`. Uses real (small)
generated point-cloud data, not synthetic mocks, since the point of these
functions is wiring several real pieces (PointCloudThrombusDataset,
pointcloud_collate_fn, field_rmse_pointwise, a continuous model) together
correctly."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from thrombus_bench.benchmark.edge_holdout_eval import evaluate_edge_holdout_degradation_continuous
from thrombus_bench.benchmark.extrapolation_eval import evaluate_extrapolation_degradation_continuous
from thrombus_bench.data.dataset import PointCloudThrombusDataset
from thrombus_bench.data.generate_dataset import _run_one_sample
from thrombus_bench.neural.baselines import ContinuousMeanFieldBaseline

PHYSIO_PATH = "configs/physio_params.yaml"


@pytest.fixture
def physio():
    with open(PHYSIO_PATH) as f:
        return yaml.safe_load(f)


def _sample(aneurysm_mm: float, vessel_mm: float, heparin_uM: float = 2.0) -> dict:
    return {
        "aneurysm_diameter_mm": aneurysm_mm,
        "vessel_diameter_mm": vessel_mm,
        "inlet_velocity_cm_s": 47.0,
        "platelet_conc_plt_ml": 3.5e8,
        "heparin_conc_uM": heparin_uM,
        "prothrombin_uM": 1.1,
        "antithrombin_uM": 2.844,
        "fibrinogen_uM": 7.0,
    }


def _write_split(tmp_path, physio, split_name, samples):
    split_dir = tmp_path / split_name
    split_dir.mkdir()
    mesh_cfg = {"target_num_elements": 150}
    for i, sample in enumerate(samples):
        result = _run_one_sample(sample, physio, mesh_cfg, end_time_s=0.2, dt_s=0.1, grid_size=(8, 8))
        np.savez(split_dir / f"sample_{i:04d}.npz", **result)


def test_evaluate_edge_holdout_degradation_continuous_end_to_end(physio, tmp_path):
    _write_split(tmp_path, physio, "train", [_sample(7.0, 3.2), _sample(8.0, 3.4)])
    _write_split(tmp_path, physio, "test", [_sample(7.5, 3.3)])
    _write_split(tmp_path, physio, "edge_holdout", [_sample(9.9, 3.9)])

    train_ds = PointCloudThrombusDataset(str(tmp_path), "train")
    test_ds = PointCloudThrombusDataset(str(tmp_path), "test")
    edge_holdout_ds = PointCloudThrombusDataset(str(tmp_path), "edge_holdout")

    model = ContinuousMeanFieldBaseline(n_neighbors=5).fit(train_ds)
    result = evaluate_edge_holdout_degradation_continuous(model, test_ds, edge_holdout_ds)

    assert set(result.keys()) == {"test", "edge_holdout", "degradation_ratio"}
    for key in ("test", "edge_holdout"):
        assert result[key]["overall"] >= 0.0
        assert np.isfinite(result[key]["overall"])
    assert result["degradation_ratio"] >= 0.0


def test_evaluate_extrapolation_degradation_continuous_end_to_end(physio, tmp_path):
    _write_split(tmp_path, physio, "train", [_sample(7.0, 3.2, heparin_uM=0.2), _sample(8.0, 3.4, heparin_uM=0.3)])
    _write_split(tmp_path, physio, "test", [_sample(7.5, 3.3, heparin_uM=0.25)])
    _write_split(tmp_path, physio, "extrapolation", [_sample(7.8, 3.3, heparin_uM=0.45)])

    train_ds = PointCloudThrombusDataset(str(tmp_path), "train")
    test_ds = PointCloudThrombusDataset(str(tmp_path), "test")
    extrapolation_ds = PointCloudThrombusDataset(str(tmp_path), "extrapolation")

    model = ContinuousMeanFieldBaseline(n_neighbors=5).fit(train_ds)
    result = evaluate_extrapolation_degradation_continuous(
        model, test_ds, extrapolation_ds,
        extrapolate_param="heparin_conc_uM", train_range=(0.1, 0.38), extrapolate_range=(0.38, 0.5),
    )

    assert result["label"] == "heparin_conc_uM extrapolation (trained on 0.1-0.38, tested on 0.38-0.5)"
    assert set(result.keys()) == {"label", "test", "extrapolation", "degradation_ratio"}
    for key in ("test", "extrapolation"):
        assert result[key]["overall"] >= 0.0
        assert np.isfinite(result[key]["overall"])
    assert result["degradation_ratio"] >= 0.0
