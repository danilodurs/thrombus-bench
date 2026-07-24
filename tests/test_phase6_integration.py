"""End-to-end integration check for Phase 6's evaluation layer (`docs/
continuous_surrogate_design.md`): real generated data -> PointCloudThrombusDataset
-> continuous baselines fit/evaluated with the point-query metrics ->
distance-to-wall breakdown, all wired together, not just each piece tested
in synthetic isolation."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from thrombus_bench.benchmark.metrics import field_rmse_by_distance_to_wall, field_rmse_pointwise
from thrombus_bench.data.dataset import PointCloudThrombusDataset
from thrombus_bench.data.generate_dataset import PARAM_ORDER, _run_one_sample
from thrombus_bench.mechanistic.geometry_sdf import signed_distance_to_wall
from thrombus_bench.mechanistic.mesh import GeometryConfig
from thrombus_bench.neural.baselines import ContinuousMeanFieldBaseline, ContinuousNearestNeighborBaseline

PHYSIO_PATH = "configs/physio_params.yaml"


@pytest.fixture
def physio():
    with open(PHYSIO_PATH) as f:
        return yaml.safe_load(f)


def _sample(aneurysm_mm: float, vessel_mm: float) -> dict:
    return {
        "aneurysm_diameter_mm": aneurysm_mm,
        "vessel_diameter_mm": vessel_mm,
        "inlet_velocity_cm_s": 47.0,
        "platelet_conc_plt_ml": 3.5e8,
        "heparin_conc_uM": 2.0,
        "prothrombin_uM": 1.1,
        "antithrombin_uM": 2.844,
        "fibrinogen_uM": 7.0,
    }


def test_continuous_baselines_evaluated_with_pointwise_and_distance_to_wall_metrics(physio, tmp_path):
    mesh_cfg = {"target_num_elements": 150}
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    # Two slightly different training samples, one held-out test sample.
    for i, (aneurysm_mm, vessel_mm) in enumerate([(7.0, 3.2), (8.0, 3.4)]):
        result = _run_one_sample(
            _sample(aneurysm_mm, vessel_mm), physio, mesh_cfg, end_time_s=0.2, dt_s=0.1, grid_size=(8, 8)
        )
        np.savez(train_dir / f"sample_{i:04d}.npz", **result)

    test_result = _run_one_sample(
        _sample(7.5, 3.3), physio, mesh_cfg, end_time_s=0.2, dt_s=0.1, grid_size=(8, 8)
    )
    np.savez(test_dir / "sample_0000.npz", **test_result)

    train_ds = PointCloudThrombusDataset(str(tmp_path), "train")
    test_ds = PointCloudThrombusDataset(str(tmp_path), "test")

    mean_field = ContinuousMeanFieldBaseline(n_neighbors=10).fit(train_ds)
    nearest_neighbor = ContinuousNearestNeighborBaseline().fit(train_ds)

    test_item = test_ds[0]
    batch_index = test_item["node_coords"].new_zeros(test_item["node_coords"].shape[0]).long()
    geometry_batch = test_item["geometry_mm"].unsqueeze(0)
    params_batch = test_item["params_with_time"].unsqueeze(0)

    for model in (mean_field, nearest_neighbor):
        pred = model(params_batch, test_item["node_coords"], batch_index, geometry_batch)
        assert pred.shape == test_item["fields"].shape
        assert np.all(np.isfinite(pred.numpy()))

        result = field_rmse_pointwise(pred.numpy(), test_item["fields"].numpy())
        assert result["overall"] >= 0.0
        assert result["per_channel"].shape == (test_item["fields"].shape[1],)

    # Distance-to-wall breakdown for the mean-field baseline's predictions,
    # using Phase 1's SDF against the test sample's own geometry -- the
    # genuinely new capability this phase adds.
    aneurysm_mm, vessel_mm = float(test_item["geometry_mm"][0]), float(test_item["geometry_mm"][1])
    geom = GeometryConfig(vessel_diameter_mm=vessel_mm, aneurysm_diameter_mm=aneurysm_mm, vessel_length_mm=50.0)
    node_coords_np = test_item["node_coords"].numpy()
    sdf_values = signed_distance_to_wall(node_coords_np[:, 0], node_coords_np[:, 1], geom)
    # Mesh nodes should all be inside/on the fluid domain -- boundary nodes
    # sit at SDF~0 exactly in analytic terms, so allow a tiny floating-point
    # tolerance rather than requiring exact non-negativity (see Phase 1's
    # own mesh-boundary cross-check tests for the same tolerance pattern).
    assert np.all(sdf_values >= -1e-8)

    pred = mean_field(params_batch, test_item["node_coords"], batch_index, geometry_batch)
    by_distance = field_rmse_by_distance_to_wall(pred.numpy(), test_item["fields"].numpy(), sdf_values)

    assert by_distance["n_points_per_bin"].sum() == node_coords_np.shape[0]
    # At least one bin should have data (near-wall nodes exist, since
    # wall_dofs are a subset of node_coords with sdf ~ 0).
    assert by_distance["n_points_per_bin"][0] > 0
