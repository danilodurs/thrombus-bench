"""Tests for `data.dataset.PointCloudThrombusDataset`/`pointcloud_collate_fn`
(Phase 3/4, `docs/continuous_surrogate_design.md`): round-tripping
`generate_dataset.py`'s point-cloud `.npz` schema through the dataset,
ragged-batch collation, and the `n_snapshots=1` regression guard.

`fields`/`M_at_target` are log-compressed (`field_to_log`, added Phase 4
after an end-to-end `train_continuous` run showed a completely flat loss
without it -- same reasoning as `ThrombusSurrogateDataset.fields`), so
"matches the mechanistic solver's raw output" tests below compare against
`field_to_log(reference)`, not the raw reference directly."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from thrombus_bench.data.dataset import FIELD_NAMES, PointCloudThrombusDataset, field_to_log, pointcloud_collate_fn
from thrombus_bench.data.generate_dataset import PARAM_ORDER, _run_one_sample
from thrombus_bench.data.sampler import ParameterSpace, normalize_params
from thrombus_bench.mechanistic.coupled_solver import run_coupled_simulation
from thrombus_bench.mechanistic.mesh import GeometryConfig, build_aneurysm_mesh

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


def _write_dataset(tmp_path, physio, n_snapshots, end_time_s=0.3, dt_s=0.1):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    mesh_cfg = {"target_num_elements": 150}
    result = _run_one_sample(
        _small_sample(), physio, mesh_cfg, end_time_s=end_time_s, dt_s=dt_s, grid_size=(8, 8), n_snapshots=n_snapshots
    )
    np.savez(split_dir / "sample_0000.npz", **result)
    return result


def test_roundtrip_matches_mechanistic_solver_raw_output_no_interpolation(physio, tmp_path):
    """No `_rasterize` involved -- node coordinates/values loaded back
    through the dataset should match a fresh, independent mechanistic run
    with identical inputs (mesh generation is deterministic given the same
    GeometryConfig/mesh_cfg -- see mesh.py's fixed internal RNG seed)."""

    _write_dataset(tmp_path, physio, n_snapshots=1)

    dataset = PointCloudThrombusDataset(str(tmp_path), "train")
    item = dataset[0]

    sample = _small_sample()
    geom = GeometryConfig(
        vessel_diameter_mm=sample["vessel_diameter_mm"],
        aneurysm_diameter_mm=sample["aneurysm_diameter_mm"],
        vessel_length_mm=50.0,
    )
    tagged_mesh = build_aneurysm_mesh(geom, {"target_num_elements": 150})
    physio_run = {k: (dict(v) if isinstance(v, dict) else v) for k, v in physio.items()}
    physio_run["species"] = dict(physio["species"])
    physio_run["species"]["resting_platelets_inlet_plt_ml"] = sample["platelet_conc_plt_ml"]
    physio_run["species"]["prothrombin_inlet_uM"] = sample["prothrombin_uM"]
    physio_run["species"]["antithrombin_inlet_uM"] = sample["antithrombin_uM"]
    physio_run["species"]["fibrinogen_inlet_uM"] = sample["fibrinogen_uM"]
    physio_run["heparin"] = dict(physio["heparin"])
    physio_run["heparin"]["concentration_uM"] = sample["heparin_conc_uM"]
    history_ref = run_coupled_simulation(
        tagged_mesh, inlet_velocity_m_s=sample["inlet_velocity_cm_s"] / 100.0, physio=physio_run,
        end_time_s=0.3, dt_s=0.1, output_every_n_steps=3, flow_resolve_every_n_steps=3,
    )
    final_ref = history_ref.states[-1]
    node_coords_ref = tagged_mesh.mesh.p.T
    n_vertices = node_coords_ref.shape[0]

    np.testing.assert_allclose(item["node_coords"].numpy(), node_coords_ref, rtol=1e-6, atol=1e-12)

    # item["fields"] is log-compressed (field_to_log), same convention as
    # ThrombusSurrogateDataset -- compare against the log-compressed
    # reference, not the raw mechanistic output directly.
    ux_ref = final_ref.flow.u[0 : 2 * n_vertices : 2]
    velocity_x_idx = FIELD_NAMES.index("velocity_x")
    np.testing.assert_allclose(item["fields"][:, velocity_x_idx].numpy(), field_to_log(ux_ref), rtol=1e-5, atol=1e-8)

    rp_idx = FIELD_NAMES.index("conc_RP")
    np.testing.assert_allclose(
        item["fields"][:, rp_idx].numpy(),
        field_to_log(final_ref.concentrations["RP"][:n_vertices]),
        rtol=1e-5,
        atol=1e-8,
    )


def test_params_with_time_normalization_matches_sampler_convention(physio, tmp_path):
    result = _write_dataset(tmp_path, physio, n_snapshots=1)
    dataset = PointCloudThrombusDataset(str(tmp_path), "train")
    item = dataset[0]

    expected_params = normalize_params(result["params"], ParameterSpace()).astype(np.float32)
    np.testing.assert_allclose(item["params_with_time"][:8].numpy(), expected_params, rtol=1e-5)
    # n_snapshots=1: the one checkpoint IS the final time -> t_norm == 1.0 exactly.
    assert item["params_with_time"][8].item() == pytest.approx(1.0)

    assert item["geometry_mm"].shape == (2,)
    aneurysm_idx, vessel_idx = PARAM_ORDER.index("aneurysm_diameter_mm"), PARAM_ORDER.index("vessel_diameter_mm")
    assert item["geometry_mm"][0].item() == pytest.approx(result["params"][aneurysm_idx])
    assert item["geometry_mm"][1].item() == pytest.approx(result["params"][vessel_idx])


def test_n_snapshots_one_needs_no_special_casing(physio, tmp_path):
    """A single-checkpoint sample should behave like any other through the
    dataset -- one item, valid t_norm, no separate code path required."""

    _write_dataset(tmp_path, physio, n_snapshots=1)
    dataset = PointCloudThrombusDataset(str(tmp_path), "train")
    assert len(dataset) == 1
    item = dataset[0]
    assert item["fields"].shape[0] == item["node_coords"].shape[0]
    assert torch.all(torch.isfinite(item["params_with_time"]))


def test_multiple_checkpoints_produce_one_dataset_item_each(physio, tmp_path):
    result = _write_dataset(tmp_path, physio, n_snapshots=3, end_time_s=0.3, dt_s=0.1)
    n_snapshots = result["time_s"].shape[0]
    assert n_snapshots > 1, "test setup should actually exercise multiple checkpoints"

    dataset = PointCloudThrombusDataset(str(tmp_path), "train")
    assert len(dataset) == n_snapshots

    t_norms = [dataset[i]["params_with_time"][8].item() for i in range(len(dataset))]
    assert t_norms == sorted(t_norms), "checkpoints should be indexed in time order"
    assert t_norms[-1] == pytest.approx(1.0)
    assert all(t > -1.0 for t in t_norms), "no true t=0 checkpoint exists (see generate_dataset docstring)"

    for i in range(len(dataset)):
        item = dataset[i]
        assert torch.equal(item["thrombin_fibrin_reliable"], item["thrombin_fibrin_reliable"])  # present, bool
        assert item["thrombin_fibrin_reliable"].dtype == torch.bool


def test_points_per_sample_subsamples_and_redraws_each_call(physio, tmp_path):
    _write_dataset(tmp_path, physio, n_snapshots=1)
    dataset = PointCloudThrombusDataset(str(tmp_path), "train", points_per_sample=5)

    item_a = dataset[0]
    item_b = dataset[0]
    assert item_a["node_coords"].shape == (5, 2)
    assert item_a["fields"].shape == (5, len(FIELD_NAMES))
    # Extremely unlikely to coincide by chance across two independent draws
    # from a mesh with far more than 5 nodes -- confirms re-drawing, not a
    # fixed subsample cached at construction time.
    assert not torch.equal(item_a["node_coords"], item_b["node_coords"])


def test_m_at_target_is_zero_off_wall_and_matches_wall_values_on_wall(physio, tmp_path):
    result = _write_dataset(tmp_path, physio, n_snapshots=1)
    dataset = PointCloudThrombusDataset(str(tmp_path), "train")
    item = dataset[0]

    wall_dofs = result["wall_dofs"]
    is_wall_expected = np.zeros(item["node_coords"].shape[0], dtype=bool)
    is_wall_expected[wall_dofs] = True

    np.testing.assert_array_equal(item["is_wall"].numpy(), is_wall_expected)
    assert torch.all(item["M_at_target"][~item["is_wall"]] == 0.0)
    # M_at_target is log-compressed (field_to_log(0) == 0, so the off-wall
    # check above is unaffected).
    np.testing.assert_allclose(
        item["M_at_target"][item["is_wall"]].numpy(),
        field_to_log(result["M_at_wall_values"][0]),
        rtol=1e-5,
        atol=1e-8,
    )
    # Wall nodes' coordinates should exactly match wall_node_coords (same
    # underlying points, not a separately-interpolated set).
    np.testing.assert_allclose(
        item["node_coords"][item["is_wall"]].numpy(), result["wall_node_coords"], rtol=1e-6, atol=1e-9
    )


def test_collate_fn_produces_correctly_shaped_ragged_batch():
    def fake_item(n_points: int, tag: float) -> dict:
        return {
            "params_with_time": torch.full((9,), tag),
            "geometry_mm": torch.full((2,), tag),
            "thrombin_fibrin_reliable": torch.tensor(tag > 0),
            "node_coords": torch.full((n_points, 2), tag),
            "fields": torch.full((n_points, len(FIELD_NAMES)), tag),
            "M_at_target": torch.full((n_points,), tag),
            "is_wall": torch.zeros(n_points, dtype=torch.bool),
        }

    counts = [5, 1, 12]
    batch = [fake_item(n, float(i)) for i, n in enumerate(counts)]
    collated = pointcloud_collate_fn(batch)

    total = sum(counts)
    assert collated["params_with_time"].shape == (3, 9)
    assert collated["geometry_mm"].shape == (3, 2)
    assert collated["thrombin_fibrin_reliable"].shape == (3,)
    assert collated["node_coords"].shape == (total, 2)
    assert collated["fields"].shape == (total, len(FIELD_NAMES))
    assert collated["M_at_target"].shape == (total,)
    assert collated["is_wall"].shape == (total,)
    assert collated["is_wall"].dtype == torch.bool
    assert collated["batch_index"].shape == (total,)
    assert collated["batch_index"].dtype == torch.long

    for i, n in enumerate(counts):
        mask = collated["batch_index"] == i
        assert int(mask.sum()) == n
        assert torch.all(collated["node_coords"][mask] == float(i))
        assert torch.all(collated["fields"][mask] == float(i))
        assert torch.all(collated["M_at_target"][mask] == float(i))
