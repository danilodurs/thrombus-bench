"""Tests for `generate_dataset._build_pointcloud_sample` -- the standalone
point-cloud `.npz` schema conversion function (design + schema only, not
yet wired into `_run_one_sample`/`generate_dataset`), see that function's
docstring and `docs/continuous_surrogate_design.md` Phase 1."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from thrombus_bench.data.dataset import FIELD_NAMES
from thrombus_bench.data.generate_dataset import _ALL_SPECIES, _build_pointcloud_sample
from thrombus_bench.mechanistic.coupled_solver import run_coupled_simulation
from thrombus_bench.mechanistic.mesh import build_channel_mesh

PHYSIO_PATH = "configs/physio_params.yaml"


@pytest.fixture
def physio():
    with open(PHYSIO_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def small_history(physio):
    tm = build_channel_mesh(50.0, 4.0, target_num_elements=150)
    history = run_coupled_simulation(
        tm, inlet_velocity_m_s=0.47, physio=physio,
        end_time_s=0.3, dt_s=0.1, output_every_n_steps=1, flow_resolve_every_n_steps=3,
    )
    return tm, history


def test_schema_shapes_and_keys(small_history, physio):
    tm, history = small_history
    params = np.arange(8, dtype=np.float64)
    m_at_critical = physio["adhesion_aggregation"]["M_at_critical_plt_cm2"]

    result = _build_pointcloud_sample(tm, history, params, m_at_critical)

    n_nodes = tm.mesh.p.shape[1]
    n_triangles = tm.mesh.t.shape[1]
    n_snapshots = len(history.states)
    n_wall = len(history.states[0].wall_dofs)

    assert result["node_coords"].shape == (n_nodes, 2)
    assert result["triangles"].shape == (n_triangles, 3)
    assert result["fields"].shape == (n_snapshots, n_nodes, len(FIELD_NAMES))
    assert result["wall_node_coords"].shape == (n_wall, 2)
    assert result["M_at_wall_values"].shape == (n_snapshots, n_wall)
    assert result["time_s"].shape == (n_snapshots,)
    assert result["thrombin_fibrin_reliable_at_checkpoint"].shape == (n_snapshots,)
    assert result["thrombin_fibrin_reliable_at_checkpoint"].dtype == bool
    assert np.array_equal(result["params"], params)

    for key in ("converged", "flow_n_iterations", "flow_residual", "thrombin_fibrin_reliable", "max_M_at", "thrombosed_fraction"):
        assert key in result
    for name in _ALL_SPECIES:
        assert f"clip_count_{name}" in result
        assert f"conc_{name}_min" in result
        assert f"conc_{name}_max" in result


def test_fields_channel_values_match_history_states(small_history, physio):
    tm, history = small_history
    params = np.zeros(8, dtype=np.float64)
    m_at_critical = physio["adhesion_aggregation"]["M_at_critical_plt_cm2"]

    result = _build_pointcloud_sample(tm, history, params, m_at_critical)
    n_vertices = tm.mesh.p.shape[1]

    for i, state in enumerate(history.states):
        expected_ux = state.flow.u[0 : 2 * n_vertices : 2]
        expected_uy = state.flow.u[1 : 2 * n_vertices : 2]
        np.testing.assert_allclose(result["fields"][i, :, 0], expected_ux, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(result["fields"][i, :, 1], expected_uy, rtol=1e-5, atol=1e-8)

        rp_channel = FIELD_NAMES.index("conc_RP")
        np.testing.assert_allclose(
            result["fields"][i, :, rp_channel], state.concentrations["RP"][:n_vertices], rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(result["M_at_wall_values"][i], state.surface.M_at, rtol=1e-5, atol=1e-8)
        assert result["time_s"][i] == pytest.approx(state.time_s)


def test_thrombin_fibrin_reliable_at_checkpoint_matches_history(small_history, physio):
    tm, history = small_history
    params = np.zeros(8, dtype=np.float64)
    m_at_critical = physio["adhesion_aggregation"]["M_at_critical_plt_cm2"]

    result = _build_pointcloud_sample(tm, history, params, m_at_critical)
    np.testing.assert_array_equal(
        result["thrombin_fibrin_reliable_at_checkpoint"], history.thrombin_fibrin_reliable_at_checkpoint
    )


def test_roundtrips_through_npz(small_history, physio, tmp_path):
    tm, history = small_history
    params = np.arange(8, dtype=np.float64)
    m_at_critical = physio["adhesion_aggregation"]["M_at_critical_plt_cm2"]
    result = _build_pointcloud_sample(tm, history, params, m_at_critical)

    path = tmp_path / "sample_0000.npz"
    np.savez(path, **result)
    loaded = np.load(path)

    assert loaded["fields"].shape == result["fields"].shape
    np.testing.assert_allclose(loaded["fields"], result["fields"])
    np.testing.assert_array_equal(
        loaded["thrombin_fibrin_reliable_at_checkpoint"], result["thrombin_fibrin_reliable_at_checkpoint"]
    )
    assert bool(loaded["thrombin_fibrin_reliable"]) == result["thrombin_fibrin_reliable"]
