"""Tests for the point-query metrics added in Phase 6 (`docs/
continuous_surrogate_design.md`): `field_rmse_pointwise`,
`field_rmse_by_checkpoint`, `field_rmse_by_distance_to_wall`
(`benchmark/metrics.py`). Hand-verified expected values throughout, same
standard as the existing `field_rmse` tests."""

from __future__ import annotations

import numpy as np
import pytest

from thrombus_bench.benchmark.metrics import (
    field_rmse_by_checkpoint,
    field_rmse_by_distance_to_wall,
    field_rmse_pointwise,
)


def test_field_rmse_pointwise_matches_hand_computed_value():
    # 3 points, 2 channels. Errors: [1,2] , [0,0] , [3,4].
    pred = np.array([[1.0, 2.0], [5.0, 5.0], [3.0, 4.0]])
    true = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 0.0]])

    result = field_rmse_pointwise(pred, true)

    # per-channel: channel 0 errors [1,0,3] -> sqrt(mean([1,0,9])) = sqrt(10/3)
    # channel 1 errors [2,0,4] -> sqrt(mean([4,0,16])) = sqrt(20/3)
    expected_ch0 = np.sqrt((1.0 + 0.0 + 9.0) / 3.0)
    expected_ch1 = np.sqrt((4.0 + 0.0 + 16.0) / 3.0)
    np.testing.assert_allclose(result["per_channel"], [expected_ch0, expected_ch1], rtol=1e-6)

    # overall: sqrt(mean of all 6 squared errors) = sqrt((1+4+0+0+9+16)/6)
    expected_overall = np.sqrt((1.0 + 4.0 + 0.0 + 0.0 + 9.0 + 16.0) / 6.0)
    assert result["overall"] == pytest.approx(expected_overall, rel=1e-6)


def test_field_rmse_by_checkpoint_separates_groups_correctly():
    # checkpoint 0: constant error 1.0 per point (2 points, 1 channel).
    # checkpoint 1: constant error 3.0 per point (1 point, 1 channel).
    pred = np.array([[1.0], [1.0], [3.0]])
    true = np.array([[0.0], [0.0], [0.0]])
    checkpoint_id = np.array([0, 0, 1])

    result = field_rmse_by_checkpoint(pred, true, checkpoint_id)

    assert set(result.keys()) == {0, 1}
    assert result[0]["overall"] == pytest.approx(1.0)
    assert result[1]["overall"] == pytest.approx(3.0)


def test_field_rmse_by_distance_to_wall_bins_and_computes_rmse_per_bin():
    # 4 points, 1 channel, with known distances and known errors.
    pred = np.array([[1.0], [2.0], [10.0], [0.0]])
    true = np.array([[0.0], [0.0], [0.0], [0.0]])
    # errors: 1, 2, 10, 0
    sdf_values = np.array([0.0009, -0.0009, 0.0031, 0.0011])  # |dist|: 0.0009, 0.0009, 0.0031, 0.0011

    bin_edges = np.array([0.0, 0.001, 0.002, 0.004])  # 3 bins: [0,0.001), [0.001,0.002), [0.002,0.004)

    result = field_rmse_by_distance_to_wall(pred, true, sdf_values, bin_edges=bin_edges)

    np.testing.assert_allclose(result["bin_edges"], bin_edges)
    # bin 0 ([0, 0.001)): points 0,1 (|dist|=0.0009 each) -> errors 1,2 -> sqrt(mean([1,4])) = sqrt(2.5)
    assert result["n_points_per_bin"][0] == 2
    assert result["rmse_per_bin"][0] == pytest.approx(np.sqrt(2.5), rel=1e-6)
    # bin 1 ([0.001, 0.002)): point 3 (|dist|=0.0011) -> error 0 -> rmse 0
    assert result["n_points_per_bin"][1] == 1
    assert result["rmse_per_bin"][1] == pytest.approx(0.0, abs=1e-9)
    # bin 2 ([0.002, 0.004)): point 2 (|dist|=0.0031) -> error 10 -> rmse 10
    assert result["n_points_per_bin"][2] == 1
    assert result["rmse_per_bin"][2] == pytest.approx(10.0, rel=1e-6)


def test_field_rmse_by_distance_to_wall_empty_bin_is_nan():
    pred = np.array([[1.0], [1.0]])
    true = np.array([[0.0], [0.0]])
    sdf_values = np.array([0.0001, 0.0002])  # both near zero, nothing in the far bin
    bin_edges = np.array([0.0, 0.001, 1.0])

    result = field_rmse_by_distance_to_wall(pred, true, sdf_values, bin_edges=bin_edges)

    assert result["n_points_per_bin"][1] == 0
    assert np.isnan(result["rmse_per_bin"][1])


def test_field_rmse_by_distance_to_wall_default_bin_edges_span_the_data():
    pred = np.zeros((5, 1))
    true = np.zeros((5, 1))
    sdf_values = np.array([0.0, 0.001, 0.002, 0.003, 0.004])

    result = field_rmse_by_distance_to_wall(pred, true, sdf_values)

    assert len(result["bin_edges"]) == 6  # default: 5 bins
    assert result["bin_edges"][0] == pytest.approx(0.0)
    assert result["bin_edges"][-1] == pytest.approx(0.004)
    assert result["n_points_per_bin"].sum() == 5
