"""Tests for benchmark metric functions on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from thrombus_bench.benchmark.metrics import (
    field_rmse,
    max_M_at_relative_error,
    runtime_comparison,
    thrombosed_fraction_error,
    thrombus_iou,
    thrombus_mask,
)


def test_thrombus_mask_thresholds():
    M_at = np.array([0.0, 1.9e7, 2.0e7, 3.0e7])
    FI = np.array([0.0, 0.0, 0.0, 0.0])
    mask = thrombus_mask(M_at, FI, M_at_critical=2.0e7, fibrin_critical=0.6)
    assert list(mask) == [False, False, True, True]


def test_thrombus_mask_fibrin_threshold():
    M_at = np.array([0.0, 0.0])
    FI = np.array([0.5, 0.7])
    mask = thrombus_mask(M_at, FI, M_at_critical=2.0e7, fibrin_critical=0.6)
    assert list(mask) == [False, True]


def test_thrombus_iou_identical_masks():
    mask = np.array([True, True, False, False, True])
    assert thrombus_iou(mask, mask) == pytest.approx(1.0)


def test_thrombus_iou_disjoint_masks():
    a = np.array([True, True, False, False])
    b = np.array([False, False, True, True])
    assert thrombus_iou(a, b) == pytest.approx(0.0)


def test_thrombus_iou_partial_overlap():
    a = np.array([True, True, False, False])
    b = np.array([True, False, False, True])
    # intersection = 1 (index 0), union = 3 (indices 0,1,3)
    assert thrombus_iou(a, b) == pytest.approx(1.0 / 3.0)


def test_thrombus_iou_empty_union_returns_one():
    a = np.array([False, False])
    b = np.array([False, False])
    assert thrombus_iou(a, b) == pytest.approx(1.0)


def test_field_rmse_identical_is_zero():
    fields = np.random.default_rng(0).normal(size=(2, 3, 4, 4))
    result = field_rmse(fields, fields)
    assert result["overall"] == pytest.approx(0.0)
    assert np.allclose(result["per_channel"], 0.0)


def test_field_rmse_known_constant_offset():
    true = np.zeros((3, 4, 4))
    pred = np.full((3, 4, 4), 2.0)
    result = field_rmse(pred, true)
    assert result["overall"] == pytest.approx(2.0)
    assert np.allclose(result["per_channel"], 2.0)


def test_field_rmse_without_mask_leaves_fluid_only_none():
    """Default mask=None must keep existing behavior identical -- no
    "fluid_only" computation, just the two pre-existing keys populated."""

    fields = np.random.default_rng(0).normal(size=(2, 3, 4, 4))
    result = field_rmse(fields, fields)
    assert result["fluid_only"] is None
    assert result["per_channel_fluid_only"] is None


def test_field_rmse_masked_ignores_exterior_error_4d():
    """Construct a batch where the exterior (mask=0) region has a huge
    error and the fluid (mask=1) region has none -- the fluid-only RMSE
    must be exactly 0, while the all-cells RMSE is dominated by the
    exterior error."""

    true = np.zeros((2, 3, 4, 4))
    pred = np.zeros((2, 3, 4, 4))
    mask = np.ones((2, 4, 4))
    mask[:, 0, 0] = 0.0  # one exterior cell per sample
    pred[:, :, 0, 0] = 1000.0  # huge error, but only in the exterior cell

    result = field_rmse(pred, true, mask=mask)

    assert result["fluid_only"] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(result["per_channel_fluid_only"], 0.0, atol=1e-9)
    assert result["overall"] > 100.0  # dominated by the exterior error


def test_field_rmse_masked_matches_hand_computation_3d():
    true = np.zeros((2, 2, 2))
    pred = np.array(
        [
            [[1.0, 2.0], [0.0, 0.0]],  # channel 0: errors 1,2 at fluid cells; 0 at exterior
            [[0.0, 0.0], [3.0, 100.0]],  # channel 1: exterior cell (row1,col1) has a huge error
        ]
    )
    mask = np.array([[1.0, 1.0], [1.0, 0.0]])  # bottom-right cell is exterior

    result = field_rmse(pred, true, mask=mask)

    # Fluid cells (3 of them, shared across channels): channel 0 errors
    # [1,2,0]^2 -> mean 5/3; channel 1 errors [0,0,3]^2 -> mean 3.
    expected_per_channel = np.sqrt([5.0 / 3.0, 3.0])
    np.testing.assert_allclose(result["per_channel_fluid_only"], expected_per_channel, rtol=1e-6)

    expected_overall = np.sqrt((1.0 + 4.0 + 0.0 + 0.0 + 0.0 + 9.0) / 6.0)
    assert result["fluid_only"] == pytest.approx(expected_overall, rel=1e-6)
    assert result["overall"] > result["fluid_only"]  # the 100.0 exterior error inflates it


def test_max_M_at_relative_error():
    pred = np.array([1.1e7, 2.0e7])
    true = np.array([1.0e7, 2.0e7])
    err = max_M_at_relative_error(pred, true)
    assert err[0] == pytest.approx(0.1, rel=1e-6)
    assert err[1] == pytest.approx(0.0, abs=1e-9)


def test_thrombosed_fraction_error():
    err = thrombosed_fraction_error(np.array([0.3, 0.9]), np.array([0.2, 0.9]))
    assert err[0] == pytest.approx(0.1)
    assert err[1] == pytest.approx(0.0)


def test_runtime_comparison_speedup():
    result = runtime_comparison(mechanistic_times_s=np.array([10.0, 12.0]), neural_times_s=np.array([0.1, 0.1]))
    assert result["mechanistic_mean_s"] == pytest.approx(11.0)
    assert result["neural_mean_s"] == pytest.approx(0.1)
    assert result["speedup_factor"] == pytest.approx(110.0)
