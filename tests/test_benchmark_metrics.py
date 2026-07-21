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
