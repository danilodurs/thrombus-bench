"""Tests for benchmark metric functions on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from thrombus_bench.benchmark.metrics import thrombus_iou, thrombus_mask


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
