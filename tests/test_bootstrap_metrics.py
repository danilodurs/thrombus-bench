"""Tests for `benchmark.metrics.bootstrap_metric_by_sample` (Phase 6, `docs/
continuous_surrogate_design.md`): confirms the resampling unit is a whole
`sample_id` row, not an individual query point or checkpoint -- see that
module's "Bootstrap resampling unit" docstring section. No bootstrap/CI
code existed anywhere in this project before this (audited via direct
search, not assumed), so this test is the first concrete guardrail against
ever getting the resampling unit wrong."""

from __future__ import annotations

import numpy as np
import pytest

from thrombus_bench.benchmark.metrics import bootstrap_metric_by_sample


def test_bootstrap_resamples_whole_sample_rows_not_individual_checkpoints():
    """Each row represents one sample's per-checkpoint values, deliberately
    constructed so every entry within a row shares the same value (a
    "sample_id signature"). If the implementation ever resampled at a
    finer grain than whole rows (e.g. flattening and resampling individual
    checkpoint entries independently), a resampled row could mix entries
    from different original samples, breaking that internal-consistency
    property -- this test would catch that immediately via the assertion
    inside metric_fn, not just check "it runs.\""""

    n_samples = 6
    checkpoints_per_sample = 4
    per_sample_values = np.stack([np.full(checkpoints_per_sample, float(i)) for i in range(n_samples)])

    calls = {"count": 0}

    def metric_fn(values: np.ndarray) -> float:
        calls["count"] += 1
        for row in values:
            assert len(set(row.tolist())) == 1, (
                f"resampled row mixes entries from different original samples: {row} "
                "-- the resampling unit must be a whole sample_id row, not individual checkpoints"
            )
        return float(values.mean())

    rng = np.random.default_rng(0)
    result = bootstrap_metric_by_sample(metric_fn, per_sample_values, n_bootstrap=200, rng=rng)

    assert calls["count"] == 201  # 1 point estimate + 200 bootstrap resamples
    assert np.isfinite(result["point_estimate"])


def test_bootstrap_resampled_rows_are_always_exact_original_rows():
    """A 1D variant of the same check: every value that ever appears in a
    bootstrap resample must be one of the original per-sample values
    exactly -- the bootstrap must never fabricate a blended/interpolated
    value between two different samples."""

    n_samples = 8
    per_sample_values = np.arange(n_samples, dtype=float) * 10.0  # [0, 10, 20, ..., 70]
    original_values = set(per_sample_values.tolist())

    def metric_fn(values: np.ndarray) -> float:
        seen = set(values.tolist())
        assert seen <= original_values, f"bootstrap introduced value(s) not in the original samples: {seen - original_values}"
        return float(values.mean())

    rng = np.random.default_rng(1)
    bootstrap_metric_by_sample(metric_fn, per_sample_values, n_bootstrap=300, rng=rng)


def test_bootstrap_confidence_interval_contains_point_estimate_and_is_sane():
    rng_data = np.random.default_rng(0)
    per_sample_values = rng_data.normal(loc=5.0, scale=1.0, size=40)

    result = bootstrap_metric_by_sample(
        np.mean, per_sample_values, n_bootstrap=500, confidence=0.95, rng=np.random.default_rng(1)
    )

    assert result["point_estimate"] == pytest.approx(float(np.mean(per_sample_values)))
    assert result["lower"] <= result["point_estimate"] <= result["upper"]
    assert result["confidence"] == 0.95
    assert result["n_bootstrap"] == 500


def test_bootstrap_with_replacement_can_repeat_a_sample_and_omit_another():
    """With-replacement resampling of n_samples rows from n_samples
    originals must be able to produce duplicates (and, correspondingly,
    omissions) -- a sanity check that this is genuine bootstrap resampling,
    not e.g. an accidental fixed permutation (which would always include
    every sample exactly once)."""

    n_samples = 5
    per_sample_values = np.arange(n_samples, dtype=float)

    saw_a_duplicate = False

    def metric_fn(values: np.ndarray) -> float:
        nonlocal saw_a_duplicate
        if len(set(values.tolist())) < len(values):
            saw_a_duplicate = True
        return float(values.mean())

    rng = np.random.default_rng(2)
    bootstrap_metric_by_sample(metric_fn, per_sample_values, n_bootstrap=100, rng=rng)

    assert saw_a_duplicate, "with n_bootstrap=100 draws, at least one resample should contain a duplicated sample"
