"""Tests for data/sampler.py's sample_with_extrapolation_holdout, the
genuinely-extrapolative counterpart to split_train_val_test_edge_holdout
(Task 4.1): train/val/test draw one parameter from a restricted
sub-range, while the "extrapolation" split draws that same parameter from
the withheld remainder -- unlike the edge-of-domain holdout, which is
still drawn from the same sampled box."""

from __future__ import annotations

import pytest

from thrombus_bench.data.sampler import ParameterSpace, sample_with_extrapolation_holdout


def test_train_val_test_and_extrapolation_have_no_heparin_range_overlap():
    space = ParameterSpace()
    train_range = (0.1, 0.38)
    extrapolate_range = (0.38, 0.5)

    splits = sample_with_extrapolation_holdout(
        space, "heparin_conc_uM", train_range, extrapolate_range,
        n_train=8, n_val=4, n_test=4, n_extrapolate=6, seed=0,
    )

    in_range_heparin = [
        s["heparin_conc_uM"] for split_name in ("train", "val", "test") for s in splits[split_name]
    ]
    extrapolation_heparin = [s["heparin_conc_uM"] for s in splits["extrapolation"]]

    assert all(train_range[0] <= v <= train_range[1] for v in in_range_heparin)
    assert all(extrapolate_range[0] <= v <= extrapolate_range[1] for v in extrapolation_heparin)

    # The literal ask: no overlap between the two ranges actually used.
    assert max(in_range_heparin) <= min(extrapolation_heparin)


def test_split_sizes_match_requested_counts():
    space = ParameterSpace()
    splits = sample_with_extrapolation_holdout(
        space, "heparin_conc_uM", (0.1, 0.38), (0.38, 0.5),
        n_train=8, n_val=4, n_test=4, n_extrapolate=6, seed=0,
    )
    assert len(splits["train"]) == 8
    assert len(splits["val"]) == 4
    assert len(splits["test"]) == 4
    assert len(splits["extrapolation"]) == 6


def test_other_parameters_still_drawn_from_full_range():
    """Only the extrapolate_param's range is restricted -- every other
    parameter should still be sampled from its full sampler.DEFAULT_RANGES
    span in both the in-range and extrapolation draws."""

    space = ParameterSpace()
    splits = sample_with_extrapolation_holdout(
        space, "heparin_conc_uM", (0.1, 0.38), (0.38, 0.5),
        n_train=8, n_val=4, n_test=4, n_extrapolate=6, seed=0,
    )
    all_samples = [s for name in ("train", "val", "test", "extrapolation") for s in splits[name]]
    for name, (lo, hi) in space.ranges.items():
        if name == "heparin_conc_uM":
            continue
        values = [s[name] for s in all_samples]
        assert all(lo <= v <= hi for v in values)


def test_overlapping_ranges_raise_value_error():
    space = ParameterSpace()
    with pytest.raises(ValueError):
        sample_with_extrapolation_holdout(
            space, "heparin_conc_uM", (0.1, 0.4), (0.38, 0.5),
            n_train=8, n_val=4, n_test=4, n_extrapolate=6, seed=0,
        )


def test_unknown_extrapolate_param_raises_value_error():
    space = ParameterSpace()
    with pytest.raises(ValueError):
        sample_with_extrapolation_holdout(
            space, "not_a_real_param", (0.0, 1.0), (1.0, 2.0),
            n_train=8, n_val=4, n_test=4, n_extrapolate=6, seed=0,
        )
