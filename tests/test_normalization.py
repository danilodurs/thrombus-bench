"""Tests for data/sampler.py's normalize_params/denormalize_params, which
min-max normalize the raw (8,) parameter vector (data/generate_dataset.
PARAM_ORDER order) to [-1, 1] before it's fed into the neural surrogate
(see data/dataset.py::ThrombusSurrogateDataset.__getitem__)."""

from __future__ import annotations

import numpy as np

from thrombus_bench.data.sampler import ParameterSpace, denormalize_params, normalize_params


def _lows_highs(space: ParameterSpace) -> tuple[np.ndarray, np.ndarray]:
    lows = np.array([space.ranges[n][0] for n in space.names])
    highs = np.array([space.ranges[n][1] for n in space.names])
    return lows, highs


def test_range_minimum_maps_to_minus_one():
    space = ParameterSpace()
    lows, _ = _lows_highs(space)
    normalized = normalize_params(lows, space)
    np.testing.assert_allclose(normalized, -1.0)


def test_range_maximum_maps_to_plus_one():
    space = ParameterSpace()
    _, highs = _lows_highs(space)
    normalized = normalize_params(highs, space)
    np.testing.assert_allclose(normalized, 1.0)


def test_round_trip_recovers_original_value():
    space = ParameterSpace()
    lows, highs = _lows_highs(space)
    rng = np.random.default_rng(0)
    raw = lows + rng.random(len(space.names)) * (highs - lows)

    recovered = denormalize_params(normalize_params(raw, space), space)

    np.testing.assert_allclose(recovered, raw, rtol=1e-12, atol=1e-12)


def test_midpoint_maps_to_zero():
    space = ParameterSpace()
    lows, highs = _lows_highs(space)
    midpoint = 0.5 * (lows + highs)

    normalized = normalize_params(midpoint, space)

    np.testing.assert_allclose(normalized, 0.0, atol=1e-12)
