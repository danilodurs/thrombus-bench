"""Latin hypercube sampling over the physiological + geometric parameter space.

Responsibility
---------------
Define the parameter space used to generate the training/val/test/
edge-of-domain dataset for the neural surrogate, and draw stratified (Latin hypercube)
samples from it via `scipy.stats.qmc.LatinHypercube`. Parameters, with
physiologically-motivated ranges drawn from the paper's sensitivity studies
(Sec. 3.3):

* Geometry: aneurysm diameter (7-10 mm, interpolated/extrapolated around the
  paper's two studied geometries), vessel diameter (scaled proportionally).
* Inlet velocity: 30-100 cm/s (paper explores 45/75/100 cm/s, Fig. 9;
  physiological range 30-70 cm/s per Sec. 3.3.3 citation).
* Resting platelet concentration: 1e8-5e8 PLT/ml (Sec. 3.3.1 citation range).
* Heparin concentration: 0.1-0.5 uM (Fig. 8 range).

The edge-of-domain holdout split (`benchmark/edge_holdout_eval.py`, still
drawn from the same sampled parameter distribution -- not a genuinely
different population) is carved out of the *extremes* of these ranges: any
sample whose parameters fall in the outer `(1 - edge_holdout_quantile)` tail
(by Euclidean distance in normalized parameter space from the
sampled-population center) is routed to this holdout set instead of
train/val/test, per `configs/training.yaml` `data.edge_holdout_quantile`.

`sample_with_extrapolation_holdout` is a genuinely different, separate
evaluation split: rather than carving edge samples out of the *same*
sampled box, it restricts ONE parameter's *training* range to a
caller-chosen sub-interval (e.g. the lower 70% of its current range) and
draws a separate "extrapolation" split with that parameter restricted to
the *withheld* remainder -- a model trained only on the training
sub-interval has never seen that withheld region during training, unlike
the edge-of-domain holdout (see `data/generate_dataset.py`'s
`generate_extrapolation_dataset`, an opt-in variant of the default dataset
generation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc

DEFAULT_RANGES = {
    "aneurysm_diameter_mm": (7.0, 10.0),
    "vessel_diameter_mm": (3.2, 4.0),
    "inlet_velocity_cm_s": (30.0, 100.0),
    "platelet_conc_plt_ml": (1.0e8, 5.0e8),
    "heparin_conc_uM": (0.1, 0.5),
    "prothrombin_uM": (0.9, 1.3),
    "antithrombin_uM": (2.3, 3.4),
    "fibrinogen_uM": (5.5, 8.5),
}


@dataclass
class ParameterSpace:
    """Named (low, high) ranges for each sampled parameter."""

    ranges: dict = field(default_factory=lambda: dict(DEFAULT_RANGES))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.ranges.keys())


def normalize_params(params: np.ndarray, space: ParameterSpace) -> np.ndarray:
    """Min-max normalize a raw parameter vector to [-1, 1], based on `space`'s
    physical ranges (the same `DEFAULT_RANGES` used for sampling -- so the
    surrogate never has to learn about e.g. `platelet_conc_plt_ml` (~1e8)
    and `heparin_conc_uM` (~0.1-0.5) living on wildly different scales).

    `params`'s last axis must be ordered to match `space.names` (i.e.
    `generate_dataset.PARAM_ORDER`, which is defined in that same order).
    Accepts a single `(n_params,)` vector or a `(..., n_params)` batch.

    `normalized = 2 * (raw - low) / (high - low) - 1`. Inverse:
    `denormalize_params`.
    """

    names = space.names
    lows = np.array([space.ranges[n][0] for n in names], dtype=np.float64)
    highs = np.array([space.ranges[n][1] for n in names], dtype=np.float64)
    return 2.0 * (np.asarray(params, dtype=np.float64) - lows) / (highs - lows) - 1.0


def denormalize_params(normalized: np.ndarray, space: ParameterSpace) -> np.ndarray:
    """Inverse of `normalize_params`: map [-1, 1]-normalized values back to
    physical units (for interpretability/plotting)."""

    names = space.names
    lows = np.array([space.ranges[n][0] for n in names], dtype=np.float64)
    highs = np.array([space.ranges[n][1] for n in names], dtype=np.float64)
    return lows + (np.asarray(normalized, dtype=np.float64) + 1.0) / 2.0 * (highs - lows)


def latin_hypercube_sample(space: ParameterSpace, n_samples: int, seed: int = 0) -> list[dict]:
    """Draw `n_samples` Latin hypercube samples from `space`.

    Returns a list of dicts, one per sample, with keys matching
    `space.ranges`.
    """

    names = space.names
    d = len(names)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # (n_samples, d) in [0, 1)

    lows = np.array([space.ranges[n][0] for n in names])
    highs = np.array([space.ranges[n][1] for n in names])
    scaled = qmc.scale(unit_samples, lows, highs)

    return [dict(zip(names, row)) for row in scaled]


def split_train_val_test_edge_holdout(
    samples: list[dict],
    space: ParameterSpace,
    edge_holdout_quantile: float,
    n_train: int,
    n_val: int,
    n_test: int,
    n_edge_holdout: int,
    seed: int = 0,
) -> dict[str, list[dict]]:
    """Partition `samples` into train/val/test/edge-of-domain sets.

    Edge-of-domain samples are those farthest (in min-max-normalized
    parameter space, Euclidean distance from the space's center) from the
    bulk of the distribution -- i.e. drawn from the outer
    `(1 - edge_holdout_quantile)` tail; train/val/test are drawn (via a
    fixed random shuffle) from the remaining core-range pool. Raises
    `ValueError` if `samples` is too small to cover all four requested split
    sizes.
    """

    names = space.names
    lows = np.array([space.ranges[n][0] for n in names])
    highs = np.array([space.ranges[n][1] for n in names])
    normalized = np.array([[(s[n] - lo) / (hi - lo) for n, lo, hi in zip(names, lows, highs)] for s in samples])
    center = 0.5 * np.ones(len(names))
    distance = np.linalg.norm(normalized - center, axis=1)

    order = np.argsort(-distance)  # farthest-from-center first
    n_edge_holdout_actual = min(n_edge_holdout, max(0, len(samples) - (n_train + n_val + n_test)))
    if n_edge_holdout_actual < n_edge_holdout:
        raise ValueError(
            f"Not enough samples ({len(samples)}) to cover train+val+test+edge_holdout "
            f"({n_train}+{n_val}+{n_test}+{n_edge_holdout})."
        )
    edge_holdout_idx = order[:n_edge_holdout_actual]
    remaining_idx = order[n_edge_holdout_actual:]

    rng = np.random.default_rng(seed)
    shuffled = remaining_idx[rng.permutation(len(remaining_idx))]
    if len(shuffled) < n_train + n_val + n_test:
        raise ValueError(
            f"Not enough in-distribution samples ({len(shuffled)}) to cover "
            f"train+val+test ({n_train}+{n_val}+{n_test})."
        )
    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train : n_train + n_val]
    test_idx = shuffled[n_train + n_val : n_train + n_val + n_test]

    return {
        "train": [samples[i] for i in train_idx],
        "val": [samples[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
        "edge_holdout": [samples[i] for i in edge_holdout_idx],
    }


def sample_with_extrapolation_holdout(
    space: ParameterSpace,
    extrapolate_param: str,
    train_range: tuple[float, float],
    extrapolate_range: tuple[float, float],
    n_train: int,
    n_val: int,
    n_test: int,
    n_extrapolate: int,
    seed: int = 0,
) -> dict[str, list[dict]]:
    """Genuine-extrapolation split (see module docstring): draw train/val/test
    samples with `extrapolate_param` restricted to `train_range` (all other
    parameters drawn from `space`'s full ranges, as usual), plus a separate
    "extrapolation" split with `extrapolate_param` restricted to
    `extrapolate_range` instead (other parameters again from their full
    ranges) -- a model trained only on `train_range` has never seen
    `extrapolate_range` during training, unlike `split_train_val_test_edge_holdout`'s
    same-box edge samples.

    `train_range`/`extrapolate_range` must not overlap (raises `ValueError`
    otherwise) -- this function doesn't pick the split point itself; that's
    a caller decision (see `data/generate_dataset.py`'s
    `generate_extrapolation_dataset` for the current heparin_conc_uM choice
    and its rationale).

    train/val/test are drawn as one `n_train+n_val+n_test`-sample LHS batch
    from the restricted-range space and split via a fixed random shuffle
    (mirroring `split_train_val_test_edge_holdout`'s train/val/test
    assignment, minus its edge-of-domain distance scoring, which doesn't
    apply here); the extrapolation split is drawn as an independent
    `n_extrapolate`-sample LHS batch (seed+1, so it isn't correlated with
    the train/val/test draw) from the extrapolate-range space.
    """

    if extrapolate_param not in space.ranges:
        raise ValueError(f"{extrapolate_param!r} is not in space.ranges: {list(space.ranges)}")
    lo1, hi1 = train_range
    lo2, hi2 = extrapolate_range
    if not (hi1 <= lo2 or hi2 <= lo1):  # intervals overlap unless one ends before the other starts
        raise ValueError(f"train_range {train_range} and extrapolate_range {extrapolate_range} overlap for {extrapolate_param!r}")

    train_space = ParameterSpace(ranges={**space.ranges, extrapolate_param: train_range})
    extrapolate_space = ParameterSpace(ranges={**space.ranges, extrapolate_param: extrapolate_range})

    n_in_range = n_train + n_val + n_test
    in_range_samples = latin_hypercube_sample(train_space, n_in_range, seed=seed)
    extrapolate_samples = latin_hypercube_sample(extrapolate_space, n_extrapolate, seed=seed + 1)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_in_range)
    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train : n_train + n_val]
    test_idx = shuffled[n_train + n_val : n_train + n_val + n_test]

    return {
        "train": [in_range_samples[i] for i in train_idx],
        "val": [in_range_samples[i] for i in val_idx],
        "test": [in_range_samples[i] for i in test_idx],
        "extrapolation": extrapolate_samples,
    }
