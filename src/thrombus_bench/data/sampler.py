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
