"""Uncertainty calibration: predicted variance vs. actual error, reliability diagrams.

Responsibility
---------------
Check whether the `neural/uncertainty.py` predictive variance is
well-calibrated: bin test-set predictions by predicted variance and compare
against observed squared error (reliability diagram / calibration curve),
and compute a scalar calibration error (expected calibration error, ECE,
adapted for regression as in e.g. Kuleshov et al. 2018).
"""

from __future__ import annotations

import numpy as np


def reliability_diagram_data(predicted_mean: np.ndarray, predicted_variance: np.ndarray, target: np.ndarray, n_bins: int = 10) -> dict:
    """Bins predictions by predicted variance (equal-count bins) and
    computes mean predicted variance vs. mean squared error per bin --
    the raw data for a reliability diagram plot (`viz/plots.plot_calibration`).

    A well-calibrated model has `mean_variance[i] ~= mean_squared_error[i]`
    for every bin `i` (predicted variance should equal expected squared
    error, in the Gaussian-likelihood sense).
    """

    variance_flat = predicted_variance.ravel()
    sq_err_flat = ((predicted_mean - target) ** 2).ravel()

    order = np.argsort(variance_flat)
    variance_sorted = variance_flat[order]
    sq_err_sorted = sq_err_flat[order]

    bin_edges_idx = np.linspace(0, len(variance_sorted), n_bins + 1).astype(int)
    mean_variance = np.zeros(n_bins)
    mean_sq_err = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        lo, hi = bin_edges_idx[i], bin_edges_idx[i + 1]
        if hi > lo:
            mean_variance[i] = variance_sorted[lo:hi].mean()
            mean_sq_err[i] = sq_err_sorted[lo:hi].mean()
            counts[i] = hi - lo

    return {"mean_variance": mean_variance, "mean_squared_error": mean_sq_err, "counts": counts}


def expected_calibration_error(predicted_mean: np.ndarray, predicted_variance: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean absolute difference between per-bin predicted variance
    and observed squared error (0 = perfectly calibrated)."""

    data = reliability_diagram_data(predicted_mean, predicted_variance, target, n_bins)
    total = data["counts"].sum()
    if total == 0:
        return float("nan")
    weights = data["counts"] / total
    return float(np.sum(weights * np.abs(data["mean_variance"] - data["mean_squared_error"])))
