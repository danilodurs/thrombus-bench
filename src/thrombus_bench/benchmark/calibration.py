"""Uncertainty calibration: predicted variance vs. actual error, reliability diagrams.

Responsibility
---------------
Check whether the `neural/uncertainty.py` predictive variance is
well-calibrated: bin test-set predictions by predicted variance and compare
against observed error (reliability diagram / calibration curve), and
compute a scalar calibration error (e.g. expected calibration error, ECE,
adapted for regression).

Not yet implemented -- this is a scaffolding stub. Depends on
`neural/uncertainty.py`.
"""

from __future__ import annotations

import numpy as np


def reliability_diagram_data(predicted_mean: np.ndarray, predicted_variance: np.ndarray, target: np.ndarray, n_bins: int = 10) -> dict:
    """Returns bin edges, mean predicted variance per bin, and mean squared
    error per bin -- the raw data for a reliability diagram plot
    (`viz/plots.py`)."""

    raise NotImplementedError("calibration.reliability_diagram_data: not yet implemented")


def expected_calibration_error(predicted_mean: np.ndarray, predicted_variance: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    raise NotImplementedError("calibration.expected_calibration_error: not yet implemented")
