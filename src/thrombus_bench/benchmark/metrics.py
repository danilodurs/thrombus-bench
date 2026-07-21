"""Accuracy, physics-fidelity, and runtime metrics comparing mechanistic vs. neural.

Responsibility
---------------
Quantitative comparison metrics between the mechanistic solver
(`mechanistic/coupled_solver.py`) and the neural surrogate
(`neural/model.py`), evaluated on the held-out test set
(`data/dataset.py` `split="test"`):

* Thrombus geometry error: height and area error of the thrombosed region
  (defined by the paper's M_at/FI thresholds, Sec. 2.6) between predicted
  and mechanistic-reference fields.
* IoU of the thrombosed region (binary mask from the same thresholds).
* Time-to-onset error: difference in the first time step at which any node
  crosses the thrombosis threshold.
* Physics-residual audit: evaluate `neural/physics_losses.py` residuals on
  the neural surrogate's *test-time* predictions (not used for training) as
  an interpretability/trust diagnostic.
* Runtime comparison: wall-clock time per simulation, mechanistic vs. neural
  (the surrogate's main practical selling point).

Not yet implemented -- this is a scaffolding stub.
"""

from __future__ import annotations

import numpy as np


def thrombus_mask(M_at: np.ndarray, FI: np.ndarray, M_at_critical: float, fibrin_critical: float) -> np.ndarray:
    """Binary thrombosed-region mask per the paper's Sec. 2.6 thresholds
    (hard threshold here, unlike the smoothed Eq. 18 viscosity multiplier --
    metrics should reflect the paper's literal definition)."""

    return (M_at >= M_at_critical) | (FI >= fibrin_critical)


def thrombus_height_error(pred_mask: np.ndarray, ref_mask: np.ndarray, node_coords: np.ndarray) -> float:
    raise NotImplementedError("metrics.thrombus_height_error: not yet implemented")


def thrombus_area_error(pred_mask: np.ndarray, ref_mask: np.ndarray, node_coords: np.ndarray, elements: np.ndarray) -> float:
    raise NotImplementedError("metrics.thrombus_area_error: not yet implemented")


def thrombus_iou(pred_mask: np.ndarray, ref_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, ref_mask).sum()
    union = np.logical_or(pred_mask, ref_mask).sum()
    return float(intersection) / float(union) if union > 0 else 1.0


def time_to_onset_error(pred_mask_series: np.ndarray, ref_mask_series: np.ndarray, times_s: np.ndarray) -> float:
    raise NotImplementedError("metrics.time_to_onset_error: not yet implemented")


def physics_residual_audit(predicted_fields: dict) -> dict:
    raise NotImplementedError("metrics.physics_residual_audit: not yet implemented")
