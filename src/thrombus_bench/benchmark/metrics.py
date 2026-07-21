"""Accuracy, physics-fidelity, and runtime metrics comparing mechanistic vs. neural.

Responsibility
---------------
Quantitative comparison metrics between the mechanistic solver
(`mechanistic/coupled_solver.py`, via `data/generate_dataset.py`'s saved
outputs) and the neural surrogate (`neural/model.py`), evaluated on the
held-out test set (`data/dataset.py` `split="test"`):

* `field_rmse`: per-channel and overall RMSE between predicted and
  mechanistic-reference field grids (velocity + 9 species concentrations).
* `thrombus_mask` / `thrombus_iou`: binary thrombosed-region mask (paper's
  Sec. 2.6 M_at/FI thresholds) and its IoU between two masks -- usable if a
  full spatial M_at/FI field is available (not saved by this project's
  reduced-scale `generate_dataset.py`, which stores only the summary
  scalars below; kept for use with a full-fidelity dataset).
* `max_M_at_relative_error` / `thrombosed_fraction_error`: errors on the
  scalar summary targets `generate_dataset.py` *does* save.
* `runtime_comparison`: wall-clock time per simulation, mechanistic vs.
  neural (the surrogate's main practical selling point).

`thrombus_height_error`, `time_to_onset_error`, and `physics_residual_audit`
are **not implemented**: they require a full spatiotemporal M_at/FI field
(height/onset-time over the mesh, over multiple checkpoints) that this
project's dataset does not store (see `data/generate_dataset.py`'s scope
note -- it saves the final-checkpoint summary scalars only). A future,
larger-scale dataset generation run that saves full per-checkpoint spatial
fields could implement these directly against the saved arrays.
"""

from __future__ import annotations

import numpy as np


def thrombus_mask(M_at: np.ndarray, FI: np.ndarray, M_at_critical: float, fibrin_critical: float) -> np.ndarray:
    """Binary thrombosed-region mask per the paper's Sec. 2.6 thresholds
    (hard threshold here, unlike the smoothed Eq. 18 viscosity multiplier --
    metrics should reflect the paper's literal definition)."""

    return (M_at >= M_at_critical) | (FI >= fibrin_critical)


def thrombus_iou(pred_mask: np.ndarray, ref_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, ref_mask).sum()
    union = np.logical_or(pred_mask, ref_mask).sum()
    return float(intersection) / float(union) if union > 0 else 1.0


def field_rmse(pred_fields: np.ndarray, true_fields: np.ndarray) -> dict:
    """RMSE between predicted and reference field grids, shape (C, H, W) or
    (B, C, H, W). Returns {"overall": float, "per_channel": (C,) array}."""

    axis = tuple(range(pred_fields.ndim - 2, pred_fields.ndim)) if pred_fields.ndim >= 3 else None
    sq_err = (pred_fields - true_fields) ** 2
    if pred_fields.ndim == 4:
        per_channel = np.sqrt(sq_err.mean(axis=(0, 2, 3)))
    elif pred_fields.ndim == 3:
        per_channel = np.sqrt(sq_err.mean(axis=(1, 2)))
    else:
        per_channel = None
    return {"overall": float(np.sqrt(sq_err.mean())), "per_channel": per_channel}


def max_M_at_relative_error(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Relative error |pred - true| / (|true| + eps) on the max-M_at summary target."""

    return np.abs(pred - true) / (np.abs(true) + eps)


def thrombosed_fraction_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Absolute error on the (already fractional, in [0,1]) thrombosed-area summary target."""

    return np.abs(pred - true)


def runtime_comparison(mechanistic_times_s: np.ndarray, neural_times_s: np.ndarray) -> dict:
    """Summary runtime statistics and the neural surrogate's speedup factor."""

    mech_mean = float(np.mean(mechanistic_times_s))
    neural_mean = float(np.mean(neural_times_s))
    return {
        "mechanistic_mean_s": mech_mean,
        "neural_mean_s": neural_mean,
        "speedup_factor": mech_mean / neural_mean if neural_mean > 0 else float("inf"),
    }


def thrombus_height_error(pred_mask: np.ndarray, ref_mask: np.ndarray, node_coords: np.ndarray) -> float:
    raise NotImplementedError(
        "metrics.thrombus_height_error: requires a full spatial M_at/FI field, not saved by this "
        "project's reduced-scale generate_dataset.py -- see module docstring"
    )


def time_to_onset_error(pred_mask_series: np.ndarray, ref_mask_series: np.ndarray, times_s: np.ndarray) -> float:
    raise NotImplementedError(
        "metrics.time_to_onset_error: requires a per-checkpoint spatial M_at/FI time series, not "
        "saved by this project's reduced-scale generate_dataset.py -- see module docstring"
    )


def physics_residual_audit(predicted_fields: dict) -> dict:
    raise NotImplementedError("metrics.physics_residual_audit: not yet implemented")
