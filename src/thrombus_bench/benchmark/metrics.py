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
note -- it saves the final-checkpoint summary scalars only). Note that as
of the point-cloud data path (`docs/continuous_surrogate_design.md`
Phase 1/3), full per-checkpoint spatial `M_at`/species fields *are* now
saved (`_build_pointcloud_sample`'s `fields`/`M_at_wall_values`, one
snapshot per `data.n_snapshots`) -- these two could plausibly be
implemented against that data now, but that's a separate, not-yet-assessed
piece of work, not something Phase 6 undertook.

Point-query metrics (`docs/continuous_surrogate_design.md` Phase 6)
------------------------------------------------------------------------
`field_rmse` above assumes a fixed `(H, W)` raster -- still correct and
used as-is for the original grid-projection FNO, kept as a comparison
baseline per the design summary. `ContinuousThrombusSurrogate` and its two
continuous baselines (`neural.baselines.Continuous*`) instead predict at
arbitrary query points, so their evaluation needs a different shape of
ground truth: a held-out sample's own mesh node coordinates (the only
place ground truth actually exists -- there's no reference value at an
arbitrary query point). `field_rmse_pointwise`, `field_rmse_by_checkpoint`,
and `field_rmse_by_distance_to_wall` below are that point-query
counterpart family.

Bootstrap resampling unit (`docs/continuous_surrogate_design.md` Phase 6)
------------------------------------------------------------------------
Audited: as of Phase 6, no bootstrap/confidence-interval code exists
anywhere in this project yet (`run_benchmark.py`, `calibration.py`,
`edge_holdout_eval.py` -- checked directly, not assumed) -- so this is
guidance for whenever one gets added, not a fix to something broken.
`bootstrap_metric_by_sample` below establishes the correct pattern
pre-emptively: **the resampling unit must be `sample_id` (a whole
simulation), never an individual query point or checkpoint**. Points from
the same mesh and checkpoints from the same run are spatially/temporally
correlated, not independent draws -- resampling at that finer grain would
treat correlated observations as independent and understate the true
uncertainty (the classic bootstrap pitfall: the resampling unit must match
the data's actual unit of exchangeability). Concretely, this means:
aggregate each held-out simulation down to one value (or one small feature
vector) per `sample_id` *before* bootstrapping -- e.g. that sample's own
mean RMSE across all its checkpoints/query points -- and resample *those*
rows with replacement, never the underlying points/checkpoints directly.
"""

from __future__ import annotations

from typing import Callable

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


def field_rmse(pred_fields: np.ndarray, true_fields: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """RMSE between predicted and reference field grids, shape (C, H, W) or
    (B, C, H, W). Returns {"overall": float, "per_channel": (C,) array,
    "fluid_only": float | None, "per_channel_fluid_only": (C,) array | None}.

    The unmasked ("all cells") `"overall"`/`"per_channel"` entries are
    unchanged from before `mask` existed -- default `mask=None` keeps every
    existing caller/test's behavior identical.

    `mask` (optional; `data/dataset.py`'s `fluid_mask`, shape (H, W) for a
    single sample or (B, H, W) for a batch -- no channel dimension, it's
    shared across channels) restricts an *additional* RMSE to fluid cells
    only. The vessel+aneurysm domain occupies only part of the
    rasterization bounding box (see `data/generate_dataset._fluid_mask`);
    the exterior background is comparatively easy to reconstruct (mostly
    constant/near-zero filler from `griddata(method="nearest")`), so an
    unmasked RMSE alone can make a surrogate look more accurate on the
    fluid interior than it really is -- this "fluid_only" pair makes that
    gap visible instead of hiding it inside a single blended number.
    """

    sq_err = (pred_fields - true_fields) ** 2
    if pred_fields.ndim == 4:
        per_channel = np.sqrt(sq_err.mean(axis=(0, 2, 3)))
    elif pred_fields.ndim == 3:
        per_channel = np.sqrt(sq_err.mean(axis=(1, 2)))
    else:
        per_channel = None

    result = {
        "overall": float(np.sqrt(sq_err.mean())),
        "per_channel": per_channel,
        "fluid_only": None,
        "per_channel_fluid_only": None,
    }
    if mask is None:
        return result

    if pred_fields.ndim == 4:
        mask_b = mask.astype(sq_err.dtype)[:, None, :, :]  # broadcast over the channel axis
        spatial_axes = (0, 2, 3)
    elif pred_fields.ndim == 3:
        mask_b = mask.astype(sq_err.dtype)[None, :, :]
        spatial_axes = (1, 2)
    else:
        raise ValueError("field_rmse: mask is only supported alongside 3D (C,H,W) or 4D (B,C,H,W) pred_fields")

    n_channels = pred_fields.shape[-3]
    n_fluid = float(mask.sum())  # fluid-cell count across batch/spatial dims, same for every channel
    masked_sq_err = sq_err * mask_b
    result["per_channel_fluid_only"] = np.sqrt(masked_sq_err.sum(axis=spatial_axes) / max(n_fluid, 1.0))
    result["fluid_only"] = float(np.sqrt(masked_sq_err.sum() / max(n_fluid * n_channels, 1.0)))
    return result


def field_rmse_pointwise(pred_fields: np.ndarray, true_fields: np.ndarray) -> dict:
    """`field_rmse`'s point-query counterpart, for `ContinuousThrombusSurrogate`
    and the continuous baselines (`neural.baselines.Continuous*`) -- see
    module docstring's "Point-query metrics" section.

    `pred_fields`/`true_fields`: `(n_points, n_channels)`, both evaluated at
    the SAME points -- a held-out sample's own mesh node coordinates, the
    only place ground truth actually exists (there is no reference value
    at an arbitrary continuous query point). Returns `{"overall": float,
    "per_channel": (n_channels,) array}`.

    No `fluid_only` variant here, unlike `field_rmse`: every point already
    is a real mesh node inside the fluid domain by construction, so there
    is no rasterization-exterior-cell problem to correct for.
    """

    sq_err = (pred_fields - true_fields) ** 2
    return {
        "overall": float(np.sqrt(sq_err.mean())),
        "per_channel": np.sqrt(sq_err.mean(axis=0)),
    }


def field_rmse_by_checkpoint(pred_fields: np.ndarray, true_fields: np.ndarray, checkpoint_id: np.ndarray) -> dict:
    """`field_rmse_pointwise`, broken down per checkpoint -- the temporal
    breakdown the design summary intended.

    `checkpoint_id`: `(n_points,)`, an integer (or otherwise hashable)
    label grouping points by which checkpoint they came from (e.g. a
    `(sample_id, checkpoint_idx)` composite mapped to an integer, or simply
    `checkpoint_idx` if pooling within one sample). Returns `{checkpoint_id:
    field_rmse_pointwise(...)}` for each distinct label present.
    """

    checkpoint_id = np.asarray(checkpoint_id)
    return {
        cp: field_rmse_pointwise(pred_fields[checkpoint_id == cp], true_fields[checkpoint_id == cp])
        for cp in np.unique(checkpoint_id)
    }


def field_rmse_by_distance_to_wall(
    pred_fields: np.ndarray, true_fields: np.ndarray, sdf_values: np.ndarray, bin_edges: np.ndarray | None = None
) -> dict:
    """RMSE binned by distance to the nearest wall -- a genuinely new
    capability the continuous model's point-query predictions enable
    directly (`mechanistic.geometry_sdf.signed_distance_to_wall`, Phase 1),
    answering "is accuracy near the wall (where the interesting thrombus
    physics is) worse than in the bulk?" continuously, rather than via the
    old grid path's `fluid_mask`-erosion approximation (`field_rmse`'s
    `mask` argument, which only distinguishes fluid-vs-exterior, not
    near-wall-vs-bulk-interior).

    `sdf_values`: `(n_points,)`, Phase 1's signed distance (meters,
    positive inside the domain) at each query point -- the caller computes
    this (e.g. via `signed_distance_to_wall` against each point's own
    sample geometry); this function only bins by it, matching `field_rmse`'s
    existing pattern of taking `mask` as a precomputed argument rather than
    computing it itself. Bins by *unsigned* distance (`abs(sdf_values)`),
    since "near the wall" is symmetric regardless of which side of zero a
    point (numerically) falls on.

    `bin_edges`: increasing distance thresholds in meters; default 5 bins
    spanning `[0, max(|sdf_values|)]` if not given (a reasonable default
    for a single call, but pooling multiple samples/geometries with very
    different scales should pass explicit, physically-meaningful edges,
    e.g. based on vessel diameter, instead of relying on the data's own
    max).

    Returns `{"bin_edges": array, "rmse_per_bin": (n_bins,) array (NaN for
    empty bins), "n_points_per_bin": (n_bins,) int array}`.
    """

    distance = np.abs(np.asarray(sdf_values))
    if bin_edges is None:
        max_distance = float(distance.max()) if len(distance) else 1.0
        bin_edges = np.linspace(0.0, max_distance, 6)
    bin_edges = np.asarray(bin_edges, dtype=float)
    n_bins = len(bin_edges) - 1

    bin_idx = np.digitize(distance, bin_edges[1:-1])
    per_point_sq_err = ((pred_fields - true_fields) ** 2).mean(axis=-1)

    rmse_per_bin = np.full(n_bins, np.nan)
    n_points_per_bin = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        in_bin = bin_idx == b
        n_points_per_bin[b] = int(in_bin.sum())
        if in_bin.any():
            rmse_per_bin[b] = float(np.sqrt(per_point_sq_err[in_bin].mean()))

    return {"bin_edges": bin_edges, "rmse_per_bin": rmse_per_bin, "n_points_per_bin": n_points_per_bin}


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


def bootstrap_metric_by_sample(
    metric_fn: Callable[[np.ndarray], float],
    per_sample_values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    """Bootstrap confidence interval for `metric_fn`, resampling **whole
    `sample_id` rows** -- see module docstring's "Bootstrap resampling
    unit" section for why this is the only valid resampling unit here.

    `per_sample_values`: `(n_samples, ...)` -- exactly one row per held-out
    *simulation*, already aggregated across that simulation's own
    checkpoints/query points (this function does not do that aggregation
    for you; it only resamples rows). `metric_fn` is applied to the whole
    `(n_samples, ...)` array (e.g. `np.mean`, or a function computing
    `field_rmse_pointwise`-style aggregate over a stacked per-sample-RMSE
    array) and must accept a resampled array of the same shape.

    Returns `{"point_estimate": metric_fn(per_sample_values), "lower":
    ..., "upper": ..., "confidence": confidence, "n_bootstrap":
    n_bootstrap}`, `lower`/`upper` from the bootstrap distribution's
    `confidence`-level percentile interval.
    """

    rng = rng if rng is not None else np.random.default_rng()
    per_sample_values = np.asarray(per_sample_values)
    n_samples = per_sample_values.shape[0]

    point_estimate = metric_fn(per_sample_values)

    bootstrap_estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample ROW INDICES (whole sample_ids), with replacement -- never
        # flatten and resample individual elements within a row.
        resampled_row_indices = rng.integers(0, n_samples, size=n_samples)
        bootstrap_estimates[i] = metric_fn(per_sample_values[resampled_row_indices])

    alpha = 1.0 - confidence
    lower, upper = np.percentile(bootstrap_estimates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "point_estimate": float(point_estimate),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }
