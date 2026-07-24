"""Evaluate neural surrogate degradation on a genuine-extrapolation split.

Responsibility
---------------
Run `benchmark/metrics.py` metrics separately on `data/dataset.py`
`split="test"` and `split="extrapolation"` of a dataset directory written
by `data/generate_dataset.generate_extrapolation_dataset`
(`--extrapolation-param` CLI flag), and report the *degradation*
(ratio) the same way `edge_holdout_eval.py` does for the edge-of-domain
holdout.

How this differs from `edge_holdout_eval.py`
-----------------------------------------------
The edge-of-domain holdout (`edge_holdout_eval.py`) is drawn from the
*same* sampled parameter box as train/val/test -- it measures degradation
toward the edge of the training distribution's support, not extrapolation
beyond it. This module evaluates a model trained with one parameter's
range *restricted* to a sub-interval (`sample_with_extrapolation_holdout`),
on samples drawn from the *withheld* remainder of that parameter's range --
values the model has genuinely never seen during training. The two must
never be confused (hence the explicit, spelled-out label this module
always attaches, e.g. "heparin_conc_uM extrapolation (trained on
0.1-0.38, tested on 0.38-0.5)") -- they answer different questions and
are not comparable numbers.

This also requires a *separately trained* model/checkpoint (one trained on
`generate_extrapolation_dataset`'s restricted-range train split): reusing
the default demo/pilot checkpoint would not test extrapolation at all,
since that checkpoint's training data already spans the full parameter
range, including whatever sub-interval would otherwise be "withheld".

`evaluate_extrapolation_degradation_continuous` (Phase 7, `docs/
continuous_surrogate_design.md`) is the point-query counterpart for
`ContinuousThrombusSurrogate`/`neural.baselines.Continuous*`, using
`PointCloudThrombusDataset` + `benchmark.metrics.field_rmse_pointwise` --
same rationale as `edge_holdout_eval.py`'s continuous counterpart for
keeping it a separate function rather than a mode-branch.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..data.dataset import ThrombusSurrogateDataset, pointcloud_collate_fn
from .metrics import field_rmse, field_rmse_pointwise


def extrapolation_label(extrapolate_param: str, train_range: tuple[float, float], extrapolate_range: tuple[float, float]) -> str:
    """Explicit, spelled-out label so this is never confused with the
    edge-of-domain holdout (see module docstring)."""

    return (
        f"{extrapolate_param} extrapolation (trained on {train_range[0]:g}-{train_range[1]:g}, "
        f"tested on {extrapolate_range[0]:g}-{extrapolate_range[1]:g})"
    )


def evaluate_extrapolation_degradation(
    model: torch.nn.Module,
    test_dataset: ThrombusSurrogateDataset,
    extrapolation_dataset: ThrombusSurrogateDataset,
    extrapolate_param: str,
    train_range: tuple[float, float],
    extrapolate_range: tuple[float, float],
) -> dict:
    """Returns {"label": str, "test": {...}, "extrapolation": {...},
    "degradation_ratio": float}, comparing overall field RMSE
    (`benchmark/metrics.field_rmse`) on the (in-training-range) test set
    vs. the withheld-range extrapolation set. `degradation_ratio` > 1 means
    the surrogate is worse on genuinely-unseen-range samples than
    in-range ones."""

    model.eval()
    results = {}
    for name, dataset in (("test", test_dataset), ("extrapolation", extrapolation_dataset)):
        loader = DataLoader(dataset, batch_size=len(dataset))
        with torch.no_grad():
            batch = next(iter(loader))
            pred = model(batch["params"])
        results[name] = field_rmse(pred.numpy(), batch["fields"].numpy())

    degradation_ratio = (
        results["extrapolation"]["overall"] / results["test"]["overall"] if results["test"]["overall"] > 0 else float("inf")
    )
    return {
        "label": extrapolation_label(extrapolate_param, train_range, extrapolate_range),
        "test": results["test"],
        "extrapolation": results["extrapolation"],
        "degradation_ratio": degradation_ratio,
    }


def evaluate_extrapolation_degradation_continuous(
    model: torch.nn.Module,
    test_dataset,
    extrapolation_dataset,
    extrapolate_param: str,
    train_range: tuple[float, float],
    extrapolate_range: tuple[float, float],
) -> dict:
    """Point-query counterpart of `evaluate_extrapolation_degradation` --
    see module docstring. `model` is any of `ContinuousThrombusSurrogate` /
    `neural.baselines.ContinuousMeanFieldBaseline` /
    `ContinuousNearestNeighborBaseline`; `test_dataset`/
    `extrapolation_dataset` are `data.dataset.PointCloudThrombusDataset`
    instances. Same return shape as the grid version, using
    `benchmark.metrics.field_rmse_pointwise` instead of `field_rmse`."""

    model.eval()
    results = {}
    for name, dataset in (("test", test_dataset), ("extrapolation", extrapolation_dataset)):
        loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=pointcloud_collate_fn)
        with torch.no_grad():
            batch = next(iter(loader))
            pred = model(batch["params_with_time"], batch["node_coords"], batch["batch_index"], batch["geometry_mm"])
        results[name] = field_rmse_pointwise(pred.numpy(), batch["fields"].numpy())

    degradation_ratio = (
        results["extrapolation"]["overall"] / results["test"]["overall"] if results["test"]["overall"] > 0 else float("inf")
    )
    return {
        "label": extrapolation_label(extrapolate_param, train_range, extrapolate_range),
        "test": results["test"],
        "extrapolation": results["extrapolation"],
        "degradation_ratio": degradation_ratio,
    }
