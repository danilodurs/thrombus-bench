"""Evaluate neural surrogate degradation on the held-out edge-of-domain split.

Responsibility
---------------
Run `benchmark/metrics.py` metrics separately on `data/dataset.py`
`split="test"` (core-range samples) and `split="edge_holdout"`
(edge-of-domain, extreme platelet/heparin/velocity combinations per
`data/sampler.py`), and report the *degradation*
(metric_edge_holdout - metric_test, or ratio) per metric.

What this does and does not measure
------------------------------------
The edge-of-domain holdout is built by taking the samples farthest
(Euclidean distance in min-max-normalized parameter space) from the center
of the *same* bounded Latin-hypercube sampling box used for train/val/test
(`sampler.split_train_val_test_edge_holdout`). It is still drawn from within
the training distribution's support -- just its edge -- not from a
genuinely different population or covariate-shifted regime. This module
therefore measures whether the surrogate's accuracy degrades toward the
edge of the sampled parameter range, which is a useful but limited signal:
it is NOT a test of extrapolation beyond the simulator's own sampled
behavior, and it says nothing about physiological realism. A genuine
extrapolation split (one parameter restricted to a sub-range during
training, tested on the withheld remainder) is implemented separately --
see `extrapolation_eval.py` and `data/sampler.
sample_with_extrapolation_holdout`.

`evaluate_edge_holdout_degradation_continuous` (Phase 7, `docs/
continuous_surrogate_design.md`) is the point-query counterpart for
`ContinuousThrombusSurrogate`/`neural.baselines.Continuous*` (same call
signature: `forward(params_with_time, query_points_m, batch_index,
geometry_mm)`), using `PointCloudThrombusDataset` + `benchmark.metrics.
field_rmse_pointwise` instead of `ThrombusSurrogateDataset` + `field_rmse`
-- kept as a separate function, not a mode-branch, matching this project's
established grid-path/continuous-path pattern (`train`/`train_continuous`,
`field_rmse`/`field_rmse_pointwise`, etc.) since the two operate on
structurally different batch shapes (fixed raster vs. ragged point cloud).

Split-level guarantee this module relies on: `data/sampler.
split_train_val_test_edge_holdout` partitions *samples* (whole simulations)
before any mechanistic run happens, and `data/generate_dataset.
_generate_from_splits` writes each sample's `.npz` (every checkpoint, every
mesh node together, per Phase 1/3's schema) to exactly one split directory
-- so a sample's checkpoints/nodes can never end up split across `test`
and `edge_holdout` for the point-cloud path either. See
`tests/test_split_sample_level_integrity.py` for an explicit check against
the actual generated files, not just the sampler-level partition.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..data.dataset import ThrombusSurrogateDataset, pointcloud_collate_fn
from .metrics import field_rmse, field_rmse_pointwise


def evaluate_edge_holdout_degradation(
    model: torch.nn.Module, test_dataset: ThrombusSurrogateDataset, edge_holdout_dataset: ThrombusSurrogateDataset
) -> dict:
    """Returns {"test": {...}, "edge_holdout": {...}, "degradation_ratio": float},
    comparing overall field RMSE (`benchmark/metrics.field_rmse`) on the
    core-range test set vs. the edge-of-domain (parameter-space tail) set.
    `degradation_ratio` > 1 means the surrogate is worse on edge-of-domain
    samples than core-range ones."""

    model.eval()
    results = {}
    for name, dataset in (("test", test_dataset), ("edge_holdout", edge_holdout_dataset)):
        loader = DataLoader(dataset, batch_size=len(dataset))
        with torch.no_grad():
            batch = next(iter(loader))
            pred = model(batch["params"])
        results[name] = field_rmse(pred.numpy(), batch["fields"].numpy())

    degradation_ratio = (
        results["edge_holdout"]["overall"] / results["test"]["overall"] if results["test"]["overall"] > 0 else float("inf")
    )
    return {"test": results["test"], "edge_holdout": results["edge_holdout"], "degradation_ratio": degradation_ratio}


def evaluate_edge_holdout_degradation_continuous(
    model: torch.nn.Module, test_dataset, edge_holdout_dataset
) -> dict:
    """Point-query counterpart of `evaluate_edge_holdout_degradation` -- see
    module docstring. `model` is any of `ContinuousThrombusSurrogate` /
    `neural.baselines.ContinuousMeanFieldBaseline` /
    `ContinuousNearestNeighborBaseline` (same call signature); `test_dataset`/
    `edge_holdout_dataset` are `data.dataset.PointCloudThrombusDataset`
    instances. Returns the same `{"test", "edge_holdout",
    "degradation_ratio"}` shape as the grid version, using
    `benchmark.metrics.field_rmse_pointwise` instead of `field_rmse`."""

    model.eval()
    results = {}
    for name, dataset in (("test", test_dataset), ("edge_holdout", edge_holdout_dataset)):
        loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=pointcloud_collate_fn)
        with torch.no_grad():
            batch = next(iter(loader))
            pred = model(batch["params_with_time"], batch["node_coords"], batch["batch_index"], batch["geometry_mm"])
        results[name] = field_rmse_pointwise(pred.numpy(), batch["fields"].numpy())

    degradation_ratio = (
        results["edge_holdout"]["overall"] / results["test"]["overall"] if results["test"]["overall"] > 0 else float("inf")
    )
    return {"test": results["test"], "edge_holdout": results["edge_holdout"], "degradation_ratio": degradation_ratio}
