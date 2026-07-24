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
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..data.dataset import ThrombusSurrogateDataset
from .metrics import field_rmse


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
