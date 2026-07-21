"""Evaluate neural surrogate degradation on the held-out OOD split.

Responsibility
---------------
Run `benchmark/metrics.py` metrics separately on `data/dataset.py`
`split="test"` (in-distribution) and `split="ood"` (out-of-distribution,
extreme platelet/heparin/velocity combinations per `data/sampler.py`), and
report the *degradation* (metric_ood - metric_test, or ratio) per metric.
This is the primary signal for whether the surrogate has learned the
underlying physics well enough to extrapolate, vs. having merely
interpolated the training distribution.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..data.dataset import ThrombusSurrogateDataset
from .metrics import field_rmse


def evaluate_ood_degradation(model: torch.nn.Module, test_dataset: ThrombusSurrogateDataset, ood_dataset: ThrombusSurrogateDataset) -> dict:
    """Returns {"test": {...}, "ood": {...}, "degradation_ratio": float},
    comparing overall field RMSE (`benchmark/metrics.field_rmse`) on the
    in-distribution test set vs. the OOD set. `degradation_ratio` > 1 means
    the surrogate is worse on OOD samples than in-distribution ones."""

    model.eval()
    results = {}
    for name, dataset in (("test", test_dataset), ("ood", ood_dataset)):
        loader = DataLoader(dataset, batch_size=len(dataset))
        with torch.no_grad():
            batch = next(iter(loader))
            pred = model(batch["params"])
        results[name] = field_rmse(pred.numpy(), batch["fields"].numpy())

    degradation_ratio = (
        results["ood"]["overall"] / results["test"]["overall"] if results["test"]["overall"] > 0 else float("inf")
    )
    return {"test": results["test"], "ood": results["ood"], "degradation_ratio": degradation_ratio}
