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

Not yet implemented -- this is a scaffolding stub. Depends on
`benchmark/metrics.py` and `data/dataset.py`.
"""

from __future__ import annotations


def evaluate_ood_degradation(model, test_dataset, ood_dataset) -> dict:
    """Returns a dict of {metric_name: {"test": v, "ood": v, "degradation": v}}."""

    raise NotImplementedError("ood_eval.evaluate_ood_degradation: not yet implemented")
