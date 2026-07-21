"""Uncertainty quantification wrapper: deep ensemble or MC-dropout.

Responsibility
---------------
Wrap `neural/model.py` `ThrombusSurrogate` instances to produce a predictive
mean and variance per output field, selected via `configs/training.yaml`
`model.uncertainty.method`:

* `"deep_ensemble"`: `n_members` independently-initialized copies of the
  full model; predictive mean/variance from the ensemble's per-member
  predictions.
* `"mc_dropout"`: keep dropout active at inference time and draw
  `n_samples` stochastic forward passes.

Consumed by `benchmark/calibration.py` to check whether predicted variance
tracks actual error (reliability diagrams).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class DeepEnsemble:
    def __init__(self, model_factory: Callable[[], nn.Module], n_members: int):
        self.models = [model_factory() for _ in range(n_members)]

    def to(self, device) -> "DeepEnsemble":
        self.models = [m.to(device) for m in self.models]
        return self

    def predict(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, variance) across ensemble members, each called
        with identical `*args, **kwargs`."""

        preds = torch.stack([m(*args, **kwargs) for m in self.models], dim=0)
        return preds.mean(dim=0), preds.var(dim=0, unbiased=False)


def _enable_mc_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


class MCDropoutWrapper:
    def __init__(self, model: nn.Module, n_samples: int, dropout_rate: float):
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def predict(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, variance) across `n_samples` stochastic forward
        passes (dropout kept active; all other layers in eval mode)."""

        self.model.eval()
        _enable_mc_dropout(self.model)
        with torch.no_grad():
            preds = torch.stack([self.model(*args, **kwargs) for _ in range(self.n_samples)], dim=0)
        return preds.mean(dim=0), preds.var(dim=0, unbiased=False)
