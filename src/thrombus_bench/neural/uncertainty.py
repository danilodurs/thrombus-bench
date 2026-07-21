"""Uncertainty quantification wrapper: deep ensemble or MC-dropout.

Responsibility
---------------
Wrap a `neural/model.py` `ThrombusSurrogate` to produce a predictive mean
and variance per output field, selected via `configs/training.yaml`
`model.uncertainty.method`:

* `"deep_ensemble"`: train `n_ensemble_members` independently-initialized
  copies of the full model; predictive mean/variance from the ensemble.
* `"mc_dropout"`: keep dropout active at inference time and draw
  `mc_dropout_n_samples` stochastic forward passes.

Consumed by `benchmark/calibration.py` to check whether predicted variance
tracks actual error (reliability diagrams).

Not yet implemented -- this is a scaffolding stub.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepEnsemble:
    def __init__(self, model_factory, n_members: int):
        self.models = [model_factory() for _ in range(n_members)]
        raise NotImplementedError("uncertainty.DeepEnsemble: not yet implemented")

    def predict(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, variance) across ensemble members."""

        raise NotImplementedError


class MCDropoutWrapper:
    def __init__(self, model: nn.Module, n_samples: int, dropout_rate: float):
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        raise NotImplementedError("uncertainty.MCDropoutWrapper: not yet implemented")

    def predict(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, variance) across stochastic forward passes."""

        raise NotImplementedError
