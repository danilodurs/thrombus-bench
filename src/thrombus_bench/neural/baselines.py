"""Trivial accuracy baselines for `benchmark/run_benchmark.py`.

Responsibility
---------------
Two "models" with the same call signature as `neural/model.py`
`ThrombusSurrogate.forward` (`forward(params) -> fields`, shape
`(batch, output_channels, H, W)`), so `benchmark/run_benchmark.py` can swap
them in as drop-in comparisons -- without a baseline, there's no way to
tell whether the FNO surrogate is adding value over something trivial:

* `MeanFieldBaseline`: ignores `params` entirely at inference and always
  returns the per-channel, per-pixel mean of the training split's target
  fields (a "smartest constant" predictor -- any model worth its training
  cost should beat this).
* `NearestNeighborBaseline`: memorizes every training sample's normalized
  param vector (`data/sampler.normalize_params`, see `data/dataset.py`'s
  `ThrombusSurrogateDataset`, which already returns `"params"` normalized
  this way -- reused here rather than re-normalizing) and target field; at
  inference, looks up and returns the nearest training sample's field by
  Euclidean distance in that normalized parameter space (a "lookup table"
  predictor -- a model that's actually learned the underlying mapping,
  rather than memorizing training points, should generalize better between
  samples than this).

Both operate directly on `ThrombusSurrogateDataset`'s `"fields"` tensors,
i.e. the same log-compressed space (`data/dataset.field_to_log`) as
`ThrombusSurrogate`'s raw output and training target -- no additional
transform here -- so `benchmark/metrics.field_rmse` comparisons across all
three models are apples-to-apples.

Neither baseline has learnable parameters or a training loop; both are
still thin `nn.Module` subclasses (storing state via `register_buffer`, not
`nn.Parameter`) purely so `.to(device)`/state-dict semantics stay uniform
with `ThrombusSurrogate` if this project ever runs on GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MeanFieldBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean_field", torch.zeros(0), persistent=False)

    def fit(self, train_dataset) -> "MeanFieldBaseline":
        """Compute and store the per-channel, per-pixel mean field across
        every sample in `train_dataset` (any object supporting `len()` and
        `__getitem__` returning a dict with a `"fields"` tensor, e.g.
        `data/dataset.py`'s `ThrombusSurrogateDataset`)."""

        fields = torch.stack([train_dataset[i]["fields"] for i in range(len(train_dataset))], dim=0)
        self.mean_field = fields.mean(dim=0)
        return self

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """params: (batch, param_dim) -> (batch, C, H, W). `params`'s values
        are never read, only its batch size -- this baseline is constant."""

        if self.mean_field.numel() == 0:
            raise RuntimeError("MeanFieldBaseline.forward called before fit()")
        batch = params.shape[0]
        return self.mean_field.unsqueeze(0).expand(batch, *self.mean_field.shape).to(params.device)


class NearestNeighborBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("train_params", torch.zeros(0), persistent=False)
        self.register_buffer("train_fields", torch.zeros(0), persistent=False)

    def fit(self, train_dataset) -> "NearestNeighborBaseline":
        """Store every training sample's normalized param vector and target
        field (same duck-typed `train_dataset` contract as
        `MeanFieldBaseline.fit`, plus a `"params"` entry) for the
        nearest-neighbor lookup in `forward`."""

        n = len(train_dataset)
        self.train_params = torch.stack([train_dataset[i]["params"] for i in range(n)], dim=0)
        self.train_fields = torch.stack([train_dataset[i]["fields"] for i in range(n)], dim=0)
        return self

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """params: (batch, param_dim), already normalized like the stored
        `train_params` -> (batch, C, H, W), each row the nearest training
        sample's field by Euclidean distance in normalized parameter
        space."""

        if self.train_params.numel() == 0:
            raise RuntimeError("NearestNeighborBaseline.forward called before fit()")
        distances = torch.cdist(params.to(self.train_params.dtype), self.train_params)  # (batch, n_train)
        nearest = distances.argmin(dim=1)
        return self.train_fields[nearest].to(params.device)
