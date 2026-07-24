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

Continuous-path baselines (`ContinuousMeanFieldBaseline`,
`ContinuousNearestNeighborBaseline`, Phase 6, `docs/
continuous_surrogate_design.md`)
------------------------------------------------------------------------
Same purpose, adapted for a point query `(params_with_time, x, y)` instead
of a fixed raster -- same call signature as `neural.coordinate_decoder.
ContinuousThrombusSurrogate.forward` (`forward(params_with_time,
query_points_m, batch_index, geometry_mm) -> (total_points,
output_channels)`), fit on `data.dataset.PointCloudThrombusDataset`-like
objects, operating on `fields`' log-compressed values the same way (see
above).

Both pool training points in **normalized** `[-1, 1]` coordinate space
(`neural.coordinate_decoder.normalize_query_points_to_unit_box`, the exact
convention `ContinuousThrombusSurrogate` itself uses), not raw physical
meters -- different sampled geometries have different physical extents, so
"the same relative position along the vessel" is only comparable across
samples once normalized to each sample's own bounding box; raw meters
would also let a sample with a large aneurysm dominate a Euclidean
distance search over one with a small one.

* `ContinuousMeanFieldBaseline`: still position-aware but conditioning-
  blind, matching the grid version's "ignores params, keeps spatial
  structure" spirit -- pools every training `(sample, checkpoint, node)`
  row's `(x_norm, y_norm)` and field value (`params_with_time` is never
  read, in fit or forward), and predicts the mean field value of the `k`
  nearest pooled points to a query location, via a single
  `scipy.spatial.cKDTree` (chosen over "rasterize a low-res mean grid,
  then look up the nearest cell," the design summary's other suggested
  option, specifically because point-cloud training data has no raster
  ground truth lying around any more -- Phase 3 deliberately stopped
  generating it by default -- so that option would mean re-introducing the
  same `griddata` interpolation cost this design moved away from, just to
  build a baseline; pooling scattered points directly needs no
  regridding at all).
* `ContinuousNearestNeighborBaseline`: a single combined-distance KNN
  (`k=1`) over `[params_with_time (9) ; x_norm, y_norm (2)]` -- an 11-dim
  Euclidean nearest-neighbor search across every training `(sample,
  checkpoint, node)` row, rather than a two-stage "nearest sample, then
  nearest node within it" lookup. Chosen for simplicity: a two-stage
  lookup risks a subtly wrong result when the nearest-by-*parameters*
  sample happens not to have any node particularly close to the query
  location, which a single combined search sidesteps entirely; concatenating
  directly (no extra relative weighting) is reasonable since both groups
  are already independently normalized to `[-1, 1]`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from .coordinate_decoder import VESSEL_LENGTH_MM, normalize_query_points_to_unit_box


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


class ContinuousMeanFieldBaseline(nn.Module):
    """Point-query counterpart of `MeanFieldBaseline` -- see module
    docstring. `n_neighbors`: how many pooled training points to average
    per query (a larger value is a smoother/less noisy "mean field" at the
    cost of more blurring near sharp local features; 20 is a reasonable
    default at this project's demo-scale dataset sizes)."""

    def __init__(self, n_neighbors: int = 20):
        super().__init__()
        self.n_neighbors = n_neighbors
        self._tree: cKDTree | None = None
        self.register_buffer("train_fields", torch.zeros(0), persistent=False)

    def fit(self, train_dataset) -> "ContinuousMeanFieldBaseline":
        """Pool every training `(sample, checkpoint, node)` row's
        normalized `(x, y)` position and field value. `train_dataset`:
        anything supporting `len()`/`__getitem__` returning a dict with
        `node_coords`, `fields`, `geometry_mm` (e.g. `data.dataset.
        PointCloudThrombusDataset`)."""

        all_coords, all_fields = [], []
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            n = item["node_coords"].shape[0]
            batch_index = torch.zeros(n, dtype=torch.long)
            geometry_mm = item["geometry_mm"].unsqueeze(0)
            coords_norm = normalize_query_points_to_unit_box(
                item["node_coords"], batch_index, geometry_mm, VESSEL_LENGTH_MM
            )
            all_coords.append(coords_norm.numpy())
            all_fields.append(item["fields"].numpy())

        pooled_coords = np.concatenate(all_coords, axis=0)
        self.train_fields = torch.from_numpy(np.concatenate(all_fields, axis=0))
        self._tree = cKDTree(pooled_coords)
        return self

    def forward(
        self,
        params_with_time: torch.Tensor,
        query_points_m: torch.Tensor,
        batch_index: torch.Tensor,
        geometry_mm: torch.Tensor,
    ) -> torch.Tensor:
        """Same call signature as `ContinuousThrombusSurrogate.forward` --
        `params_with_time` is never read (see module docstring). Returns
        `(total_points, output_channels)`, each row the mean field value
        of the `n_neighbors` nearest pooled training points to that query's
        normalized position."""

        if self._tree is None:
            raise RuntimeError("ContinuousMeanFieldBaseline.forward called before fit()")
        query_norm = normalize_query_points_to_unit_box(query_points_m, batch_index, geometry_mm, VESSEL_LENGTH_MM)
        k = min(self.n_neighbors, self.train_fields.shape[0])
        _, idx = self._tree.query(query_norm.detach().cpu().numpy(), k=k)
        if k == 1:
            idx = idx[:, None]
        neighbor_fields = self.train_fields[torch.from_numpy(idx)]  # (total_points, k, C)
        return neighbor_fields.mean(dim=1).to(query_points_m.device)


class ContinuousNearestNeighborBaseline(nn.Module):
    """Point-query counterpart of `NearestNeighborBaseline` -- see module
    docstring: a single combined-distance `k=1` nearest-neighbor search
    over `[params_with_time ; x_norm, y_norm]`."""

    def __init__(self):
        super().__init__()
        self._tree: cKDTree | None = None
        self.register_buffer("train_fields", torch.zeros(0), persistent=False)

    def fit(self, train_dataset) -> "ContinuousNearestNeighborBaseline":
        """Pool every training `(sample, checkpoint, node)` row's
        `[params_with_time ; normalized (x, y)]` feature vector and field
        value. Same duck-typed `train_dataset` contract as
        `ContinuousMeanFieldBaseline.fit`, plus a `params_with_time`
        entry."""

        all_features, all_fields = [], []
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            n = item["node_coords"].shape[0]
            batch_index = torch.zeros(n, dtype=torch.long)
            geometry_mm = item["geometry_mm"].unsqueeze(0)
            coords_norm = normalize_query_points_to_unit_box(
                item["node_coords"], batch_index, geometry_mm, VESSEL_LENGTH_MM
            )
            params_broadcast = item["params_with_time"].unsqueeze(0).expand(n, -1)
            features = torch.cat([params_broadcast, coords_norm], dim=-1)
            all_features.append(features.numpy())
            all_fields.append(item["fields"].numpy())

        pooled_features = np.concatenate(all_features, axis=0)
        self.train_fields = torch.from_numpy(np.concatenate(all_fields, axis=0))
        self._tree = cKDTree(pooled_features)
        return self

    def forward(
        self,
        params_with_time: torch.Tensor,
        query_points_m: torch.Tensor,
        batch_index: torch.Tensor,
        geometry_mm: torch.Tensor,
    ) -> torch.Tensor:
        """Returns `(total_points, output_channels)`, each row the nearest
        pooled training point's field by Euclidean distance in the
        combined `[params_with_time ; x_norm, y_norm]` space."""

        if self._tree is None:
            raise RuntimeError("ContinuousNearestNeighborBaseline.forward called before fit()")
        query_norm = normalize_query_points_to_unit_box(query_points_m, batch_index, geometry_mm, VESSEL_LENGTH_MM)
        params_per_point = params_with_time[batch_index]
        query_features = torch.cat([params_per_point, query_norm], dim=-1)
        _, idx = self._tree.query(query_features.detach().cpu().numpy(), k=1)
        return self.train_fields[torch.from_numpy(idx)].to(query_points_m.device)
