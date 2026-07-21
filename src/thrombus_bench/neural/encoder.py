"""Geometry/parameter encoder mapping scalar inputs to a latent spatial grid.

Responsibility
---------------
Map the per-sample input (geometry parameters, physiological parameters,
inlet velocity -- `configs/training.yaml` `model.encoder.param_dim`, 8
scalars) plus a rasterized signed-distance/occupancy representation of the
mesh geometry onto a fixed-resolution latent grid
(`model.encoder.latent_grid_size`, default 64x64) that the operator core
(`operator_core.py`) consumes.

This lets the surrogate handle the two studied geometries (and interpolated
variants, per `data/sampler.py`'s geometry ranges) with a shared,
fixed-shape representation, since the mechanistic solver's unstructured
mesh varies in node count/connectivity across samples.

Not yet implemented -- this is a scaffolding stub.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeometryParamEncoder(nn.Module):
    def __init__(self, param_dim: int, latent_grid_size: tuple[int, int], hidden_channels: int, n_layers: int):
        super().__init__()
        self.param_dim = param_dim
        self.latent_grid_size = latent_grid_size
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        raise NotImplementedError("encoder.GeometryParamEncoder: not yet implemented")

    def forward(self, params: torch.Tensor, geometry_sdf: torch.Tensor) -> torch.Tensor:
        """params: (batch, param_dim); geometry_sdf: (batch, 1, H, W) ->
        (batch, hidden_channels, H, W) latent grid."""

        raise NotImplementedError
