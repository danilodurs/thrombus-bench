"""Geometry/parameter encoder mapping scalar inputs to a latent spatial grid.

Responsibility
---------------
Map the per-sample input parameter vector (geometry + physiological
parameters + inlet velocity -- `configs/training.yaml`
`model.encoder.param_dim`, 8 scalars matching
`data/generate_dataset.PARAM_ORDER`) onto a fixed-resolution latent grid
(`model.encoder.latent_grid_size`) that the operator core (`operator_core.py`)
consumes.

Design: an MLP maps the parameter vector to a per-channel bias/scale
(FiLM-style modulation), applied to a learned constant base grid
concatenated with a fixed sinusoidal coordinate embedding (so the network
has explicit access to spatial position), followed by a couple of
convolutions. This lets the surrogate handle the sampled geometry/physio
range with a shared, fixed-shape representation, since the mechanistic
solver's mesh varies in node count/connectivity across samples (already
rasterized onto this fixed grid by `data/generate_dataset.py`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _coordinate_grid(h: int, w: int) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, h)
    xs = torch.linspace(-1.0, 1.0, w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=0)  # (2, H, W)


class GeometryParamEncoder(nn.Module):
    def __init__(self, param_dim: int, latent_grid_size: tuple[int, int], hidden_channels: int, n_layers: int):
        super().__init__()
        self.param_dim = param_dim
        self.latent_grid_size = tuple(latent_grid_size)
        self.hidden_channels = hidden_channels

        self.param_mlp = nn.Sequential(
            nn.Linear(param_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 2 * hidden_channels),  # scale, bias for FiLM
        )
        self.register_buffer("coord_grid", _coordinate_grid(*self.latent_grid_size), persistent=False)
        self.base = nn.Parameter(torch.zeros(hidden_channels, *self.latent_grid_size))
        self.coord_proj = nn.Conv2d(2, hidden_channels, kernel_size=1)

        convs = []
        for _ in range(max(1, n_layers)):
            convs.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            convs.append(nn.SiLU())
        self.convs = nn.Sequential(*convs)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """params: (batch, param_dim) -> (batch, hidden_channels, H, W) latent grid."""

        batch = params.shape[0]
        film = self.param_mlp(params)  # (batch, 2*hidden_channels)
        scale, bias = film.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)

        coord_feat = self.coord_proj(self.coord_grid.unsqueeze(0))  # (1, hidden, H, W)
        grid = self.base.unsqueeze(0) + coord_feat  # (1, hidden, H, W)
        grid = grid.expand(batch, -1, -1, -1)

        modulated = grid * (1.0 + scale) + bias
        return self.convs(modulated)
