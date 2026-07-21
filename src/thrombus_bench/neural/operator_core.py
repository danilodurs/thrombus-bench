"""Neural operator core: Fourier Neural Operator (FNO) or GNN, configurable.

Responsibility
---------------
The learned solution operator mapping the encoder's latent grid
(`encoder.py`) to a predicted field grid (velocity, viscosity, species
concentrations, surface coverage -- `configs/training.yaml`
`model.output_channels`). Two interchangeable backbones, selected via
`configs/training.yaml` `model.operator_core.type`:

* `"fno"`: a Fourier Neural Operator (spectral convolutions in Fourier
  space, resolution-invariant), appropriate given the encoder already
  rasterizes onto a fixed regular grid.
* `"gnn"`: a message-passing graph network operating directly on the FEM
  mesh graph (nodes = mesh vertices, edges = mesh connectivity from
  `mechanistic/mesh.py`), avoiding the rasterization step at the cost of
  needing per-sample graph batching.

Not yet implemented -- this is a scaffolding stub.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FourierNeuralOperator(nn.Module):
    def __init__(self, modes: int, hidden_channels: int, n_layers: int, out_channels: int):
        super().__init__()
        self.modes = modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.out_channels = out_channels
        raise NotImplementedError("operator_core.FourierNeuralOperator: not yet implemented")

    def forward(self, latent_grid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GraphOperator(nn.Module):
    def __init__(self, hidden_channels: int, n_message_passing_steps: int, out_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_message_passing_steps = n_message_passing_steps
        self.out_channels = out_channels
        raise NotImplementedError("operator_core.GraphOperator: not yet implemented")

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_operator_core(cfg: dict, out_channels: int) -> nn.Module:
    """Factory selecting FNO vs GNN per `cfg['type']` (configs/training.yaml
    `model.operator_core`)."""

    if cfg["type"] == "fno":
        return FourierNeuralOperator(out_channels=out_channels, **cfg["fno"])
    if cfg["type"] == "gnn":
        return GraphOperator(out_channels=out_channels, **cfg["gnn"])
    raise ValueError(f"Unknown operator_core.type: {cfg['type']!r}")
