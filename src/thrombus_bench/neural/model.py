"""Assembled neural surrogate: encoder + operator core.

Responsibility
---------------
Wire `encoder.py` and `operator_core.py` into a single `nn.Module` that
maps the 8-scalar parameter vector (geometry + physiological parameters +
inlet velocity, `data/generate_dataset.PARAM_ORDER`) to the predicted
field grid (velocity x/y + 9 species concentrations, `data/dataset.FIELD_NAMES`,
`configs/training.yaml` `model.output_channels`).

This is the model `benchmark/run_benchmark.py` compares against the
mechanistic solver (`mechanistic/coupled_solver.py`). Dropout (for
`neural/uncertainty.py`'s MC-dropout path) is inserted between the encoder
and operator core.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import GeometryParamEncoder
from .operator_core import build_operator_core


class ThrombusSurrogate(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.encoder = GeometryParamEncoder(**cfg["encoder"])
        self.dropout = nn.Dropout2d(p=cfg.get("uncertainty", {}).get("mc_dropout_rate", 0.1))
        self.operator_core = build_operator_core(cfg["operator_core"], out_channels=cfg["output_channels"])

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """params: (batch, param_dim) -> (batch, output_channels, H, W)."""

        latent = self.encoder(params)
        latent = self.dropout(latent)
        return self.operator_core(latent)
