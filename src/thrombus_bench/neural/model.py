"""Assembled neural surrogate: encoder + operator core + physics head + uncertainty head.

Responsibility
---------------
Wire `encoder.py`, `operator_core.py`, `physics_losses.py`, and
`uncertainty.py` into a single `nn.Module` that maps (geometry params,
physiological params, inlet velocity, mesh geometry) -> predicted time
series of (velocity, pressure, viscosity, 9 species concentrations, 4
surface coverage fields), matching `configs/training.yaml`
`model.output_channels`.

This is the model `benchmark/run_benchmark.py` compares against the
mechanistic solver (`mechanistic/coupled_solver.py`).

Not yet implemented -- this is a scaffolding stub. Depends on
`encoder.py` and `operator_core.py`.
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
        self.operator_core = build_operator_core(cfg["operator_core"], out_channels=cfg["output_channels"])
        raise NotImplementedError("model.ThrombusSurrogate: not yet implemented (physics/uncertainty heads pending)")

    def forward(self, params: torch.Tensor, geometry_sdf: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns a dict of named output fields (velocity, pressure,
        viscosity, per-species concentrations, per-surface-field coverage)."""

        raise NotImplementedError
