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

Optional 12th channel: `M_at_wall` prediction
-------------------------------------------------
`cfg["predict_M_at_wall"]` (default `False`, so existing checkpoints/configs
keep working unchanged) adds one extra output channel predicting
`data/dataset.py`'s `M_at_wall` (the rasterized wall-band `surface_ode.
SurfaceState.M_at`, needed for `benchmark/metrics.thrombus_mask`/
`thrombus_iou`) alongside the usual `output_channels` physical-field
channels -- `forward()` still returns a single `(batch, output_channels [+1],
H, W)` tensor, ordered `[<output_channels physical fields>, M_at_wall]`.
A single extra channel (rather than a separate decoder head returning a
dict/tuple) was chosen because `encoder.py`/`operator_core.py` are already
generic in channel count, so this needs no changes there, and it preserves
`forward()`'s single-tensor contract that `neural/uncertainty.py`
(MC-dropout/deep-ensemble), `calibration.py`, `edge_holdout_eval.py`, and
`neural/baselines.py` all assume -- a dict/tuple return would have required
updating every one of those call sites instead of just
`neural/train.py`/`benchmark/run_benchmark.py` (see those modules for how
the extra channel is split back out before/after use).
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
        self.predict_M_at_wall = bool(cfg.get("predict_M_at_wall", False))
        out_channels = cfg["output_channels"] + (1 if self.predict_M_at_wall else 0)
        self.operator_core = build_operator_core(cfg["operator_core"], out_channels=out_channels)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """params: (batch, param_dim) -> (batch, output_channels [+1 if
        predict_M_at_wall], H, W)."""

        latent = self.encoder(params)
        latent = self.dropout(latent)
        return self.operator_core(latent)
