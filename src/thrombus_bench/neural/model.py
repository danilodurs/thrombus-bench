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

Stage 1 / Stage 2 split (`docs/continuous_surrogate_design.md`)
--------------------------------------------------------------------
`SurrogateBackbone` factors out exactly the "encoder + FNO trunk, no
output projection" computation (Stage 1) that both this module's
`ThrombusSurrogate` (legacy grid-projection head, kept fully intact as a
comparison baseline) and `coordinate_decoder.ContinuousThrombusSurrogate`
(new coordinate-decoder head) need -- `ThrombusSurrogate` is now literally
`SurrogateBackbone` + a `Conv2d` projection head, so the encoder/FNO
implementation exists in exactly one place. See `operator_core.py`'s
`FNOBackbone`/`build_operator_backbone` for the corresponding split on the
operator-core side.

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
from .operator_core import build_operator_backbone


class SurrogateBackbone(nn.Module):
    """Stage 1: `GeometryParamEncoder` + the FNO trunk, stopping before any
    output-channel projection -- see module docstring "Stage 1 / Stage 2
    split". `cfg` is the same `model` config block `ThrombusSurrogate`
    takes (only its `encoder`/`operator_core`/`uncertainty` keys are used
    here; `output_channels`/`predict_M_at_wall` are a head-level concern,
    not this backbone's)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder = GeometryParamEncoder(**cfg["encoder"])
        self.dropout = nn.Dropout2d(p=cfg.get("uncertainty", {}).get("mc_dropout_rate", 0.1))
        self.backbone = build_operator_backbone(cfg["operator_core"])
        self.hidden_channels = cfg["encoder"]["hidden_channels"]

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """params: (batch, param_dim) -> (batch, hidden_channels, H_latent,
        W_latent) latent feature grid (no output projection)."""

        latent = self.encoder(params)
        latent = self.dropout(latent)
        return self.backbone(latent)


class ThrombusSurrogate(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = SurrogateBackbone(cfg)
        self.predict_M_at_wall = bool(cfg.get("predict_M_at_wall", False))
        out_channels = cfg["output_channels"] + (1 if self.predict_M_at_wall else 0)
        self.head = nn.Conv2d(self.backbone.hidden_channels, out_channels, kernel_size=1)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """params: (batch, param_dim) -> (batch, output_channels [+1 if
        predict_M_at_wall], H, W)."""

        return self.head(self.backbone(params))
