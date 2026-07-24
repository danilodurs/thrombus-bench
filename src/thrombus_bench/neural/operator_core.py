"""Neural operator core: Fourier Neural Operator (FNO) or GNN, configurable.

Responsibility
---------------
The learned solution operator mapping the encoder's latent grid
(`encoder.py`) to a predicted field grid (velocity, species concentrations
-- `configs/training.yaml` `model.output_channels`). Two interchangeable
backbones, selected via `configs/training.yaml` `model.operator_core.type`:

* `"fno"`: a Fourier Neural Operator -- spectral convolution (truncated
  2D FFT, per-mode learned complex weights) plus a pointwise residual
  convolution per layer, following Li et al. (2021). Implemented here.
* `"gnn"`: a message-passing graph network operating directly on the FEM
  mesh graph. **Not implemented** in this project (scope note: the FNO path
  is exercised end-to-end by `train.py`/`run_benchmark.py`; the GNN path is
  left as a documented extension point -- `build_operator_core` raises
  `NotImplementedError` for `type: "gnn"`).

`build_operator_core` returns a full backbone+projection-head module
(`FourierNeuralOperator`), used by the legacy grid-projection surrogate
(`neural/model.ThrombusSurrogate`). `build_operator_backbone` returns just
the trunk (`FNOBackbone`, no head) for the continuous-surrogate path's
shared Stage 1 backbone (`neural/model.SurrogateBackbone`) -- see
`docs/continuous_surrogate_design.md`. `FourierNeuralOperator` is itself
built from an `FNOBackbone` plus a projection head, so the block
implementation exists in exactly one place either way.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """A single truncated-Fourier spectral convolution layer (Li et al. 2021).

    Transforms to Fourier space via `torch.fft.rfft2`, applies a learned
    complex-valued linear map independently to each of the lowest `modes`
    frequency components (higher frequencies are dropped -- the operator's
    implicit smoothness/resolution-invariance prior), then transforms back.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        modes = min(self.modes, h, x_ft.shape[-1])
        out_ft = torch.zeros(batch, self.out_channels, h, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :modes, :modes] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :modes, :modes], self.weight[:, :, :modes, :modes]
        )
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.pointwise(x))


class FNOBackbone(nn.Module):
    """FNO trunk only: the spectral+pointwise block stack, with no final
    output-channel projection.

    This is Stage 1's operator-core half in the continuous-surrogate design
    (`docs/continuous_surrogate_design.md`) -- both the legacy
    grid-projection `FourierNeuralOperator` below (which adds a projection
    head on top) and the new coordinate-decoder path's
    `neural/model.SurrogateBackbone` are built from this same class, so the
    block implementation exists in exactly one place regardless of which
    head consumes its `(batch, hidden_channels, H, W)` output.
    """

    def __init__(self, modes: int, hidden_channels: int, n_layers: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.blocks = nn.Sequential(*[FNOBlock(hidden_channels, modes) for _ in range(n_layers)])

    def forward(self, latent_grid: torch.Tensor) -> torch.Tensor:
        return self.blocks(latent_grid)


class FourierNeuralOperator(nn.Module):
    def __init__(self, modes: int, hidden_channels: int, n_layers: int, out_channels: int):
        super().__init__()
        self.backbone = FNOBackbone(modes, hidden_channels, n_layers)
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, latent_grid: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(latent_grid))


class GraphOperator(nn.Module):
    """Message-passing graph operator on the FEM mesh graph -- not
    implemented (see module docstring)."""

    def __init__(self, hidden_channels: int, n_message_passing_steps: int, out_channels: int):
        super().__init__()
        raise NotImplementedError(
            "operator_core.GraphOperator: not implemented in this project (scope note in module docstring); "
            "use configs/training.yaml model.operator_core.type=fno"
        )

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


def build_operator_backbone(cfg: dict) -> nn.Module:
    """Stage-1-only counterpart of `build_operator_core`: the FNO trunk
    with no output projection (`FNOBackbone`), for callers that only need
    the shared latent feature grid (`neural/model.SurrogateBackbone`).
    Never allocates an unused projection head, unlike calling
    `build_operator_core(cfg, out_channels=hidden_channels)` and discarding
    its head would."""

    if cfg["type"] == "fno":
        return FNOBackbone(**cfg["fno"])
    if cfg["type"] == "gnn":
        raise NotImplementedError(
            "operator_core.build_operator_backbone: gnn backbone not implemented (see GraphOperator docstring); "
            "use configs/training.yaml model.operator_core.type=fno"
        )
    raise ValueError(f"Unknown operator_core.type: {cfg['type']!r}")
