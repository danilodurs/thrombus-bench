"""Stage 2 (coordinate decoder head) + the continuous-surrogate top-level
model, per `docs/continuous_surrogate_design.md`.

Responsibility
---------------
`CoordinateDecoder` maps a continuous query point's local information --
Phase 1's Fourier-feature-encoded position (`coordinate_encoding.py`), a
bilinearly-interpolated slice of Stage 1's latent feature grid at that
point (`torch.nn.functional.grid_sample`), and Phase 1's analytic
signed-distance-to-wall value (`mechanistic/geometry_sdf.py`) -- to the
physical field values at exactly that point, in `data/dataset.FIELD_NAMES`
channel order.

`ContinuousThrombusSurrogate` wires `neural/model.SurrogateBackbone`
(Stage 1, shared with the legacy `ThrombusSurrogate`, unchanged) to
`CoordinateDecoder` (Stage 2), and owns everything that needs each
sample's *raw* (unnormalized) geometry rather than the encoder's
normalized inputs: per-sample coordinate normalization and the analytic
SDF evaluation.

Coordinate normalization convention
--------------------------------------
Query points arrive as raw physical `(x, y)` in meters. Both `grid_sample`
and Phase 1's `FourierFeatureEncoding` expect `[-1, 1]`-normalized input
(matching `encoder.py`'s `_coordinate_grid` convention -- `-1`/`+1` are the
corner pixel *centers*, hence `grid_sample(..., align_corners=True)`
below). Each sample's own analytic bounding box is used for this
normalization -- `x in [0, L]`, `y in [0, D + R]` (`mesh.py`'s domain
always starts at the origin; `vessel_length_mm` is fixed at
`VESSEL_LENGTH_MM` everywhere in this project, never a sampled parameter,
see `data/generate_dataset.py::_run_one_sample`) -- rather than a single
shared physical scale across samples, since different samples' domains
have different physical extents (in particular different `D + R` sac
heights) and the legacy raster path (`generate_dataset._rasterize`)
already used this same per-sample-bounding-box convention.

Batching convention for ragged query points
------------------------------------------------
A batch mixes samples that may each contribute a different number of
query points (e.g. all of one sample's mesh nodes vs. a fixed random
subsample of another's). Following the standard flat-array-plus-index
pattern for variable-cardinality batching (the same idea PyTorch Geometric
uses for batching graphs, without needing that library): query points are
one flat `(total_points, 2)` tensor, and `batch_index: (total_points,)`
maps each point to which of the `batch` samples it belongs to.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dataset import FIELD_NAMES
from ..mechanistic.geometry_sdf import signed_distance_to_wall
from ..mechanistic.mesh import GeometryConfig
from .coordinate_encoding import DEFAULT_N_FREQUENCIES, FourierFeatureEncoding
from .model import SurrogateBackbone

# vessel_length_mm is fixed everywhere in this project (never a sampled
# parameter -- see data/sampler.DEFAULT_RANGES, which has no entry for it,
# and every geometry preset/generation call site hardcoding 50.0).
VESSEL_LENGTH_MM = 50.0

DEFAULT_MLP_HIDDEN = 128
DEFAULT_N_RESIDUAL_BLOCKS = 3


class _ResidualMLPBlock(nn.Module):
    """Two-layer MLP block with a residual (skip) connection, SiLU
    activations. A plain deep MLP over a coordinate-derived input is
    known to be hard to optimize (vanishing-gradient-like degradation with
    depth, same motivation as ResNet); a couple of residual blocks is a
    standard, low-risk fix and is cheap at this model's scale."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


def _grid_sample_per_sample(
    latent_grid: torch.Tensor, query_points_norm: torch.Tensor, batch_index: torch.Tensor
) -> torch.Tensor:
    """Bilinearly-interpolated Stage-1 features at each query point, from
    its own sample's latent grid.

    `latent_grid`: `(batch, C, H, W)`. `query_points_norm`: `(total_points,
    2)`, already normalized to `[-1, 1]`. `batch_index`: `(total_points,)`,
    ragged counts per sample allowed. Returns `(total_points, C)`.

    Loops over samples (not points): `F.grid_sample` requires the same
    leading batch dimension on `input` and `grid`, but a ragged batch's
    samples contribute different numbers of query points -- there is no
    single `(batch, H_out, W_out, 2)` grid shape representing a variable
    point count per sample without padding. Looping over the (typically
    small, e.g. `optim.batch_size: 16`) sample count instead avoids wasted
    computation on pad points and is exact for every sample regardless of
    its point count.
    """

    batch, channels = latent_grid.shape[0], latent_grid.shape[1]
    out = query_points_norm.new_zeros(query_points_norm.shape[0], channels)
    for b in range(batch):
        mask = batch_index == b
        if not torch.any(mask):
            continue
        grid = query_points_norm[mask].view(1, 1, -1, 2)
        sampled = F.grid_sample(latent_grid[b : b + 1], grid, mode="bilinear", padding_mode="border", align_corners=True)
        out[mask] = sampled.view(channels, -1).transpose(0, 1)
    return out


def normalize_query_points_to_unit_box(
    query_points_m: torch.Tensor, batch_index: torch.Tensor, geometry_mm: torch.Tensor, vessel_length_mm: float
) -> torch.Tensor:
    """Per-sample bounding-box normalization to `[-1, 1]` -- see module
    docstring's "Coordinate normalization convention". Factored out of
    `ContinuousThrombusSurrogate.forward` so other code that needs the same
    normalized coordinates (e.g. `neural/baselines.py`'s continuous
    baselines, Phase 6) doesn't reimplement it separately.

    `query_points_m`: `(total_points, 2)` raw physical `(x, y)` meters.
    `batch_index`: `(total_points,)`. `geometry_mm`: `(batch, 2)` raw
    `[aneurysm_diameter_mm, vessel_diameter_mm]`. Returns `(total_points,
    2)` in `[-1, 1]`.
    """

    D_m = geometry_mm[:, 1] * 1.0e-3
    R_m = geometry_mm[:, 0] * 0.5e-3
    L_m = query_points_m.new_full(D_m.shape, vessel_length_mm * 1.0e-3)

    D_pt, R_pt, L_pt = D_m[batch_index], R_m[batch_index], L_m[batch_index]
    x_norm = 2.0 * query_points_m[:, 0] / L_pt - 1.0
    y_norm = 2.0 * query_points_m[:, 1] / (D_pt + R_pt) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def _sdf_per_point(
    query_points_m: torch.Tensor, batch_index: torch.Tensor, geometry_mm: torch.Tensor, vessel_length_mm: float
) -> torch.Tensor:
    """Analytic signed distance (Phase 1's `geometry_sdf.
    signed_distance_to_wall`, reused exactly rather than reimplemented) at
    every query point, using each point's own sample's geometry.

    Looped per unique sample rather than vectorized across the whole
    ragged batch: `signed_distance_to_wall` takes one `GeometryConfig` for
    its whole input array, and different samples in a batch generally have
    different geometries (sampled `aneurysm_diameter_mm`/
    `vessel_diameter_mm`) -- batch sizes here are small (e.g.
    `optim.batch_size: 16`), so this is cheap.

    Runs entirely on CPU via numpy and returns a plain (non-`requires_grad`)
    tensor -- consistent with the SDF having no learnable parameters and
    needing no gradient (see `test_coordinate_decoder.py`'s gradient smoke
    test for confirmation this doesn't block gradient flow to anything
    else).
    """

    device, dtype = query_points_m.device, query_points_m.dtype
    xs_np = query_points_m[:, 0].detach().cpu().numpy()
    ys_np = query_points_m[:, 1].detach().cpu().numpy()
    batch_np = batch_index.detach().cpu().numpy()
    geometry_np = geometry_mm.detach().cpu().numpy()

    out = np.zeros(xs_np.shape[0], dtype=np.float32)
    for b in np.unique(batch_np):
        mask = batch_np == b
        geom = GeometryConfig(
            vessel_diameter_mm=float(geometry_np[b, 1]),
            aneurysm_diameter_mm=float(geometry_np[b, 0]),
            vessel_length_mm=vessel_length_mm,
        )
        out[mask] = signed_distance_to_wall(xs_np[mask], ys_np[mask], geom)
    return torch.from_numpy(out).to(device=device, dtype=dtype)


class CoordinateDecoder(nn.Module):
    """Stage 2: per-query-point MLP head (see module docstring)."""

    def __init__(
        self,
        hidden_channels: int,
        output_channels: int = len(FIELD_NAMES),
        n_frequencies: int = DEFAULT_N_FREQUENCIES,
        mlp_hidden: int = DEFAULT_MLP_HIDDEN,
        n_residual_blocks: int = DEFAULT_N_RESIDUAL_BLOCKS,
    ):
        super().__init__()
        self.coord_encoding = FourierFeatureEncoding(n_coords=2, n_frequencies=n_frequencies)
        in_dim = self.coord_encoding.output_dim + hidden_channels + 1  # +1: SDF
        self.input_proj = nn.Linear(in_dim, mlp_hidden)
        self.input_act = nn.SiLU()
        self.blocks = nn.ModuleList([_ResidualMLPBlock(mlp_hidden) for _ in range(n_residual_blocks)])
        self.output_proj = nn.Linear(mlp_hidden, output_channels)

    def forward(
        self,
        latent_grid: torch.Tensor,
        query_points_norm: torch.Tensor,
        batch_index: torch.Tensor,
        sdf_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """`latent_grid`: `(batch, hidden_channels, H, W)`.
        `query_points_norm`: `(total_points, 2)`, in `[-1, 1]`.
        `batch_index`: `(total_points,)`. `sdf_normalized`: `(total_points,)`
        dimensionless (already scaled by the caller -- see
        `ContinuousThrombusSurrogate`). Returns `(total_points,
        output_channels)`.
        """

        pos_features = self.coord_encoding(query_points_norm)
        sampled_features = _grid_sample_per_sample(latent_grid, query_points_norm, batch_index)
        x = torch.cat([pos_features, sampled_features, sdf_normalized.unsqueeze(-1)], dim=-1)
        x = self.input_act(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class ContinuousThrombusSurrogate(nn.Module):
    """Stage 1 (`SurrogateBackbone`) + Stage 2 (`CoordinateDecoder`) wired
    together (see module docstring). Predicts the same physical fields as
    `ThrombusSurrogate`/`data.dataset.FIELD_NAMES`, but at arbitrary
    continuous query points instead of a fixed raster grid.
    """

    def __init__(self, cfg: dict):
        """`cfg` is the `model` config block (matching `ThrombusSurrogate`'s
        convention -- `encoder`/`operator_core`/`uncertainty` are shared
        with that path via `SurrogateBackbone`; `output_channels`/
        `predict_M_at_wall` mean exactly what they do there too, see
        `neural/model.py`'s docstring). Continuous-specific keys, all
        optional with defaults:

        - `coordinate_encoding.num_frequency_bands` (int, default
          `coordinate_encoding.DEFAULT_N_FREQUENCIES`): Phase 1's Fourier
          feature encoding `L`.
        - `coordinate_decoder.mlp_hidden`/`n_residual_blocks`: `
          CoordinateDecoder`'s MLP width/depth.
        - `vessel_length_mm` (default `VESSEL_LENGTH_MM`): see module
          docstring -- fixed everywhere in this project, never sampled.
        """

        super().__init__()
        self.backbone = SurrogateBackbone(cfg)
        coord_enc_cfg = cfg.get("coordinate_encoding", {})
        coord_dec_cfg = cfg.get("coordinate_decoder", {})
        self.predict_M_at_wall = bool(cfg.get("predict_M_at_wall", False))
        output_channels = cfg["output_channels"] + (1 if self.predict_M_at_wall else 0)
        self.decoder = CoordinateDecoder(
            hidden_channels=self.backbone.hidden_channels,
            output_channels=output_channels,
            n_frequencies=coord_enc_cfg.get("num_frequency_bands", DEFAULT_N_FREQUENCIES),
            mlp_hidden=coord_dec_cfg.get("mlp_hidden", DEFAULT_MLP_HIDDEN),
            n_residual_blocks=coord_dec_cfg.get("n_residual_blocks", DEFAULT_N_RESIDUAL_BLOCKS),
        )
        self.vessel_length_mm = cfg.get("vessel_length_mm", VESSEL_LENGTH_MM)

    def forward(
        self,
        params_with_time: torch.Tensor,
        query_points_m: torch.Tensor,
        batch_index: torch.Tensor,
        geometry_mm: torch.Tensor,
    ) -> torch.Tensor:
        """
        `params_with_time`: `(batch, 9)` -- `SurrogateBackbone`/encoder
        input, already normalized (existing 8 `data/generate_dataset.
        PARAM_ORDER` scalars + normalized time, per the design summary).
        `query_points_m`: `(total_points, 2)` -- raw physical `(x, y)` in
        meters. `batch_index`: `(total_points,)` long, values in `[0,
        batch)`, ragged counts per sample allowed. `geometry_mm`: `(batch,
        2)` -- raw (unnormalized) `[aneurysm_diameter_mm,
        vessel_diameter_mm]` per sample, needed here (not derivable from
        `params_with_time`, which is normalized) for the analytic SDF and
        per-sample bounding-box coordinate normalization.

        Returns `(total_points, output_channels)`.
        """

        latent = self.backbone(params_with_time)

        query_points_norm = normalize_query_points_to_unit_box(
            query_points_m, batch_index, geometry_mm, self.vessel_length_mm
        )

        D_pt = geometry_mm[:, 1][batch_index] * 1.0e-3
        sdf_raw = _sdf_per_point(query_points_m, batch_index, geometry_mm, self.vessel_length_mm)
        # Dimensionless-scale the SDF by the vessel diameter (this domain's
        # natural length scale, already used the same way elsewhere in the
        # codebase, e.g. coupled_solver.py's gamma_w_ref) so it sits at a
        # similar O(1) input scale to the Fourier features / latent
        # features rather than raw meter values (~1e-3).
        sdf_normalized = (sdf_raw / D_pt).detach()

        return self.decoder(latent, query_points_norm, batch_index, sdf_normalized)
