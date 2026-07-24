"""Fourier-feature positional encoding for the continuous decoder's (x, y) input.

Responsibility
---------------
Stage 2 of the continuous-surrogate design (see
`docs/continuous_surrogate_design.md`) evaluates the model at an arbitrary
continuous query point `(x, y)` rather than a fixed raster cell. A plain MLP
fed raw coordinates struggles to represent high-frequency spatial variation
(the well-known "spectral bias" of coordinate networks) -- SIREN/NeRF-style
work fixes this by lifting the input through a bank of fixed sinusoids
before the MLP, so each input dimension gets an explicit high-frequency
basis to compose from instead of the network having to learn one.

This module implements that lift only: `FourierFeatureEncoding` maps a
batch of `(x, y)` points, each already normalized to `[-1, 1]` (the same
convention `neural/encoder.py`'s `_coordinate_grid` uses for its latent
grid, so both stages agree on what "position" means), to a fixed-size
encoding

    gamma(p) = [sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^{L-1} pi p), cos(2^{L-1} pi p)]

applied independently per coordinate and concatenated, giving a
`2 * n_coords * L`-dimensional output.

Choice of `L` (default 8)
--------------------------
`configs/training.yaml`'s `model.encoder.latent_grid_size` is `[32, 32]` --
Stage 1's shared FNO backbone already represents spatial variation up to
roughly that grid's own Nyquist resolution (~16 cycles across `[-1, 1]`).
`L=8` gives a highest frequency band of `2**7 = 128`, comfortably above
that (so the decoder MLP is not the bottleneck when resolving variation
*within* a single latent grid cell, e.g. steep gradients near the wall)
while staying small enough (`4*L = 32`-dim encoding for 2D input) to be
cheap and to avoid encoding frequencies far beyond what a few-thousand-
element FEM mesh can even resolve. This matches the common NeRF-family
default order of magnitude (NeRF itself uses `L=10` for 3D position on
scenes with much finer high-frequency detail).

Known boundary artifact
------------------------
Every frequency is an exact integer multiple of `pi`, matching the `[-1, 1]`
domain's own period-2 width -- so, for *any* `L`, the two domain endpoints
`p = -1` and `p = +1` are indistinguishable under this encoding
(`sin(2^k pi * (+-1))` and `cos(2^k pi * (+-1))` both collapse to the same
value for every integer `k`). This is an inherent property of the
NeRF-style formula as specified (not implementation-specific), affects only
that single boundary pair, and is exercised explicitly by
`test_coordinate_encoding.test_boundary_endpoints_collide_by_construction`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

DEFAULT_N_FREQUENCIES = 8


class FourierFeatureEncoding(nn.Module):
    """Deterministic sinusoidal positional encoding, applied per coordinate.

    No learnable parameters -- this is a fixed reparameterization of the
    input, not a trained embedding, so it is exactly reproducible given the
    same input and `n_frequencies`.
    """

    def __init__(self, n_coords: int = 2, n_frequencies: int = DEFAULT_N_FREQUENCIES):
        super().__init__()
        if n_frequencies < 1:
            raise ValueError(f"n_frequencies must be >= 1, got {n_frequencies}")
        self.n_coords = n_coords
        self.n_frequencies = n_frequencies
        freqs = math.pi * 2.0 ** torch.arange(n_frequencies, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def output_dim(self) -> int:
        return 2 * self.n_coords * self.n_frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., n_coords), each entry expected in `[-1, 1]` ->
        (..., 2 * n_coords * n_frequencies)."""

        if coords.shape[-1] != self.n_coords:
            raise ValueError(f"expected last dim {self.n_coords}, got {coords.shape[-1]}")
        # (..., n_coords, 1) * (n_frequencies,) -> (..., n_coords, n_frequencies)
        angles = coords.unsqueeze(-1) * self.freqs
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., n_coords, 2*n_frequencies)
        return encoded.flatten(start_dim=-2)  # (..., 2 * n_coords * n_frequencies)
