"""Grid-style visualization utility for `ContinuousThrombusSurrogate` (Phase 6,
`docs/continuous_surrogate_design.md`).

Responsibility
---------------
`rasterize_continuous_model` queries a trained `ContinuousThrombusSurrogate`
on a regular grid so existing/future plotting code (`viz/plots.py`) can
still produce grid-style images (`imshow`/`pcolormesh`) without the model
itself being grid-based -- this is what "keep `_rasterize` as a
legacy/visualization utility" (`data/generate_dataset.py`'s Phase 3
docstring) refers to on the *consumption* side: the model doesn't need a
grid to make a prediction (that's the entire point of the continuous
design), only a human looking at a plot does.

Unlike `data/generate_dataset._rasterize` (which interpolates an existing
FEM mesh's scattered node values onto a grid via `griddata`, since the
mechanistic solver has no other way to give you a value at an arbitrary
point), this queries the model *directly* at each grid point -- no
interpolation, no FEM mesh needed at inference time at all. The grid spans
the sample's own analytic bounding box `[0, L] x [0, D + R]` (`geometry_sdf.
py`'s module docstring convention), and exterior cells (outside the actual
vessel+aneurysm domain, which doesn't fill its bounding box -- an L/T-shaped
union) are masked via Phase 1's analytic SDF, exactly like
`data/generate_dataset._fluid_mask` does for the legacy raster path, but
computed in closed form here rather than via mesh triangulation.
"""

from __future__ import annotations

import numpy as np
import torch

from ..mechanistic.geometry_sdf import signed_distance_to_wall
from ..mechanistic.mesh import GeometryConfig
from ..neural.coordinate_decoder import VESSEL_LENGTH_MM, ContinuousThrombusSurrogate


def rasterize_continuous_model(
    model: ContinuousThrombusSurrogate,
    params_with_time: torch.Tensor,
    geometry_mm: torch.Tensor,
    vessel_length_mm: float = VESSEL_LENGTH_MM,
    grid_size: tuple[int, int] = (64, 64),
) -> tuple[np.ndarray, np.ndarray]:
    """Query `model` on a regular `grid_size` grid for one sample.

    `params_with_time`: `(9,)` -- one sample's normalized encoder input
    (same convention as `ContinuousThrombusSurrogate.forward`).
    `geometry_mm`: `(2,)` -- that sample's raw `[aneurysm_diameter_mm,
    vessel_diameter_mm]`. `grid_size`: `(n_rows, n_cols)`, matching
    `data/generate_dataset._rasterize`'s convention.

    Returns `(fields_grid, fluid_mask)`:
    - `fields_grid`: `(n_rows, n_cols, output_channels)` float32, NaN
      outside the fluid domain (so `matplotlib.pyplot.imshow`/`pcolormesh`
      skip those cells automatically -- this is purely a display
      convenience, not something the model needed to make its prediction).
    - `fluid_mask`: `(n_rows, n_cols)` bool, True where the analytic SDF
      says the cell center is inside the fluid domain.
    """

    aneurysm_diameter_mm, vessel_diameter_mm = float(geometry_mm[0]), float(geometry_mm[1])
    geom = GeometryConfig(
        vessel_diameter_mm=vessel_diameter_mm,
        aneurysm_diameter_mm=aneurysm_diameter_mm,
        vessel_length_mm=vessel_length_mm,
    )
    L_m = vessel_length_mm * 1.0e-3
    D_m = vessel_diameter_mm * 1.0e-3
    R_m = aneurysm_diameter_mm * 0.5e-3

    n_rows, n_cols = grid_size
    xs = np.linspace(0.0, L_m, n_cols)
    ys = np.linspace(0.0, D_m + R_m, n_rows)
    gx, gy = np.meshgrid(xs, ys)  # (n_rows, n_cols) each

    query_points_m = torch.from_numpy(np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32))
    batch_index = torch.zeros(query_points_m.shape[0], dtype=torch.long)
    params_batch = params_with_time.unsqueeze(0)
    geometry_batch = geometry_mm.unsqueeze(0).to(dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred = model(params_batch, query_points_m, batch_index, geometry_batch)

    output_channels = pred.shape[-1]
    fields_grid = pred.numpy().reshape(n_rows, n_cols, output_channels).astype(np.float32)

    sdf = signed_distance_to_wall(gx.ravel(), gy.ravel(), geom).reshape(n_rows, n_cols)
    # >= 0, not > 0: this is a *display* mask, and the grid's own edges
    # (x=0, y=0) sit exactly on the analytic boundary (SDF==0) by
    # construction -- a boundary cell is visually still part of the
    # vessel, not a hole, so it should render, unlike the strict >0 "is
    # this point in the fluid interior" test used elsewhere for training.
    fluid_mask = sdf >= 0.0

    fields_grid = fields_grid.copy()
    fields_grid[~fluid_mask] = np.nan
    return fields_grid, fluid_mask
