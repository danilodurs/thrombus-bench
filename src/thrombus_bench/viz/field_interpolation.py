"""Grid-interpolation and mesh-quality postprocessing for showcase visualization.

Responsibility
---------------
Presentation-support utilities consumed by `viz/showcase_plots.py`, kept
separate from `mechanistic/flow_solver.py` (which owns genuine FEM
postprocessing quantities like `vorticity`/`wall_traction`, computed
directly from a `Basis`) since neither function here needs a scikit-fem
`Basis`/`Mesh` object at all -- both operate on plain node-coordinate +
value arrays, the same shape of input `data/generate_dataset.py` already
works with:

* `interpolate_velocity_to_grid`: griddata-based velocity field on a
  regular grid, for `matplotlib.pyplot.streamplot` (which requires a
  regular grid, unlike the FEM mesh's scattered/unstructured nodes).
  Follows `data/generate_dataset._rasterize`'s exact griddata pattern and
  reuses `data/generate_dataset._fluid_mask` directly for exterior-cell
  masking (the vessel+aneurysm domain is an L/T-shaped union, not a
  rectangle -- see that function's docstring) rather than duplicating that
  triangulation-based point-in-domain logic; `notebooks/
  01_explore_mechanistic_baseline.ipynb` already imports `_fluid_mask`
  directly for the same reason, so this is consistent with existing
  project precedent, not a new coupling.
* `triangle_min_angles_deg`: minimum interior angle (degrees) of every mesh
  triangle -- a standard mesh-quality diagnostic (low min-angle = a
  sliver/degenerate triangle), used by the mesh-quality showcase figure.

Neither function touches solver output or writes to any dataset -- both
are pure presentation-support (grid resampling / geometric mesh
diagnostics), computed on demand from an already-solved flow field or
already-built mesh, never persisted.

Input shape convention
-------------------------
Both functions accept node coordinates / triangle connectivity in either
`(2, n_nodes)`/`(3, n_triangles)` (the raw `skfem.MeshTri.p`/`.t`
convention used directly in `notebooks/01_explore_mechanistic_baseline.ipynb`)
or `(n_nodes, 2)`/`(n_triangles, 3)` (the point-cloud `.npz` schema's
`node_coords`/`triangles` convention, `data/generate_dataset.
_build_pointcloud_sample`, used when loading a saved sample e.g. in
`notebooks/02_explore_continuous_surrogate.ipynb`) -- detected from
whichever axis has length 2 or 3, since both call sites are real,
existing conventions in this project and neither should have to
transpose by hand before calling.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt

from ..data.generate_dataset import _fluid_mask


def _canonicalize_2n(node_coords: np.ndarray) -> np.ndarray:
    """(2, n_nodes), see module docstring's "Input shape convention"."""

    node_coords = np.asarray(node_coords)
    return node_coords if node_coords.shape[0] == 2 else node_coords.T


def _canonicalize_3t(triangles: np.ndarray) -> np.ndarray:
    """(3, n_triangles), see module docstring's "Input shape convention"."""

    triangles = np.asarray(triangles)
    return triangles if triangles.shape[0] == 3 else triangles.T


def rasterize_reference_on_grid(node_coords: np.ndarray, values: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Interpolate scattered reference (mesh-node) values onto an
    explicitly-given `(xs, ys)` grid -- e.g. matching `viz.
    rasterize_continuous.rasterize_continuous_model`'s own analytic
    bounding-box grid exactly, for a pixel-aligned mechanistic/surrogate/
    error comparison (a "triptych" figure).

    Unlike `data/generate_dataset._rasterize` (which derives its own grid
    from `node_coords`' bounding box), this takes the grid as given so the
    caller controls exact pixel alignment against a second,
    independently-generated raster. Uses `method="nearest"`, matching
    `_rasterize`'s own convention (safer than linear interpolation for the
    spiky, log-compressed concentration fields this project works with).

    `node_coords`: either shape convention (see module docstring).
    `values`: `(n_nodes,)`. `xs`/`ys`: `(n_cols,)`/`(n_rows,)` 1D grid
    coordinates. Returns `(len(ys), len(xs))`.
    """

    coords_n2 = _canonicalize_2n(node_coords).T
    gx, gy = np.meshgrid(xs, ys)
    return griddata(coords_n2, values, (gx, gy), method="nearest")


def interpolate_velocity_to_grid(
    node_coords: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    triangles: np.ndarray,
    grid_size: tuple[int, int] = (80, 200),
) -> dict:
    """Interpolate a nodal velocity field onto a regular grid for
    `matplotlib.pyplot.streamplot`.

    `node_coords`: mesh vertex coordinates (meters), either shape
    convention (see module docstring). `ux`/`uy`: `(n_nodes,)` nodal
    velocity components. `triangles`: mesh connectivity, either shape
    convention -- used only to build the fluid-domain mask (`_fluid_mask`).

    Returns `{"x": (n_cols,), "y": (n_rows,), "u": (n_rows, n_cols),
    "v": (n_rows, n_cols), "mask": (n_rows, n_cols) bool}` -- `u`/`v` are
    NaN outside `mask` (exterior cells), matching `viz/rasterize_continuous.
    rasterize_continuous_model`'s NaN-outside-domain convention so callers
    can `imshow`/`streamplot` directly without a separate masking step.
    """

    coords_2n = _canonicalize_2n(node_coords)
    tri_3t = _canonicalize_3t(triangles)

    xmin, ymin = coords_2n.min(axis=1)
    xmax, ymax = coords_2n.max(axis=1)
    xs = np.linspace(xmin, xmax, grid_size[1])
    ys = np.linspace(ymin, ymax, grid_size[0])
    gx, gy = np.meshgrid(xs, ys)

    points_n2 = coords_2n.T
    u_grid = griddata(points_n2, ux, (gx, gy), method="linear")
    v_grid = griddata(points_n2, uy, (gx, gy), method="linear")
    # Linear interpolation leaves NaN outside node_coords' convex hull;
    # nearest-neighbor-fill those before masking (matching
    # generate_dataset._rasterize's method="nearest") so a streamplot
    # doesn't get holes from convex-hull edge effects distinct from the
    # true (non-convex) fluid-domain boundary, which `_fluid_mask` below
    # already handles correctly.
    nan_mask = np.isnan(u_grid) | np.isnan(v_grid)
    if nan_mask.any():
        u_fallback = griddata(points_n2, ux, (gx, gy), method="nearest")
        v_fallback = griddata(points_n2, uy, (gx, gy), method="nearest")
        u_grid = np.where(nan_mask, u_fallback, u_grid)
        v_grid = np.where(nan_mask, v_fallback, v_grid)

    mask = _fluid_mask(coords_2n, tri_3t, grid_size).astype(bool)
    u_grid = np.where(mask, u_grid, np.nan).astype(np.float32)
    v_grid = np.where(mask, v_grid, np.nan).astype(np.float32)

    return {"x": xs, "y": ys, "u": u_grid, "v": v_grid, "mask": mask}


def vorticity_from_grid(grid: dict) -> np.ndarray:
    """Finite-difference curl (dv/dx - du/dy) on a regular grid, e.g. the
    output of `interpolate_velocity_to_grid`.

    This is a *display* approximation of vorticity for a quick pcolormesh
    over the same regular grid `streamplot` already uses, distinct from
    `mechanistic.flow_solver.vorticity_at_quadrature`'s exact FEM value
    (the authoritative one for anything quantitative -- unit-tested in
    `tests/test_flow_postprocessing.py`; this function has no such
    guarantee and should never be used for a numerical claim). NaN cells
    (outside the fluid domain, `grid["mask"]`) are nearest-neighbor-filled
    before differencing, to avoid spurious gradients from a hard NaN edge,
    then the output is re-masked to NaN with the same mask.
    """

    u, v, mask = grid["u"], grid["v"], grid["mask"]

    def _fill(a: np.ndarray) -> np.ndarray:
        if np.all(mask):
            return a
        idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        return a[tuple(idx)]

    u_filled, v_filled = _fill(u), _fill(v)
    dx = grid["x"][1] - grid["x"][0]
    dy = grid["y"][1] - grid["y"][0]
    dv_dx = np.gradient(v_filled, dx, axis=1)
    du_dy = np.gradient(u_filled, dy, axis=0)
    omega = dv_dx - du_dy
    return np.where(mask, omega, np.nan)


def triangle_min_angles_deg(node_coords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Minimum interior angle (degrees) of every mesh triangle -- a
    standard shape-quality metric (an equilateral triangle has all angles
    60 degrees; a sliver triangle has a min angle near 0), used by the
    mesh-quality showcase figure.

    `node_coords`/`triangles`: either shape convention (see module
    docstring). Returns `(n_triangles,)`.
    """

    coords_n2 = _canonicalize_2n(node_coords).T
    tri_t3 = _canonicalize_3t(triangles).T

    p0 = coords_n2[tri_t3[:, 0]]
    p1 = coords_n2[tri_t3[:, 1]]
    p2 = coords_n2[tri_t3[:, 2]]

    def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Interior angle (degrees) at vertex `a`, between edges a->b, a->c."""

        v1, v2 = b - a, c - a
        cos_theta = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    angle0 = _angle_at(p0, p1, p2)
    angle1 = _angle_at(p1, p0, p2)
    angle2 = 180.0 - angle0 - angle1
    return np.minimum(np.minimum(angle0, angle1), angle2)
