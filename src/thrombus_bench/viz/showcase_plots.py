"""Publication-facing ("showcase") figures, kept separate from `viz/plots.py`.

Responsibility
---------------
`viz/plots.py` renders the existing diagnostic/sanity-check style used by
`benchmark/run_benchmark.py`'s report (plain `tripcolor`, no physical-unit
axis labels, no shared-colorbar convention across panels) -- left exactly
as-is here, since it is a tested, in-use module. This module is a
file-level-separate home for new, more presentation-polished figures
(Stage 1 of the visualization-quality work): velocity vectors, wall
traction, streamlines, vorticity, pressure, wall-band platelet coverage
alongside bulk fibrin, and mesh quality -- consistent with the project's
figure conventions:

* Equal physical aspect ratio (`ax.set_aspect("equal")`, never `"auto"`).
* Axes labeled in physical units (millimeters here -- more legible than
  meters at this domain's ~cm scale).
* A shared-colorbar helper (`compute_shared_vlim`) for callers comparing
  multiple panels (e.g. mechanistic vs. surrogate) on the same scale.

Like `viz/plots.py`, every function here takes already-computed arrays
(from `mechanistic/flow_solver.py`'s FEM postprocessing or `viz/
field_interpolation.py`'s grid/mesh-quality utilities) -- no function in
this module runs a solve or writes to disk; all are pure rendering.

Wall-vs-bulk fields (explicit, see README.md's "Assumptions & Deviations"
and `surface_ode.py`'s docstring): `M`/`M_r`/`M_as`/`M_at` are wall-only
surface densities and are only ever plotted here along the wall geometry
itself (`plot_wall_band_M_at`), never as a bulk field. Fibrin (`FI`) is a
genuine bulk species (`data.dataset.FIELD_NAMES`) and is plotted with the
generic bulk-field renderer (`plot_bulk_field`), in its own panel --
never merged into the wall-band representation, even when both are shown
in the same figure (e.g. Stage 2's aggregation/fibrin narrative section).

Every quantity here is a diagnostic/visualization proxy over the
mechanistic model's own output (or, in Stage 3, the surrogate's
prediction of it) -- none of it implies a solved moving boundary or clot
geometry; neither this project nor the source paper solves one.
"""

from __future__ import annotations

import numpy as np

from . import plots as plots_mod
from .field_interpolation import _canonicalize_2n, _canonicalize_3t, triangle_min_angles_deg, vorticity_from_grid

_M_TO_MM = 1000.0


def compute_shared_vlim(*fields: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    """`(vmin, vmax)` spanning every array in `fields` (NaNs ignored) --
    pass the same tuple to every panel's `vmin`/`vmax` (or `norm`) when
    comparing multiple fields in one figure, per the shared-colorbar figure
    convention.

    `symmetric=True` returns `(-m, m)` with `m = max(|vmin|, |vmax|)` --
    appropriate for diverging quantities like vorticity or error signed
    around zero, so zero sits at the colormap's center.
    """

    vmin = min(float(np.nanmin(f)) for f in fields if np.any(np.isfinite(f)))
    vmax = max(float(np.nanmax(f)) for f in fields if np.any(np.isfinite(f)))
    if symmetric:
        m = max(abs(vmin), abs(vmax))
        return -m, m
    return vmin, vmax


def _prep_axes(ax, xlabel: str = "x [mm]", ylabel: str = "y [mm]"):
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_velocity_quiver(node_coords: np.ndarray, ux: np.ndarray, uy: np.ndarray, ax=None, max_arrows: int = 400):
    """Velocity vector field at (a subsample of) mesh nodes -- unlike the
    existing 1D wall shear-rate line plot, this shows direction and
    magnitude over the whole 2D domain.

    `node_coords`: `(2, n_nodes)` or `(n_nodes, 2)` physical coordinates
    (meters). `ux`/`uy`: `(n_nodes,)` nodal velocity components (m/s).
    """

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    coords_n2 = _canonicalize_2n(node_coords).T
    n = coords_n2.shape[0]
    if n > max_arrows:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_arrows, replace=False)
    else:
        idx = np.arange(n)

    x_mm, y_mm = coords_n2[idx, 0] * _M_TO_MM, coords_n2[idx, 1] * _M_TO_MM
    speed = np.hypot(ux[idx], uy[idx])
    q = ax.quiver(x_mm, y_mm, ux[idx], uy[idx], speed, cmap="viridis", angles="xy")
    ax.set_title("Velocity field")
    _prep_axes(ax)
    return q


def plot_wall_traction_map(traction_result: dict, ax=None, quiver: bool = True):
    """2D map of wall traction magnitude along the actual wall geometry
    (`mechanistic.flow_solver.wall_traction`'s output), optionally
    overlaid with direction arrows -- the 2D generalization of the
    existing 1D twin-axis wall shear-rate line plot, and a genuine stress
    (pressure + viscous), not just the shear-rate scalar invariant.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    points_mm = traction_result["points"] * _M_TO_MM
    magnitude = traction_result["magnitude"]
    sc = ax.scatter(points_mm[:, 0], points_mm[:, 1], c=magnitude, cmap="magma", s=8)
    if quiver:
        traction = traction_result["traction"]
        norm = np.maximum(magnitude, 1e-30)
        ax.quiver(
            points_mm[:, 0], points_mm[:, 1], traction[:, 0] / norm, traction[:, 1] / norm,
            color="white", alpha=0.6, angles="xy", scale=30, width=0.003,
        )
    ax.set_title("Wall traction magnitude [Pa]")
    _prep_axes(ax)
    return sc


def plot_streamlines(grid: dict, ax=None, density: float = 1.2):
    """Streamline plot from `viz.field_interpolation.interpolate_velocity_to_grid`'s
    regular-grid velocity field -- `matplotlib.pyplot.streamplot` requires a
    regular grid, unlike the FEM mesh's scattered nodes.

    Returns the `StreamplotSet.lines` `LineCollection` (not the `ax`), so a
    caller can attach a colorbar the same way as every other function in
    this module (`fig.colorbar(plot_streamlines(...), ax=ax)`)."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    x_mm, y_mm = grid["x"] * _M_TO_MM, grid["y"] * _M_TO_MM
    u, v = np.nan_to_num(grid["u"]), np.nan_to_num(grid["v"])
    speed = np.hypot(u, v)
    stream = ax.streamplot(x_mm, y_mm, u, v, color=speed, cmap="viridis", density=density)
    ax.set_xlim(x_mm.min(), x_mm.max())
    ax.set_ylim(y_mm.min(), y_mm.max())
    ax.set_title("Streamlines")
    _prep_axes(ax)
    return stream.lines


def plot_vorticity_field(grid: dict, ax=None, vlim: tuple[float, float] | None = None):
    """Vorticity field (`viz.field_interpolation.vorticity_from_grid`) as a
    diverging pcolormesh over the same regular grid `plot_streamlines`
    uses -- see that function's docstring for why this is a display
    approximation, not the exact FEM value."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    omega = vorticity_from_grid(grid)
    x_mm, y_mm = grid["x"] * _M_TO_MM, grid["y"] * _M_TO_MM
    if vlim is None:
        finite = omega[np.isfinite(omega)]
        # 98th percentile, not the raw max: a handful of near-wall
        # finite-difference cells (steepest part of the boundary layer,
        # right next to a hard no-slip/exterior-mask edge) can be an order
        # of magnitude larger than the rest of the field and would
        # otherwise saturate the whole colormap to near-uniform, hiding
        # the bulk/recirculation structure this plot exists to show.
        m = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
        vlim = (-m, m)
    mesh = ax.pcolormesh(x_mm, y_mm, omega, cmap="RdBu_r", vmin=vlim[0], vmax=vlim[1], shading="auto")
    ax.set_title("Vorticity [1/s] (display approximation)")
    _prep_axes(ax)
    return mesh


def plot_pressure_field(node_coords: np.ndarray, triangles: np.ndarray, pressure: np.ndarray, ax=None):
    """Pressure field (`mechanistic.flow_solver.FlowSolution.p`), previously
    unused in any plot -- reuses `viz.plots.plot_mesh_field`'s tripcolor
    renderer directly (same shared implementation `plot_viscosity_field`/
    `plot_thrombus_map`/`plot_error_heatmap` already use) and adds this
    module's physical-unit axis labels on top.

    `node_coords`/`triangles`: mesh vertex coordinates (meters) / mesh
    connectivity, `skfem.MeshTri.p`/`.t` convention (`(2, n_nodes)`/
    `(3, n_triangles)`).
    """

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    node_coords_mm = _canonicalize_2n(node_coords) * _M_TO_MM
    triangles_3t = _canonicalize_3t(triangles)
    tpc = plots_mod.plot_mesh_field(node_coords_mm, triangles_3t, pressure, ax, title="Pressure [Pa]", cmap="coolwarm")
    _prep_axes(ax)
    return tpc


def plot_bulk_field(node_coords: np.ndarray, triangles: np.ndarray, values: np.ndarray, ax=None, title: str = "", cmap: str = "viridis"):
    """Generic bulk (whole-domain) scalar field renderer -- e.g. fibrin
    (`conc_FI`), which is a genuine bulk species (present at every mesh
    node), NOT a wall-only quantity like M_at. Kept as its own function
    (rather than folded into `plot_wall_band_M_at`) so a figure combining
    both never blends the two physically-different supports into one
    panel -- see module docstring's "Wall-vs-bulk fields" note.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    node_coords_mm = _canonicalize_2n(node_coords) * _M_TO_MM
    triangles_3t = _canonicalize_3t(triangles)
    tpc = plots_mod.plot_mesh_field(node_coords_mm, triangles_3t, values, ax, title=title or "Bulk field", cmap=cmap)
    _prep_axes(ax)
    return tpc


def plot_wall_band_M_at(wall_node_coords: np.ndarray, m_at_values: np.ndarray, ax=None):
    """M_at (platelet coverage, PLT/cm^2) plotted along the actual wall
    node coordinates -- wall-only by construction (there is no bulk M_at to
    confuse this with), and exact (no rasterization/`griddata` involved,
    unlike the legacy `_rasterize_wall_band` grid-band approximation),
    since the point-cloud data path already saves exact wall node
    coordinates + `M_at` values (`data/dataset.py`'s `wall_node_coords`/
    `M_at_wall_values`, or `surface_ode.SurfaceState.M_at` directly from a
    live `CoupledSimulationHistory`).

    `wall_node_coords`: `(n_wall_nodes, 2)` or `(2, n_wall_nodes)`, meters.
    `m_at_values`: `(n_wall_nodes,)`.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    coords_n2 = _canonicalize_2n(wall_node_coords).T
    x_mm, y_mm = coords_n2[:, 0] * _M_TO_MM, coords_n2[:, 1] * _M_TO_MM
    sc = ax.scatter(x_mm, y_mm, c=m_at_values, cmap="Reds", s=10)
    ax.set_title("Wall platelet coverage M_at [PLT/cm^2] (wall-only)")
    _prep_axes(ax)
    return sc


def plot_mesh_quality(node_coords: np.ndarray, triangles: np.ndarray, ax=None, geom=None, inset: bool = True):
    """Triangle-quality (minimum interior angle) colormap, with an optional
    zoomed inset over the proximal/distal neck region -- the mesh's most
    refined and most failure-prone area (`mechanistic/mesh.py`'s docstring:
    past neck-tagging bugs broke mass conservation), so it's the region
    most worth visually auditing for slivers.

    `geom`: optional `mechanistic.mesh.GeometryConfig` for the inset's
    bounds (`vessel_length_mm`/2 +/- `aneurysm_diameter_mm`/2); if omitted,
    `inset` is ignored (no geometry to center it on).
    """

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    coords_n2 = _canonicalize_2n(node_coords).T
    triangles_t3 = _canonicalize_3t(triangles).T

    min_angles = triangle_min_angles_deg(node_coords, triangles)
    x_mm, y_mm = coords_n2[:, 0] * _M_TO_MM, coords_n2[:, 1] * _M_TO_MM
    tpc = ax.tripcolor(x_mm, y_mm, triangles_t3, facecolors=min_angles, cmap="viridis_r", vmin=0.0, vmax=60.0)
    ax.set_title("Mesh quality: min triangle angle [deg]")
    _prep_axes(ax)

    if inset and geom is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        xc_mm = 0.5 * geom.vessel_length_mm
        r_mm = 0.5 * geom.aneurysm_diameter_mm
        d_mm = geom.vessel_diameter_mm
        axins = inset_axes(ax, width="35%", height="35%", loc="upper right")
        axins.tripcolor(x_mm, y_mm, triangles_t3, facecolors=min_angles, cmap="viridis_r", vmin=0.0, vmax=60.0)
        axins.set_xlim(xc_mm - r_mm * 1.3, xc_mm + r_mm * 1.3)
        axins.set_ylim(d_mm - r_mm * 0.3, d_mm + r_mm * 1.3)
        axins.set_aspect("equal")
        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    return tpc
