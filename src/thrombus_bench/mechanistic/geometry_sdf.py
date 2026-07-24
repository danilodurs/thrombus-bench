"""Closed-form signed distance to the idealized vessel+aneurysm wall.

Responsibility
---------------
The continuous-surrogate decoder (see `docs/continuous_surrogate_design.md`)
needs, for an arbitrary continuous query point `(x, y)`, an analytic
signed-distance-to-wall value -- computed directly from `GeometryConfig`,
with no mesh lookup -- since the domain is an idealized parametric shape
(`mesh.py::build_aneurysm_mesh`), not an arbitrary CAD boundary.

Sign convention
-----------------
Positive **inside** the fluid domain, negative **outside**, zero **on** the
wall -- i.e. `signed_distance_to_wall(p) = +dist(p, wall)` for `p` in the
fluid interior and `-dist(p, wall)` otherwise. (Note this is the opposite
sign convention from the common raymarching/CSG convention where "inside"
is negative; it is chosen here to read naturally as "how far inside the
vessel is this point," matching how the decoder will use it as a feature.)

Geometry, matched exactly to `mesh.py`
-----------------------------------------
`mesh.py::_aneurysm_geometry_points` is reused directly (not
reimplemented) for the domain's derived quantities (`L`, `D`, `R`, `xc`,
neck x-positions) specifically so this module can never silently drift out
of sync with the mesh it must match -- a duplicated-but-slightly-wrong
formula here would misinform the decoder about where the wall actually is
without any visible error.

The fluid domain is the union of:
  - a rectangle `[0, L] x [0, D]` (the parent vessel), and
  - the upper half (`y >= D`) of a disk of radius `R` centered at
    `(xc, D)` (the aneurysm sac), where `xc = L / 2`.

These two pieces meet exactly along the shared segment
`y = D, x in [xc - R, xc + R]` (the disk's own diameter coincides with part
of the vessel's top wall line) -- they do not overlap in area, so that
segment is interior to the union, not part of its boundary. This matches
`mesh.py::_build_boundary_polygon` exactly: the top wall is only tagged
`wall_vessel` outside `[xl, xr]`, and the arc (tagged `wall_sac`) spans
from `xr` up over the top to `xl`. The union's boundary is therefore
*exactly* six pieces -- five straight segments (bottom, right, the two
outer top-wall stubs, left) plus the sac arc -- and the unsigned distance
to the wall is the minimum distance to any of those six pieces (not a
naive `min`/`max` of two independent whole-shape SDFs, which would be
wrong right at the neck transition: see module tests for a mesh-boundary
cross-check that would fail on that naive approach).
"""

from __future__ import annotations

import numpy as np

from .mesh import GeometryConfig, _aneurysm_geometry_points


def _dist_to_segment(x: np.ndarray, y: np.ndarray, a: tuple, b: tuple) -> np.ndarray:
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    ab_len2 = abx * abx + aby * aby
    t = ((x - ax) * abx + (y - ay) * aby) / ab_len2
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * abx, ay + t * aby
    return np.hypot(x - cx, y - cy)


def _dist_to_sac_arc(x: np.ndarray, y: np.ndarray, xc: float, D: float, R: float) -> np.ndarray:
    """Distance to the upper-half-circle arc only (theta in [0, pi]), not
    the full circle -- for query points whose angular position from the
    center falls outside that range, the nearest arc point is whichever
    endpoint (`theta=0` or `theta=pi`) is closer, not the radial
    projection."""

    theta = np.arctan2(y - D, x - xc)
    r = np.hypot(x - xc, y - D)
    on_arc = (theta >= 0.0) & (theta <= np.pi)
    d_radial = np.abs(r - R)
    d_end_right = np.hypot(x - (xc + R), y - D)  # theta = 0 endpoint
    d_end_left = np.hypot(x - (xc - R), y - D)  # theta = pi endpoint
    d_endpoint = np.minimum(d_end_right, d_end_left)
    return np.where(on_arc, d_radial, d_endpoint)


def signed_distance_to_wall(x, y, geometry_cfg: dict | GeometryConfig) -> np.ndarray:
    """Signed distance from `(x, y)` (meters, same convention as
    `mesh.py`'s node coordinates) to the nearest wall of the idealized
    vessel+aneurysm domain described by `geometry_cfg`.

    `x`/`y` may be scalars or arrays of matching shape (standard numpy
    broadcasting applies). See module docstring for the sign convention and
    exact geometry match to `mesh.py`.
    """

    geom = geometry_cfg if isinstance(geometry_cfg, GeometryConfig) else GeometryConfig.from_preset(geometry_cfg)
    g = _aneurysm_geometry_points(geom)
    L, D, R, xc = g["L"], g["D"], g["R"], g["xc"]
    xl, xr = g["x_neck_left"], g["x_neck_right"]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    segments = (
        ((0.0, 0.0), (L, 0.0)),  # bottom
        ((L, 0.0), (L, D)),  # right
        ((L, D), (xr, D)),  # top-right wall stub
        ((xl, D), (0.0, D)),  # top-left wall stub
        ((0.0, D), (0.0, 0.0)),  # left
    )
    d_unsigned = _dist_to_sac_arc(x, y, xc, D, R)
    for a, b in segments:
        d_unsigned = np.minimum(d_unsigned, _dist_to_segment(x, y, a, b))

    inside_rect = (x >= 0.0) & (x <= L) & (y >= 0.0) & (y <= D)
    inside_disk = (y >= D) & ((x - xc) ** 2 + (y - D) ** 2 <= R * R)
    inside = inside_rect | inside_disk

    signed = np.where(inside, d_unsigned, -d_unsigned)
    return signed[()] if signed.ndim == 0 else signed
