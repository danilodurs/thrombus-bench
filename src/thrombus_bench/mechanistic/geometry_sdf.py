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
neck x-positions, and -- geometry-redesign Phase 4b, see `docs/
geometry_redesign_assessment.md` -- the sac's ellipse parameters
`xc_ellipse`/`a_left`/`a_right`/`b`) specifically so this module can never
silently drift out of sync with the mesh it must match -- a
duplicated-but-slightly-wrong formula here would misinform the decoder
about where the wall actually is without any visible error.

The fluid domain is the union of:
  - a rectangle `[0, L] x [0, D]` (the parent vessel), and
  - the upper half (`y >= D`) of an asymmetric ellipse silhouette (two
    quarter-ellipse pieces sharing an apex, semi-axes `a_left`/`a_right`/
    `b`) attached to the vessel top wall at `(xc +/- R, D)` (the aneurysm
    sac) -- a circle of radius `R` centered at `(xc, D)` is the special
    case `a_left == a_right == b == R` (`sac_asymmetry == 0`,
    `sac_height_mm` unset), reproduced exactly, not approximately.

These two pieces meet exactly along the shared segment
`y = D, x in [xc - R, xc + R]` (the ellipse pieces' own base chord
coincides with part of the vessel's top wall line, regardless of
`sac_asymmetry` -- see `_aneurysm_geometry_points`) -- they do not overlap
in area, so that segment is interior to the union, not part of its
boundary. This matches `mesh.py::_build_boundary_polygon` exactly: the top
wall is only tagged `wall_vessel` outside `[xl, xr]`, and the arc (tagged
`wall_sac`) spans from `xr` up over the top to `xl`. The union's boundary
is therefore *exactly* six pieces -- five straight segments (bottom,
right, the two outer top-wall stubs, left) plus the two-piece sac arc --
and the unsigned distance to the wall is the minimum distance to any of
those pieces (not a naive `min`/`max` of two independent whole-shape SDFs,
which would be wrong right at the neck transition: see module tests for a
mesh-boundary cross-check that would fail on that naive approach).

Unlike the straight segments (and unlike the old single-circle arc), a
general ellipse arc has no closed-form nearest-point formula -- `_dist_to_
ellipse_quarter` below uses the standard Newton iteration on the ellipse's
own parametrization instead (a handful of iterations, vectorized; see its
docstring). This is still "closed-form" in the sense the module docstring
above promises (no mesh lookup, computed directly from `GeometryConfig`),
just not literally an algebraic formula for this one piece.
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


def _dist_to_ellipse_quarter(
    px: np.ndarray, py: np.ndarray, a: float, b: float, theta_lo: float, theta_hi: float,
    n_grid: int = 25, n_newton: int = 6,
) -> np.ndarray:
    """Distance from `(px, py)` (already translated so the ellipse center
    is the origin) to the quarter-ellipse arc `x = a*cos(theta), y =
    b*sin(theta)`, `theta` restricted to `[theta_lo, theta_hi]`.

    No closed form exists for nearest-point-on-ellipse in general (unlike
    a circle). The optimality condition `g(theta) = (b^2-a^2)*sin(theta)*
    cos(theta) + a*px*sin(theta) - b*py*cos(theta) = 0` can have multiple
    roots (the ellipse-distance function can have more than one critical
    point), so plain Newton iteration from a single "auxiliary circle"
    initial guess (`atan2(py/b, px/a)`) is **not** globally convergent --
    for a sufficiently asymmetric ellipse (`a` and `b` far apart) and a
    query point on the "wrong side" of this particular quarter, that
    initial guess can land outside `[theta_lo, theta_hi]` entirely and
    Newton converges to the wrong (non-global, still a valid critical
    point of the *unconstrained* problem) root instead of the true nearest
    point on this bounded arc. Caught by a brute-force cross-check during
    Phase 4b development (a ~0.6mm error on an asymmetric test geometry,
    orders of magnitude past the mesh-spacing-scale tolerance this module
    otherwise achieves) -- not a hypothetical concern.

    Fix: a coarse grid search over `[theta_lo, theta_hi]` first (cheap --
    this range only ever spans 90 degrees) to find a starting point
    already in the correct basin of attraction, *then* a handful of Newton
    steps to polish it to high precision. Finally clamp to
    `[theta_lo, theta_hi]` -- if the polished optimum still sits at (or
    was pulled past) an endpoint, the true nearest point on this *bounded*
    arc is that endpoint instead (taking the min over both quarters in
    `_dist_to_sac_arc`, rather than pre-classifying which side a query
    point is "nominally" on, is what makes the two-piece union correct for
    every query point regardless of which quarter's endpoint wins).

    Reduces to the exact radial-projection answer when `a == b` (a
    circle): the optimality condition degenerates to `sin(theta)*px =
    cos(theta)*py` (independent of `a`/`b`), satisfied by the true polar
    angle of `(px, py)` -- which the grid search (spacing < 90/24 degrees)
    plus Newton polish converge to exactly.
    """

    a = max(a, 1e-12)
    b = max(b, 1e-12)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    orig_shape = px.shape
    px_flat, py_flat = px.ravel(), py.ravel()

    grid = np.linspace(theta_lo, theta_hi, n_grid)
    gx, gy = a * np.cos(grid), b * np.sin(grid)
    dist2 = (gx[:, None] - px_flat[None, :]) ** 2 + (gy[:, None] - py_flat[None, :]) ** 2
    theta = grid[np.argmin(dist2, axis=0)]

    for _ in range(n_newton):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        g = (b * b - a * a) * sin_t * cos_t + a * px_flat * sin_t - b * py_flat * cos_t
        g_prime = (b * b - a * a) * (cos_t * cos_t - sin_t * sin_t) + a * px_flat * cos_t + b * py_flat * sin_t
        # np.divide(..., where=...), not np.where(..., g/g_prime, 0.0): the
        # latter still evaluates g/g_prime everywhere (including the
        # near-zero-g_prime elements it then discards), raising a spurious
        # RuntimeWarning for a value that was never actually used.
        step = np.divide(g, g_prime, out=np.zeros_like(g), where=np.abs(g_prime) > 1e-12)
        theta = theta - step
    theta = np.clip(theta, theta_lo, theta_hi)
    ex, ey = a * np.cos(theta), b * np.sin(theta)
    return np.hypot(px_flat - ex, py_flat - ey).reshape(orig_shape)


def _dist_to_sac_arc(
    x: np.ndarray, y: np.ndarray, xc_ellipse: float, D: float, a_left: float, a_right: float, b: float
) -> np.ndarray:
    """Distance to the sac boundary (two quarter-ellipse arcs sharing an
    apex, see `mesh.py::_aneurysm_geometry_points`) -- the min of each
    quarter's own nearest-point distance."""

    px, py = x - xc_ellipse, y - D
    d_right = _dist_to_ellipse_quarter(px, py, a_right, b, 0.0, 0.5 * np.pi)
    d_left = _dist_to_ellipse_quarter(px, py, a_left, b, 0.5 * np.pi, np.pi)
    return np.minimum(d_left, d_right)


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
    L, D = g["L"], g["D"]
    xc_ellipse, a_left, a_right, b = g["xc_ellipse"], g["a_left"], g["a_right"], g["b"]
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
    d_unsigned = _dist_to_sac_arc(x, y, xc_ellipse, D, a_left, a_right, b)
    for seg_a, seg_b in segments:
        d_unsigned = np.minimum(d_unsigned, _dist_to_segment(x, y, seg_a, seg_b))

    inside_rect = (x >= 0.0) & (x <= L) & (y >= 0.0) & (y <= D)
    dx = x - xc_ellipse
    a_of_pt = np.where(dx >= 0.0, a_right, a_left)
    inside_sac = (y >= D) & ((dx / a_of_pt) ** 2 + ((y - D) / b) ** 2 <= 1.0)
    inside = inside_rect | inside_sac

    signed = np.where(inside, d_unsigned, -d_unsigned)
    return signed[()] if signed.ndim == 0 else signed
