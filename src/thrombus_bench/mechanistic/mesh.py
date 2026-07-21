"""Mesh generation for the idealized 2D aneurysm geometry.

Responsibility
---------------
Build the triangular meshes used by every downstream mechanistic module
(flow_solver, species_transport, activation, surface_ode, fibrin,
coupled_solver). Two geometries are produced:

1. ``build_channel_mesh``: a plain structured rectangular channel (no
   aneurysm). Used for validation/sanity cases (e.g. Poiseuille flow, mass
   conservation) where an analytical or trivially-checkable reference
   solution exists.
2. ``build_aneurysm_mesh``: the idealized 2D aneurysm domain of Cardillo,
   Pouponneau & Barakat (2026), Fig. 1(c)/(f) -- a straight parent vessel of
   length ``vessel_length_mm`` and diameter ``vessel_diameter_mm``, with a
   circular sac of diameter ``aneurysm_diameter_mm`` bulging from the top
   wall.

Solver backend note
--------------------
The paper uses COMSOL Multiphysics with an unstructured mesh of ~10,000
triangles, locally refined at the proximal/distal neck. This project uses
``scikit-fem`` (pip-installable, pure-Python + compiled kernels, no MPI/
PETSc/system-package requirements) rather than FEniCSx, whose Python
bindings (``fenics-dolfinx``) are not reliably pip-installable across
platforms -- they require conda or building against a system PETSc/MPI
stack. See README.md "Solver backend" for the full rationale.

Because scikit-fem does not ship a CAD/CSG geometry+meshing engine (unlike
FEniCSx+gmsh), the aneurysm domain is meshed here directly with a
lightweight boundary-conforming Delaunay triangulation (boundary points
placed along the analytic domain outline, interior points filled on a
graded background grid, triangles outside the polygon discarded). This
avoids adding a heavy external meshing dependency (e.g. ``gmsh``) while
remaining fully offline/pip-only. It is adequate for the coarse, idealized,
few-thousand-element meshes this project targets; it is not a substitute
for a robust CSG mesher on complex or patient-specific geometries.

Geometry model (explicit modeling assumption)
-----------------------------------------------
The source paper shows the idealized geometries only as images (Fig. 1) and
gives only the vessel/aneurysm diameters and vessel length (Table 1 caption
context, Sec. 2.8) -- it does not give a parametric definition of the
sac shape. This implementation models the sac as the union of the vessel
rectangle with the upper half-disk of a circle of diameter
``aneurysm_diameter_mm`` centered on the vessel's top wall at the vessel's
midpoint. This gives a neck width equal to the full aneurysm diameter and a
smooth, symmetric sac, matching the qualitative shape in Fig. 1. This
specific parametrization is a documented simplification -- see README.md
"Assumptions & Deviations from Source Paper".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path
from scipy.spatial import Delaunay, cKDTree
from skfem import MeshTri

MM_TO_M = 1.0e-3


@dataclass
class GeometryConfig:
    """Vessel/aneurysm dimensions, in millimeters (as in configs/geometry.yaml)."""

    vessel_diameter_mm: float
    aneurysm_diameter_mm: float
    vessel_length_mm: float

    @classmethod
    def from_preset(cls, preset: dict) -> "GeometryConfig":
        return cls(
            vessel_diameter_mm=float(preset["vessel_diameter_mm"]),
            aneurysm_diameter_mm=float(preset["aneurysm_diameter_mm"]),
            vessel_length_mm=float(preset["vessel_length_mm"]),
        )


@dataclass
class MeshConfig:
    """Mesh resolution controls (as in configs/geometry.yaml `mesh:` block)."""

    target_num_elements: int = 2000
    neck_refinement_factor: float = 3.0
    neck_band_fraction: float = 0.15
    min_boundary_points_per_edge: int = 4


@dataclass
class TaggedMesh:
    """A scikit-fem mesh plus named boundary facet sets.

    ``mesh.boundaries`` (set via ``MeshTri.with_boundaries``) already carries
    these tags; this wrapper just documents which tag names downstream code
    can rely on and carries geometry metadata used for unit bookkeeping.
    """

    mesh: MeshTri
    geometry: GeometryConfig
    boundary_names: tuple[str, ...] = field(
        default_factory=lambda: ("inlet", "outlet", "wall_vessel", "wall_sac")
    )


def build_channel_mesh(
    length_mm: float, diameter_mm: float, target_num_elements: int = 400
) -> TaggedMesh:
    """Structured rectangular channel mesh (no aneurysm) for validation cases.

    Boundaries: ``inlet`` (x=0), ``outlet`` (x=length), ``wall_vessel``
    (y=0 and y=diameter). ``wall_sac`` is present but always empty (kept so
    downstream code can treat channel and aneurysm meshes uniformly).
    """

    length_m = length_mm * MM_TO_M
    diameter_m = diameter_mm * MM_TO_M

    # Choose nx, ny (number of grid lines) so nx*ny*2 triangles ~= target.
    aspect = length_m / diameter_m
    ny = max(2, int(round(np.sqrt(target_num_elements / (2.0 * aspect)))))
    nx = max(2, int(round(target_num_elements / (2.0 * ny))))

    x = np.linspace(0.0, length_m, nx + 1)
    y = np.linspace(0.0, diameter_m, ny + 1)
    mesh = MeshTri.init_tensor(x, y)

    tol = 1e-9 * max(length_m, diameter_m)
    mesh = mesh.with_boundaries(
        {
            "inlet": lambda p: np.isclose(p[0], 0.0, atol=tol),
            "outlet": lambda p: np.isclose(p[0], length_m, atol=tol),
            "wall_vessel": lambda p: np.isclose(p[1], 0.0, atol=tol)
            | np.isclose(p[1], diameter_m, atol=tol),
            "wall_sac": lambda p: np.zeros(p.shape[1], dtype=bool),
        }
    )
    geometry = GeometryConfig(diameter_mm, 0.0, length_mm)
    return TaggedMesh(mesh=mesh, geometry=geometry)


def _aneurysm_geometry_points(geom: GeometryConfig) -> dict:
    """Analytic quantities (in meters) describing the vessel+sac outline."""

    L = geom.vessel_length_mm * MM_TO_M
    D = geom.vessel_diameter_mm * MM_TO_M
    R = 0.5 * geom.aneurysm_diameter_mm * MM_TO_M
    xc = 0.5 * L
    x_neck_right = xc + R
    x_neck_left = xc - R
    if x_neck_left < 0.0 or x_neck_right > L:
        raise ValueError(
            "Aneurysm diameter is too large relative to vessel_length_mm: "
            f"neck spans [{x_neck_left:.4f}, {x_neck_right:.4f}] m but the "
            f"vessel is only [0, {L:.4f}] m long."
        )
    return dict(L=L, D=D, R=R, xc=xc, x_neck_left=x_neck_left, x_neck_right=x_neck_right)


def _build_boundary_polygon(geom: GeometryConfig, mesh_cfg: MeshConfig, h: float) -> np.ndarray:
    """Ordered (CCW) boundary polygon vertices for the vessel+sac domain.

    Path: (0,0) -> (L,0) -> (L,D) -> along top wall to the right neck point
    -> over the sac arc (refined near both neck points) -> along top wall
    from the left neck point back to (0,D) -> (0,0).
    """

    g = _aneurysm_geometry_points(geom)
    L, D, R, xc = g["L"], g["D"], g["R"], g["xc"]
    xr, xl = g["x_neck_right"], g["x_neck_left"]
    n_min = mesh_cfg.min_boundary_points_per_edge

    def edge(p0, p1):
        n = max(n_min, int(round(np.linalg.norm(np.array(p1) - np.array(p0)) / h)))
        t = np.linspace(0.0, 1.0, n + 1)[:-1]  # drop last point, next edge continues from it
        return np.outer(1 - t, p0) + np.outer(t, p1)

    bottom = edge((0.0, 0.0), (L, 0.0))
    right = edge((L, 0.0), (L, D))
    top_right = edge((L, D), (xr, D))

    # Graded arc sampling: denser near theta=0 and theta=pi (the two necks).
    band = mesh_cfg.neck_band_fraction
    refine = mesh_cfg.neck_refinement_factor
    n_arc = max(3 * n_min, int(round(np.pi * R / h * refine)))
    u = np.linspace(0.0, 1.0, n_arc)
    # Graded map u -> theta in [0, pi], compressing points toward both ends
    # by an amount controlled by `band`/`refine` (a smooth-step warp).
    theta = np.pi * (
        u - (refine - 1.0) / (2.0 * np.pi * refine) * np.sin(2.0 * np.pi * u) * (2.0 * band)
    )
    theta = np.clip(theta, 0.0, np.pi)
    arc = np.column_stack([xc + R * np.cos(theta), D + R * np.sin(theta)])

    top_left = edge((xl, D), (0.0, D))
    left = edge((0.0, D), (0.0, 0.0))

    poly = np.vstack([bottom, right, top_right, arc, top_left, left])
    # Deduplicate consecutive coincident points (can happen at joins).
    keep = np.ones(len(poly), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(poly, axis=0), axis=1) > 1e-12
    return poly[keep]


def _polygon_path(boundary: np.ndarray) -> Path:
    """A closed `matplotlib.path.Path` for point-in-polygon containment tests.

    `boundary` is an open vertex loop (last vertex does not repeat the
    first); `Path` does not implicitly close such a loop for
    `contains_points`, so the closing edge is added explicitly here.
    """

    return Path(np.vstack([boundary, boundary[:1]]))


def _fill_interior_points(
    boundary: np.ndarray, geom: GeometryConfig, mesh_cfg: MeshConfig, h: float
) -> np.ndarray:
    """Interior fill points, smoothly graded from `h/neck_refinement_factor`
    near the proximal/distal neck to `h` elsewhere.

    Built as a single candidate point cloud (finest spacing everywhere) that
    is then greedily thinned to the locally-desired spacing. This -- rather
    than generating two separate grids (coarse background + a disjoint fine
    patch near the neck) -- guarantees the whole domain interior is covered
    with no density-mismatch gaps between grids, which previously produced
    small unmeshed slivers right at the neck that broke the domain's
    watertightness (and hence mass conservation in flow_solver.py).
    """

    g = _aneurysm_geometry_points(geom)
    D, R, xc = g["D"], g["R"], g["xc"]
    neck_pts = np.array([[xc - R, D], [xc + R, D]])

    path = _polygon_path(boundary)
    xmin, ymin = boundary.min(axis=0)
    xmax, ymax = boundary.max(axis=0)

    h_fine = h / mesh_cfg.neck_refinement_factor
    taper_radius = 6.0 * h  # distance over which spacing relaxes from h_fine back to h

    def local_spacing(pts: np.ndarray) -> np.ndarray:
        d = np.min(np.linalg.norm(pts[:, None, :] - neck_pts[None, :, :], axis=-1), axis=1)
        t = np.clip(d / taper_radius, 0.0, 1.0)
        smoothstep = t * t * (3.0 - 2.0 * t)
        return h_fine + smoothstep * (h - h_fine)

    xs = np.arange(xmin + 0.5 * h_fine, xmax, h_fine)
    ys = np.arange(ymin + 0.5 * h_fine, ymax, h_fine)
    gx, gy = np.meshgrid(xs, ys)
    candidates = np.column_stack([gx.ravel(), gy.ravel()])

    rng = np.random.default_rng(0)
    candidates += rng.uniform(-0.15 * h_fine, 0.15 * h_fine, size=candidates.shape)
    candidates = candidates[path.contains_points(candidates)]
    if len(candidates) == 0:
        return candidates

    spacing = local_spacing(candidates)

    # Drop points too close to the boundary (avoid slivers); the boundary
    # polygon itself already supplies resolution there.
    btree = cKDTree(boundary)
    dist_to_boundary, _ = btree.query(candidates)
    keep = dist_to_boundary > 0.5 * spacing
    candidates, spacing = candidates[keep], spacing[keep]
    if len(candidates) == 0:
        return candidates

    # Greedy Poisson-disk-style thinning using a uniform spatial hash (cell
    # size = h_fine) so neighbor lookups stay O(1) amortized rather than
    # rebuilding a KD-tree on every insertion.
    order = rng.permutation(len(candidates))
    cell = h_fine
    window = int(np.ceil(h / cell)) + 1
    grid: dict[tuple[int, int], list[int]] = {}
    accepted: list[int] = []

    for idx in order:
        pt = candidates[idx]
        r = spacing[idx]
        ci, cj = int(np.floor(pt[0] / cell)), int(np.floor(pt[1] / cell))
        too_close = False
        for di in range(-window, window + 1):
            if too_close:
                break
            for dj in range(-window, window + 1):
                bucket = grid.get((ci + di, cj + dj))
                if not bucket:
                    continue
                neighbors = candidates[bucket]
                if np.any(np.linalg.norm(neighbors - pt, axis=1) < 0.9 * r):
                    too_close = True
                    break
        if not too_close:
            grid.setdefault((ci, cj), []).append(idx)
            accepted.append(idx)

    return candidates[accepted]


def build_aneurysm_mesh(
    geometry_cfg: dict | GeometryConfig, mesh_cfg: dict | MeshConfig | None = None
) -> TaggedMesh:
    """Build the idealized 2D vessel+aneurysm mesh (see module docstring).

    Parameters
    ----------
    geometry_cfg:
        Either a ``GeometryConfig`` or a dict/OmegaConf mapping with keys
        matching one preset in ``configs/geometry.yaml`` (``vessel_diameter_mm``,
        ``aneurysm_diameter_mm``, ``vessel_length_mm``).
    mesh_cfg:
        Either a ``MeshConfig`` or the ``mesh:`` block of
        ``configs/geometry.yaml``. Defaults to ``MeshConfig()`` if omitted.
    """

    geom = geometry_cfg if isinstance(geometry_cfg, GeometryConfig) else GeometryConfig.from_preset(
        geometry_cfg
    )
    if mesh_cfg is None:
        mcfg = MeshConfig()
    elif isinstance(mesh_cfg, MeshConfig):
        mcfg = mesh_cfg
    else:
        mcfg = MeshConfig(**{k: v for k, v in dict(mesh_cfg).items() if k in MeshConfig.__dataclass_fields__})

    g = _aneurysm_geometry_points(geom)
    L, D, R = g["L"], g["D"], g["R"]
    approx_area = L * D + 0.5 * np.pi * R**2
    # Average equilateral-triangle area a = sqrt(3)/4 * h^2, so
    # h = sqrt(4 * area / (sqrt(3) * n_elements)).
    h = float(np.sqrt(4.0 * approx_area / (np.sqrt(3.0) * mcfg.target_num_elements)))

    boundary = _build_boundary_polygon(geom, mcfg, h)
    interior = _fill_interior_points(boundary, geom, mcfg, h)

    all_points = np.vstack([boundary, interior]) if len(interior) else boundary
    tri = Delaunay(all_points)

    path = _polygon_path(boundary)
    centroids = all_points[tri.simplices].mean(axis=1)
    inside = path.contains_points(centroids)
    # Guard against zero-area (degenerate/collinear) triangles.
    v0 = all_points[tri.simplices[:, 1]] - all_points[tri.simplices[:, 0]]
    v1 = all_points[tri.simplices[:, 2]] - all_points[tri.simplices[:, 0]]
    area2 = np.abs(v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0])
    valid = inside & (area2 > 1e-14 * h * h)

    simplices = tri.simplices[valid]
    used = np.unique(simplices)
    remap = -np.ones(len(all_points), dtype=int)
    remap[used] = np.arange(len(used))
    p = all_points[used].T
    t = remap[simplices].T

    mesh = MeshTri(p, t)

    tol = 1e-7 * max(L, D)
    xc, xl, xr = g["xc"], g["x_neck_left"], g["x_neck_right"]

    # Facet-midpoint tolerance for arc membership must accommodate the chord
    # sagitta (a straight facet between two points on the circle has its
    # midpoint strictly inside the circle by ~h^2/(8R)); use a tolerance
    # tied to the background mesh spacing `h` rather than machine precision.
    arc_tol = max(0.6 * h, 5 * tol)

    def is_on_arc(pts: np.ndarray) -> np.ndarray:
        r = np.hypot(pts[0] - xc, pts[1] - D)
        return (np.abs(r - R) < arc_tol) & (pts[1] >= D - tol)

    def is_inlet(pts: np.ndarray) -> np.ndarray:
        return np.isclose(pts[0], 0.0, atol=tol)

    def is_outlet(pts: np.ndarray) -> np.ndarray:
        return np.isclose(pts[0], L, atol=tol)

    def is_wall_sac(pts: np.ndarray) -> np.ndarray:
        return is_on_arc(pts) & ~is_inlet(pts) & ~is_outlet(pts)

    def is_wall_vessel(pts: np.ndarray) -> np.ndarray:
        # Everything on the boundary that is neither inlet, outlet, nor sac
        # wall is vessel wall. Defined as this complement (rather than an
        # independent geometric test, e.g. "y close to 0 or D") so that every
        # boundary facet is classified as exactly one of the four tags with
        # no gap -- independent geometric predicates for wall_sac/wall_vessel
        # can both narrowly miss the handful of small transitional facets
        # right at the neck corner, leaving them untagged (no BC enforced,
        # which silently breaks mass conservation).
        return ~is_inlet(pts) & ~is_outlet(pts) & ~is_wall_sac(pts)

    mesh = mesh.with_boundaries(
        {
            "inlet": is_inlet,
            "outlet": is_outlet,
            "wall_sac": is_wall_sac,
            "wall_vessel": is_wall_vessel,
        }
    )
    return TaggedMesh(mesh=mesh, geometry=geom)
