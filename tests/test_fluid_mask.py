"""Tests for data/generate_dataset.py's _fluid_mask, which flags which cells
of _rasterize's bounding-box grid actually fall inside the FEM mesh domain
(the vessel+aneurysm domain is an L/T-shaped union, so the axis-aligned
bounding box used for rasterization contains genuine exterior cells that
griddata(method="nearest") would otherwise silently fill with a nearby
in-domain node's value -- see this repo's README "Known limitations")."""

from __future__ import annotations

import numpy as np

from thrombus_bench.data.generate_dataset import _fluid_mask
from thrombus_bench.mechanistic.mesh import GeometryConfig, MeshConfig, build_aneurysm_mesh

# Small, fast geometry/mesh -- matches other tests' scale (~150 elements).
_GEOM = GeometryConfig(vessel_diameter_mm=3.2, aneurysm_diameter_mm=7.0, vessel_length_mm=50.0)
_MESH_CFG = MeshConfig(target_num_elements=150)


def test_fluid_mask_marks_vessel_interior_true_and_bbox_corners_false():
    tagged_mesh = build_aneurysm_mesh(_GEOM, _MESH_CFG)
    node_coords = tagged_mesh.mesh.p
    triangles = tagged_mesh.mesh.t
    grid_size = (16, 16)

    mask = _fluid_mask(node_coords, triangles, grid_size)

    assert mask.shape == grid_size
    assert mask.dtype == np.float32
    assert set(np.unique(mask)) <= {0.0, 1.0}

    xmin, ymin = node_coords.min(axis=1)
    xmax, ymax = node_coords.max(axis=1)
    xs = np.linspace(xmin, xmax, grid_size[1])
    ys = np.linspace(ymin, ymax, grid_size[0])

    # A point well inside the vessel rectangle -- away from the sac (which
    # is centered on the domain's x-midpoint) and away from the walls.
    L_m = _GEOM.vessel_length_mm * 1e-3
    D_m = _GEOM.vessel_diameter_mm * 1e-3
    vessel_interior = (0.5 * L_m, 0.25 * D_m)
    j = int(np.argmin(np.abs(xs - vessel_interior[0])))
    i = int(np.argmin(np.abs(ys - vessel_interior[1])))
    assert mask[i, j] == 1.0

    # The bounding box's top-left/top-right corners: above the vessel's top
    # wall (y > D) but far in x from the sac's footprint (which only bulges
    # above the vessel near the domain's x-midpoint) -- genuinely exterior.
    assert mask[-1, 0] == 0.0
    assert mask[-1, -1] == 0.0
