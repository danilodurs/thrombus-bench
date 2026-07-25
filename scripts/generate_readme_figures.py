"""One-off script: render the README's illustrative figures.

Not part of the installed package or test suite -- run manually whenever
the README's figures need regenerating (e.g. after a geometry/solver
change that would make the images stale). Writes PNGs to `docs/figures/`.

Every figure here is a genuine rendering of this project's own mechanistic
solver output (steady Carreau-viscosity Stokes flow, idealized 2D
geometry) -- no fabricated or schematic artwork. See README.md's "Gallery"
section for the caveats these figures are captioned with (Stokes not
Navier-Stokes, qualitative not quantitative, wall-vs-bulk field
distinction).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import yaml

from thrombus_bench.mechanistic.flow_solver import CarreauParams, solve_steady_flow, wall_traction
from thrombus_bench.mechanistic.mesh import GeometryConfig, MeshConfig, build_aneurysm_mesh
from thrombus_bench.viz.field_interpolation import interpolate_velocity_to_grid
from thrombus_bench.viz.showcase_plots import plot_mesh_quality, plot_streamlines, plot_vorticity_field, plot_wall_traction_map

OUT_DIR = "docs/figures"


def _load_carreau() -> CarreauParams:
    with open("configs/physio_params.yaml") as f:
        physio = yaml.safe_load(f)
    return CarreauParams.from_config(physio["fluid"]["carreau"])


def _load_geometry_yaml() -> dict:
    with open("configs/geometry.yaml") as f:
        return yaml.safe_load(f)


def geometry_gallery() -> None:
    """Mesh-quality panels for both paper presets plus one asymmetric
    half-ellipse variant (geometry-redesign Phase 4b), demonstrating the
    parametrization without needing a new config file."""

    geometry_yaml = _load_geometry_yaml()
    mesh_cfg = MeshConfig(target_num_elements=2000)

    panels = [
        ("aneurysm_7mm preset\n(3.2 mm vessel / 7 mm sac)", GeometryConfig.from_preset(geometry_yaml["presets"]["aneurysm_7mm"])),
        ("aneurysm_10mm preset\n(4 mm vessel / 10 mm sac)", GeometryConfig.from_preset(geometry_yaml["presets"]["aneurysm_10mm"])),
        (
            "asymmetric variant (illustrative)\n(sac_height_mm=5, sac_asymmetry=0.4)",
            GeometryConfig(vessel_diameter_mm=3.2, aneurysm_diameter_mm=7.0, vessel_length_mm=50.0, sac_height_mm=5.0, sac_asymmetry=0.4),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 2.7), constrained_layout=True)
    for ax, (title, geom) in zip(axes, panels):
        tm = build_aneurysm_mesh(geom, mesh_cfg)
        tpc = plot_mesh_quality(tm.mesh.p, tm.mesh.t, ax=ax, geom=geom, inset=True)
        ax.set_title(title, fontsize=10)
    fig.colorbar(tpc, ax=axes, shrink=0.7, label="min triangle angle [deg]", location="bottom", pad=0.02, aspect=40)
    fig.suptitle("Idealized 2D vessel + aneurysm geometry (self-contained Delaunay mesher)", fontsize=12)
    fig.savefig(f"{OUT_DIR}/geometry_gallery.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def flow_field_gallery() -> None:
    """Streamlines, vorticity, and wall traction for a steady Carreau-Stokes
    solve on the aneurysm_7mm preset -- the flow field the whole mechanical-
    aggregation mechanism is gated on (negative wall shear-rate gradient)."""

    geometry_yaml = _load_geometry_yaml()
    preset = geometry_yaml["presets"]["aneurysm_7mm"]
    geom = GeometryConfig.from_preset(preset)
    mesh_cfg = MeshConfig(target_num_elements=2000)
    tm = build_aneurysm_mesh(geom, mesh_cfg)
    carreau = _load_carreau()
    inlet_velocity_m_s = preset["inlet_velocity_cm_s"] / 100.0
    flow = solve_steady_flow(tm, inlet_velocity_m_s=inlet_velocity_m_s, carreau=carreau)

    node_coords = tm.mesh.p  # (2, n_vertices) -- P1 mesh vertices, a prefix of the P2 velocity dofs
    n_vertices = node_coords.shape[1]
    ux = flow.u[0 : 2 * n_vertices : 2]
    uy = flow.u[1 : 2 * n_vertices : 2]
    grid = interpolate_velocity_to_grid(node_coords, ux, uy, tm.mesh.t, grid_size=(90, 220))
    # Both wall boundaries, not just "wall_vessel": the sac wall ("wall_sac")
    # is where the shear-gradient-driven mechanical aggregation mechanism
    # actually operates, so it's the more interesting half of this figure.
    traction_vessel = wall_traction(flow, carreau, "wall_vessel")
    traction_sac = wall_traction(flow, carreau, "wall_sac")
    traction = {k: np.concatenate([traction_vessel[k], traction_sac[k]]) for k in traction_vessel}

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), constrained_layout=True)
    stream_lines = plot_streamlines(grid, ax=axes[0])
    fig.colorbar(stream_lines, ax=axes[0], shrink=0.85, label="speed [m/s]")
    vort_mesh = plot_vorticity_field(grid, ax=axes[1])
    fig.colorbar(vort_mesh, ax=axes[1], shrink=0.85, label="vorticity [1/s]")
    traction_sc = plot_wall_traction_map(traction, ax=axes[2])
    fig.colorbar(traction_sc, ax=axes[2], shrink=0.85, label="traction magnitude [Pa]")
    fig.suptitle(
        "Steady Stokes flow, Carreau viscosity -- aneurysm_7mm preset "
        f"(inlet {inlet_velocity_m_s:.2f} m/s, converged={flow.converged})",
        fontsize=12,
    )
    fig.savefig(f"{OUT_DIR}/flow_field_gallery.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    geometry_gallery()
    print(f"wrote {OUT_DIR}/geometry_gallery.png")
    flow_field_gallery()
    print(f"wrote {OUT_DIR}/flow_field_gallery.png")
