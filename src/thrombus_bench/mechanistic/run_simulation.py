"""CLI entrypoint: run one mechanistic simulation from a config, save as .npz.

Responsibility
---------------
Wire together `mesh.py`, `flow_solver.py`, and `coupled_solver.py` into a
single command, reading `configs/geometry.yaml` and `configs/physio_params.yaml`,
running the full transient simulation, and writing the time-series output
(mesh coordinates/connectivity, per-checkpoint velocity/pressure/species/
surface fields) as a `.npz` archive for `data/generate_dataset.py` and
`viz/plots.py` to consume.
"""

from __future__ import annotations

import argparse

import numpy as np
import yaml

from thrombus_bench.mechanistic.coupled_solver import run_coupled_simulation
from thrombus_bench.mechanistic.flow_solver import CarreauParams, solve_steady_flow
from thrombus_bench.mechanistic.mesh import GeometryConfig, MeshConfig, build_aneurysm_mesh

_ALL_SPECIES = ("RP", "AP", "APR", "APS", "T", "AT", "PT", "FG", "FI")


def run_flow_only(geometry_cfg: dict, mesh_cfg: dict, carreau_cfg: dict, inlet_velocity_m_s: float, output_path: str) -> None:
    """Mesh + steady flow solve only, saved as .npz. Useful for quick
    flow-field verification without running the full transient model."""

    geom = GeometryConfig.from_preset(geometry_cfg)
    tagged_mesh = build_aneurysm_mesh(geom, mesh_cfg)
    carreau = CarreauParams.from_config(carreau_cfg)
    flow = solve_steady_flow(tagged_mesh, inlet_velocity_m_s=inlet_velocity_m_s, carreau=carreau)

    np.savez(
        output_path,
        node_coords=tagged_mesh.mesh.p,
        elements=tagged_mesh.mesh.t,
        velocity_dofs=flow.u,
        pressure_dofs=flow.p,
        converged=flow.converged,
        n_iterations=flow.n_iterations,
    )


def run_full_simulation(
    geometry_cfg: dict,
    mesh_cfg: dict,
    physio: dict,
    inlet_velocity_m_s: float,
    end_time_s: float,
    dt_s: float,
    output_path: str,
    output_every_n_steps: int = 5,
    flow_resolve_every_n_steps: int = 5,
) -> None:
    """Full transient mechanistic simulation, saved as .npz.

    Saved arrays: mesh (`node_coords`, `elements`), per-checkpoint
    `times_s`, `velocity_{i}`/`pressure_{i}` (flow), `conc_{species}_{i}`
    (species concentrations), `surface_M_{i}`/`surface_Mat_{i}` (wall
    surface coverage, aligned with `wall_dofs`), where `{i}` indexes
    checkpoints in `times_s`.
    """

    geom = GeometryConfig.from_preset(geometry_cfg)
    tagged_mesh = build_aneurysm_mesh(geom, mesh_cfg)

    history = run_coupled_simulation(
        tagged_mesh,
        inlet_velocity_m_s=inlet_velocity_m_s,
        physio=physio,
        end_time_s=end_time_s,
        dt_s=dt_s,
        output_every_n_steps=output_every_n_steps,
        flow_resolve_every_n_steps=flow_resolve_every_n_steps,
    )

    save_dict = {
        "node_coords": tagged_mesh.mesh.p,
        "elements": tagged_mesh.mesh.t,
        "wall_dofs": history.states[0].wall_dofs,
        "times_s": np.array([s.time_s for s in history.states]),
    }
    for i, s in enumerate(history.states):
        save_dict[f"velocity_{i}"] = s.flow.u
        save_dict[f"pressure_{i}"] = s.flow.p
        for name in _ALL_SPECIES:
            save_dict[f"conc_{name}_{i}"] = s.concentrations[name]
        save_dict[f"surface_M_{i}"] = s.surface.M
        save_dict[f"surface_Mat_{i}"] = s.surface.M_at

    np.savez(output_path, **save_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-preset", type=str, default="aneurysm_7mm", choices=["aneurysm_7mm", "aneurysm_10mm"])
    parser.add_argument("--geometry-config", type=str, default="configs/geometry.yaml")
    parser.add_argument("--physio-config", type=str, default="configs/physio_params.yaml")
    parser.add_argument("--target-num-elements", type=int, default=1500)
    parser.add_argument("--end-time-s", type=float, default=2.0)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--flow-only", action="store_true", help="Run only the steady flow solve, skip species/surface transport.")
    parser.add_argument("--output", type=str, default="simulation.npz")
    args = parser.parse_args()

    with open(args.geometry_config) as f:
        geometry_yaml = yaml.safe_load(f)
    geometry_cfg = geometry_yaml["presets"][args.geometry_preset]
    inlet_velocity_m_s = geometry_cfg["inlet_velocity_cm_s"] / 100.0
    mesh_cfg = dict(geometry_yaml["mesh"])
    mesh_cfg["target_num_elements"] = args.target_num_elements

    with open(args.physio_config) as f:
        physio = yaml.safe_load(f)
    carreau_cfg = physio["fluid"]["carreau"]

    if args.flow_only:
        run_flow_only(geometry_cfg, mesh_cfg, carreau_cfg, inlet_velocity_m_s, args.output)
    else:
        run_full_simulation(
            geometry_cfg, mesh_cfg, physio, inlet_velocity_m_s,
            end_time_s=args.end_time_s, dt_s=args.dt_s, output_path=args.output,
        )
    print(f"Wrote simulation output to {args.output}")


if __name__ == "__main__":
    main()
