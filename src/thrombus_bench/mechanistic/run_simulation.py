"""CLI entrypoint: run one mechanistic simulation from a Hydra config.

Responsibility
---------------
Wire together `mesh.py`, `flow_solver.py`, and (once implemented)
`coupled_solver.py` into a single command:

    thrombus-run-simulation geometry=aneurysm_7mm physio=default

reading `configs/geometry.yaml` and `configs/physio_params.yaml` via Hydra,
running the full time-series simulation, and writing results to
`.npz`/`.h5` (mesh coordinates/connectivity, per-timestep velocity/pressure/
species/surface fields) for consumption by `data/generate_dataset.py` and
`viz/plots.py`.

Currently only the steady flow-only path is wired up (mesh + flow_solver),
usable for flow-field validation ahead of the full coupled model landing in
`coupled_solver.py`. Full transient thrombus simulation output is not yet
implemented -- this is a scaffolding stub pending `coupled_solver.py`.
"""

from __future__ import annotations

import argparse

import numpy as np

from thrombus_bench.mechanistic.flow_solver import CarreauParams, solve_steady_flow
from thrombus_bench.mechanistic.mesh import GeometryConfig, MeshConfig, build_aneurysm_mesh


def run_flow_only(geometry_cfg: dict, mesh_cfg: dict, carreau_cfg: dict, inlet_velocity_m_s: float, output_path: str) -> None:
    """Mesh + steady flow solve only, saved as .npz. A minimal working path
    ahead of the full coupled_solver.py transient simulation."""

    geom = GeometryConfig.from_preset(geometry_cfg)
    tagged_mesh = build_aneurysm_mesh(geom, MeshConfig(**mesh_cfg) if mesh_cfg else None)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vessel-diameter-mm", type=float, default=3.2)
    parser.add_argument("--aneurysm-diameter-mm", type=float, default=7.0)
    parser.add_argument("--vessel-length-mm", type=float, default=50.0)
    parser.add_argument("--inlet-velocity-cm-s", type=float, default=47.0)
    parser.add_argument("--target-num-elements", type=int, default=2000)
    parser.add_argument("--output", type=str, default="flow_solution.npz")
    args = parser.parse_args()

    geometry_cfg = {
        "vessel_diameter_mm": args.vessel_diameter_mm,
        "aneurysm_diameter_mm": args.aneurysm_diameter_mm,
        "vessel_length_mm": args.vessel_length_mm,
    }
    mesh_cfg = {"target_num_elements": args.target_num_elements}
    carreau_cfg = {"mu_inf_pa_s": 0.0035, "mu_0_pa_s": 0.056, "lambda_s": 3.313, "n": 0.3568}

    run_flow_only(
        geometry_cfg,
        mesh_cfg,
        carreau_cfg,
        inlet_velocity_m_s=args.inlet_velocity_cm_s / 100.0,
        output_path=args.output,
    )
    print(f"Wrote flow solution to {args.output}")


if __name__ == "__main__":
    main()
