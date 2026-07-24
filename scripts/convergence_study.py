"""Mesh/time-step self-convergence study for `mechanistic/coupled_solver.py`.

This is NOT validation against a reference/analytic solution (none exists
for this idealized geometry+physics combination -- see README.md
"Known limitations", "No quantitative validation against the paper's
reported results"). It is self-convergence: running the same physical
scenario at increasingly fine mesh resolution and increasingly small macro
time step, and checking whether summary quantities *stabilize* rather than
keep drifting -- evidence the discretization is fine enough that further
refinement wouldn't change the answer much, not evidence the answer is
"correct" in any absolute sense.

Grid: 3 mesh resolutions (`target_num_elements`) x 3 macro time steps
(`dt_s`), for each of the two paper geometries (`aneurysm_7mm`,
`aneurysm_10mm`, at their respective paper-reported inlet velocities,
`configs/geometry.yaml`).

Runtime: measured directly at the grid's most expensive corner (finest
mesh, smallest dt) on one dev machine before committing to the full grid
-- ~3.5-5.8s per run including mesh build (see this script's git history /
the task that added it for the exact numbers) -- so the full 2 x 3 x 3 = 18
runs finish in roughly a couple of minutes total, comfortably inside a
budget where asking before running would have just added latency. If you
change MESH_RESOLUTIONS/DT_VALUES/GEOMETRY_PRESETS to something larger,
re-time the worst corner yourself first.

Reported quantities, all from the *final* checkpoint of each run:
- velocity-field L2(Omega) norm (`_velocity_l2_norm`)
- pressure drop (inlet mean - outlet mean, `_pressure_drop`)
- peak wall shear rate (`_wall_shear_rate_nodal`'s nodal L2 projection,
  restricted to `state.wall_dofs`)
- total wall coverage (state.surface.M, arc-length-integrated separately
  over the top/bottom wall branches, `_total_wall_coverage`) -- an
  extensive PLT-count-like total (per unit out-of-plane depth), not the
  mean areal density
- domain-integrated resting/activated platelet "mass" (`_domain_integral`
  over `basis_c` of concentrations["RP"]/["AP"]) -- this project's model is
  a 2D idealization with no explicit out-of-plane depth, so this is
  concentration (PLT/mL) times an area (m^2), a convergence proxy, not a
  physically clean 3D platelet count
- max thrombin / max fibrin (nodal max of concentrations["T"]/["FI"]),
  flagged with `thrombin_fibrin_reliable` (Task 1.4/README "Known
  limitations": the concentration-cap safety clip may have bound during
  the run, in which case [T]/[FI] should not be read as physically
  meaningful, only as a numerically-bounded proxy)

Output: a Markdown table per geometry, written to
`verification/convergence_study.md` (not `results/`, which is gitignored
except `.gitkeep` -- see README.md "Repository layout" -- since this is a
committed reference artifact, not a regenerable benchmark report).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import yaml
from skfem import Functional
from skfem.helpers import dot

from thrombus_bench.mechanistic.coupled_solver import (
    _wall_branches,
    _wall_shear_rate_nodal,
    run_coupled_simulation,
)
from thrombus_bench.mechanistic.mesh import GeometryConfig, build_aneurysm_mesh

_ALL_SPECIES = ("RP", "AP", "APR", "APS", "T", "AT", "PT", "FG", "FI")

MESH_RESOLUTIONS = (400, 800, 1600)
DT_VALUES = (0.2, 0.1, 0.05)
END_TIME_S = 2.0
GEOMETRY_PRESETS = ("aneurysm_7mm", "aneurysm_10mm")


@Functional
def _velocity_sq_form(w):
    return dot(w["u"], w["u"])


@Functional
def _total_amount_form(w):
    return w["c"]


def _velocity_l2_norm(flow) -> float:
    """sqrt(integral_Omega |u|^2 dA), the standard FEM self-convergence
    velocity norm."""

    return float(np.sqrt(_velocity_sq_form.assemble(flow.basis_u, u=flow.basis_u.interpolate(flow.u))))


def _pressure_drop(flow) -> float:
    inlet_dofs = flow.basis_p.get_dofs("inlet").all()
    outlet_dofs = flow.basis_p.get_dofs("outlet").all()
    return float(flow.p[inlet_dofs].mean() - flow.p[outlet_dofs].mean())


def _domain_integral(basis_c, values: np.ndarray) -> float:
    return float(_total_amount_form.assemble(basis_c, c=basis_c.interpolate(values)))


def _total_wall_coverage(basis_c, wall_dofs: np.ndarray, top_branch: np.ndarray, bottom_branch: np.ndarray, M_values: np.ndarray) -> float:
    """Arc-length (trapezoidal) integral of `M_values` (aligned positionally
    with `wall_dofs`, per `CoupledSimulationState`) over the wall boundary,
    summed across the top and bottom branches independently (mirroring
    `coupled_solver._axial_shear_gradient`'s branch split, since the wall
    isn't a single x-monotonic curve).

    `wall_dofs` is `np.union1d`-sorted (ascending global DOF index), NOT
    arc-length ordered, so `M_values[k]` corresponds to global DOF
    `wall_dofs[k]` -- `np.searchsorted` recovers each branch's (x-sorted)
    values from that positional alignment before integrating."""

    total = 0.0
    for branch in (top_branch, bottom_branch):
        if len(branch) < 2:
            continue
        idx_in_wall_dofs = np.searchsorted(wall_dofs, branch)
        vals = M_values[idx_in_wall_dofs]
        coords_m = basis_c.doflocs[:, branch]
        seg_len_cm = np.hypot(np.diff(coords_m[0]), np.diff(coords_m[1])) * 100.0
        avg_vals = 0.5 * (vals[:-1] + vals[1:])
        total += float(np.sum(avg_vals * seg_len_cm))
    return total


def run_one(
    geometry_cfg: dict, mesh_cfg_base: dict, physio: dict, target_num_elements: int, dt_s: float, end_time_s: float
) -> dict:
    geom = GeometryConfig.from_preset(geometry_cfg)
    mesh_cfg = dict(mesh_cfg_base)
    mesh_cfg["target_num_elements"] = target_num_elements
    inlet_velocity_m_s = geometry_cfg["inlet_velocity_cm_s"] / 100.0

    t0 = time.perf_counter()
    tagged_mesh = build_aneurysm_mesh(geom, mesh_cfg)
    n_macro_steps = max(1, int(round(end_time_s / dt_s)))
    history = run_coupled_simulation(
        tagged_mesh, inlet_velocity_m_s=inlet_velocity_m_s, physio=physio,
        end_time_s=end_time_s, dt_s=dt_s,
        output_every_n_steps=n_macro_steps, flow_resolve_every_n_steps=n_macro_steps,
    )
    elapsed_s = time.perf_counter() - t0

    final = history.states[-1]
    basis_c = history.basis_c
    vessel_diameter_m = geom.vessel_diameter_mm * 1.0e-3
    top_branch, bottom_branch = _wall_branches(basis_c, vessel_diameter_m)
    gamma_nodal = _wall_shear_rate_nodal(basis_c, final.flow)

    return {
        "target_num_elements": target_num_elements,
        "n_elements_actual": int(tagged_mesh.mesh.t.shape[1]),
        "dt_s": dt_s,
        "elapsed_s": elapsed_s,
        "velocity_l2_norm": _velocity_l2_norm(final.flow),
        "pressure_drop_pa": _pressure_drop(final.flow),
        "peak_wall_shear_rate_s_inv": float(gamma_nodal[final.wall_dofs].max()),
        "total_wall_M_plt_per_cm": _total_wall_coverage(basis_c, final.wall_dofs, top_branch, bottom_branch, final.surface.M),
        "integrated_RP": _domain_integral(basis_c, final.concentrations["RP"]),
        "integrated_AP": _domain_integral(basis_c, final.concentrations["AP"]),
        "max_T_uM": float(final.concentrations["T"].max()),
        "max_FI_uM": float(final.concentrations["FI"].max()),
        "thrombin_fibrin_reliable": bool(history.thrombin_fibrin_reliable),
    }


def _format_row(r: dict) -> str:
    reliable = "yes" if r["thrombin_fibrin_reliable"] else "**NO** (cap hit)"
    return (
        f"| {r['target_num_elements']} ({r['n_elements_actual']}) | {r['dt_s']:g} | {r['elapsed_s']:.2f} | "
        f"{r['velocity_l2_norm']:.4e} | {r['pressure_drop_pa']:.4e} | {r['peak_wall_shear_rate_s_inv']:.4e} | "
        f"{r['total_wall_M_plt_per_cm']:.4e} | {r['integrated_RP']:.4e} | {r['integrated_AP']:.4e} | "
        f"{r['max_T_uM']:.4e} | {r['max_FI_uM']:.4e} | {reliable} |"
    )


_HEADER = (
    "| Mesh (target / actual elements) | dt (s) | Wall time (s) | Velocity L2 norm | "
    "Pressure drop (Pa) | Peak wall shear (s⁻¹) | Total wall M (PLT/cm) | ∫RP (PLT·m²/mL) | "
    "∫AP (PLT·m²/mL) | Max [T] (µM) | Max [FI] (µM) | [T]/[FI] reliable? |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-config", type=str, default="configs/geometry.yaml")
    parser.add_argument("--physio-config", type=str, default="configs/physio_params.yaml")
    parser.add_argument("--output", type=str, default="verification/convergence_study.md")
    parser.add_argument("--end-time-s", type=float, default=END_TIME_S)
    args = parser.parse_args()

    with open(args.geometry_config) as f:
        geometry_yaml = yaml.safe_load(f)
    with open(args.physio_config) as f:
        physio = yaml.safe_load(f)
    mesh_cfg_base = dict(geometry_yaml["mesh"])

    sections = []
    grid_t0 = time.perf_counter()
    for preset_name in GEOMETRY_PRESETS:
        geometry_cfg = geometry_yaml["presets"][preset_name]
        rows = []
        for target_num_elements in MESH_RESOLUTIONS:
            for dt_s in DT_VALUES:
                print(f"[{preset_name}] target_num_elements={target_num_elements}, dt_s={dt_s} ...", flush=True)
                r = run_one(geometry_cfg, mesh_cfg_base, physio, target_num_elements, dt_s, args.end_time_s)
                print(f"  -> {r['elapsed_s']:.2f}s, n_elements_actual={r['n_elements_actual']}", flush=True)
                rows.append(r)
        sections.append((preset_name, geometry_cfg, rows))
    grid_elapsed = time.perf_counter() - grid_t0
    print(f"Total grid runtime: {grid_elapsed:.1f}s", flush=True)

    lines = [
        "# Mechanistic solver: mesh / time-step self-convergence study",
        "",
        "Self-convergence only (no analytic/reference solution exists for this "
        "idealized geometry+physics combination) -- see `scripts/convergence_study.py`'s "
        "module docstring for methodology and column definitions, and README.md "
        '"Known limitations" for the project-wide validation caveat. Read each '
        "table top-to-bottom-right: values should stabilize (stop changing "
        "much) as mesh resolution increases and dt decreases, not match any "
        "particular target value.",
        "",
        f"End time: {args.end_time_s:g} s. Total grid runtime: {grid_elapsed:.1f} s "
        f"({len(GEOMETRY_PRESETS) * len(MESH_RESOLUTIONS) * len(DT_VALUES)} runs).",
        "",
        "**Peak wall shear rate is a pointwise (max-over-nodes) quantity near the "
        "proximal/distal neck's geometric corners, where the true continuum shear "
        "field is singular/near-singular** -- it can converge much more slowly, and "
        "less monotonically, with mesh refinement than the integrated/domain "
        "quantities (velocity L2 norm, total wall M, ∫RP/∫AP) in the same table. "
        "A large jump in that one column between rows is expected mesh sensitivity "
        "at a sharp corner, not necessarily a sign the rest of the run is unreliable.",
        "",
    ]
    for preset_name, geometry_cfg, rows in sections:
        lines.append(
            f"## `{preset_name}` (vessel {geometry_cfg['vessel_diameter_mm']} mm / "
            f"aneurysm {geometry_cfg['aneurysm_diameter_mm']} mm, "
            f"inlet velocity {geometry_cfg['inlet_velocity_cm_s']} cm/s)"
        )
        lines.append("")
        lines.append(_HEADER)
        for r in rows:
            lines.append(_format_row(r))
        lines.append("")

        max_t_vals = [r["max_T_uM"] for r in rows]
        max_fi_vals = [r["max_FI_uM"] for r in rows]
        if not any(r["thrombin_fibrin_reliable"] for r in rows) and np.allclose(max_t_vals, max_t_vals[0]) and np.allclose(max_fi_vals, max_fi_vals[0]):
            lines.append(
                "**Note:** every run above hit the `[T]`/`[FI]` concentration-cap safety "
                "clip (`thrombin_fibrin_reliable=False`), and both columns are identical "
                "to 4 significant figures across the whole grid -- that's the clip ceiling, "
                'not a converged physical value. See README.md "Known limitations"; these '
                "two columns cannot be used to assess convergence here."
            )
            lines.append("")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
