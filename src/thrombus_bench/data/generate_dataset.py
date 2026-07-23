"""Batch-run the mechanistic solver over sampled parameters to build the surrogate dataset.

Responsibility
---------------
For each parameter sample from `sampler.py`:

1. Build the mesh (`mechanistic/mesh.py`) for the sampled geometry.
2. Run the full coupled mechanistic simulation (`mechanistic/coupled_solver.py`)
   for the sampled physiological parameters and inlet velocity.
3. Rasterize the final-checkpoint velocity/pressure/species fields onto a
   fixed-resolution regular grid (matching `configs/training.yaml`
   `model.encoder.latent_grid_size`) via nearest-neighbor interpolation
   (`scipy.interpolate.griddata`), since the FEM mesh's node count/
   connectivity varies per sample but the neural surrogate needs a
   fixed-shape input/target.
4. Save each sample's rasterized fields + parameter vector + summary scalars
   to `data/processed/{split}/sample_NNN.npz`: `max_M_at`,
   `thrombosed_fraction`, `converged` (flow Picard convergence),
   `flow_n_iterations`/`flow_residual` (`mechanistic/flow_solver.
   FlowSolution`'s final-checkpoint Picard diagnostics), `thrombin_fibrin_
   reliable` (see `mechanistic/coupled_solver.CoupledSimulationHistory`
   docstring; False whenever the run's [T]/[FI] concentration-cap safety
   clip actually bound, flagging that sample's T/FI -- and downstream
   FI-derived fields -- as not physically trustworthy rather than silently
   capping them), `clip_count_{species}` (per-species cumulative
   concentration-cap clip-event count over the whole run, from
   `CoupledSimulationHistory.clip_event_counts`), and `conc_{species}_min`/
   `conc_{species}_max` (raw nodal field extrema, pre-rasterization -- a
   cheap way to spot NaN/Inf/out-of-range values without scanning the full
   grid). This is this project's only per-sample QC signal beyond visual
   inspection; none of it is consumed by training/evaluation automatically
   -- see `data/dataset.py`'s docstring for how to read it back.
5. Route each sample to train/val/test/edge_holdout per
   `sampler.split_train_val_test_edge_holdout`.

Scope note: runs serially (no multiprocessing) -- scikit-fem `Basis`/`Mesh`
objects are not trivially picklable across a `multiprocessing.Pool`, and
given this project's reduced-scale demo dataset (see README.md "Project
status"), serial execution is fast enough in practice.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import yaml
from scipy.interpolate import griddata

from thrombus_bench.mechanistic.coupled_solver import run_coupled_simulation
from thrombus_bench.mechanistic.mesh import GeometryConfig, MeshConfig, build_aneurysm_mesh

from .sampler import ParameterSpace, latin_hypercube_sample, split_train_val_test_edge_holdout

_ALL_SPECIES = ("RP", "AP", "APR", "APS", "T", "AT", "PT", "FG", "FI")
PARAM_ORDER = (
    "aneurysm_diameter_mm", "vessel_diameter_mm", "inlet_velocity_cm_s", "platelet_conc_plt_ml",
    "heparin_conc_uM", "prothrombin_uM", "antithrombin_uM", "fibrinogen_uM",
)


def _rasterize(node_coords: np.ndarray, values: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor rasterization of a nodal field onto a regular grid
    covering the node coordinates' bounding box."""

    xmin, ymin = node_coords.min(axis=1)
    xmax, ymax = node_coords.max(axis=1)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, grid_size[1]), np.linspace(ymin, ymax, grid_size[0]))
    grid = griddata(node_coords.T, values, (gx, gy), method="nearest")
    return grid


def _run_one_sample(sample: dict, physio_base: dict, mesh_cfg: dict, end_time_s: float, dt_s: float, grid_size: tuple[int, int]) -> dict:
    geom = GeometryConfig(
        vessel_diameter_mm=sample["vessel_diameter_mm"],
        aneurysm_diameter_mm=sample["aneurysm_diameter_mm"],
        vessel_length_mm=50.0,
    )
    tagged_mesh = build_aneurysm_mesh(geom, mesh_cfg)

    physio = {k: (dict(v) if isinstance(v, dict) else v) for k, v in physio_base.items()}
    physio["species"] = dict(physio_base["species"])
    physio["species"]["resting_platelets_inlet_plt_ml"] = sample["platelet_conc_plt_ml"]
    physio["species"]["prothrombin_inlet_uM"] = sample["prothrombin_uM"]
    physio["species"]["antithrombin_inlet_uM"] = sample["antithrombin_uM"]
    physio["species"]["fibrinogen_inlet_uM"] = sample["fibrinogen_uM"]
    physio["heparin"] = dict(physio_base["heparin"])
    physio["heparin"]["concentration_uM"] = sample["heparin_conc_uM"]

    inlet_velocity_m_s = sample["inlet_velocity_cm_s"] / 100.0
    history = run_coupled_simulation(
        tagged_mesh, inlet_velocity_m_s=inlet_velocity_m_s, physio=physio,
        end_time_s=end_time_s, dt_s=dt_s, output_every_n_steps=max(1, int(round(end_time_s / dt_s))),
        flow_resolve_every_n_steps=max(1, int(round(end_time_s / dt_s))),
    )
    final = history.states[-1]
    node_coords = tagged_mesh.mesh.p

    n_vertices = node_coords.shape[1]
    ux = final.flow.u[0 : 2 * n_vertices : 2]
    uy = final.flow.u[1 : 2 * n_vertices : 2]

    fields = {
        "velocity_x": _rasterize(node_coords, ux, grid_size),
        "velocity_y": _rasterize(node_coords, uy, grid_size),
    }
    conc_min_max = {}
    for name in _ALL_SPECIES:
        nodal = final.concentrations[name][:n_vertices]
        fields[f"conc_{name}"] = _rasterize(node_coords, nodal, grid_size)
        # Cheap min/max of the raw (pre-rasterization) nodal field -- lets a
        # QC pass spot NaN/Inf or wildly out-of-range values at a glance
        # without having to load and scan the full rasterized grid.
        conc_min_max[f"conc_{name}_min"] = float(np.min(nodal))
        conc_min_max[f"conc_{name}_max"] = float(np.max(nodal))

    params = np.array([sample[name] for name in PARAM_ORDER], dtype=np.float64)
    M_at_critical = physio["adhesion_aggregation"]["M_at_critical_plt_cm2"]
    max_M_at = float(final.surface.M_at.max())
    thrombosed_fraction = float(np.mean(final.surface.M_at >= M_at_critical))

    clip_counts = {f"clip_count_{name}": int(history.clip_event_counts[name]) for name in _ALL_SPECIES}

    return {
        "params": params,
        "max_M_at": max_M_at,
        "thrombosed_fraction": thrombosed_fraction,
        "converged": bool(final.flow.converged),
        "flow_n_iterations": int(final.flow.n_iterations),
        "flow_residual": float(final.flow.residual),
        "thrombin_fibrin_reliable": bool(history.thrombin_fibrin_reliable),
        **clip_counts,
        **conc_min_max,
        **fields,
    }


def _new_qc_summary() -> dict:
    return {
        "n_samples": 0,
        "n_converged": 0,
        "n_thrombin_fibrin_reliable": 0,
        "n_samples_clip_hit": {name: 0 for name in _ALL_SPECIES},
    }


def _update_qc_summary(qc: dict, result: dict) -> None:
    """Accumulate one sample's QC fields (see `_run_one_sample`'s return
    dict) into the running per-run summary `main()` prints -- sample-level
    counts (how many samples were affected), not raw per-node clip-event
    totals."""

    qc["n_samples"] += 1
    qc["n_converged"] += int(result["converged"])
    qc["n_thrombin_fibrin_reliable"] += int(result["thrombin_fibrin_reliable"])
    for name in _ALL_SPECIES:
        if result[f"clip_count_{name}"] > 0:
            qc["n_samples_clip_hit"][name] += 1


def generate_dataset(
    config: dict,
    physio_base: dict,
    geometry_mesh_cfg: dict,
    output_dir: str,
    end_time_s: float = 1.0,
    dt_s: float = 0.1,
    grid_size: tuple[int, int] = (32, 32),
) -> tuple[dict[str, int], dict]:
    """Run `sampler.latin_hypercube_sample` + `split_train_val_test_edge_holdout`,
    then batch-run the mechanistic solver over every sample, writing results
    under `output_dir/{train,val,test,edge_holdout}/`. Returns
    `(counts, qc_summary)`: the number of samples written per split, and an
    aggregate QC summary (see `_new_qc_summary`) across all splits combined,
    for `main()`'s printout -- see `_run_one_sample`'s docstring for what
    each per-sample QC field means."""

    space = ParameterSpace()
    n_total = config["n_train"] + config["n_val"] + config["n_test"] + config["n_edge_holdout"]
    samples = latin_hypercube_sample(space, n_total, seed=config.get("seed", 0))
    splits = split_train_val_test_edge_holdout(
        samples, space, edge_holdout_quantile=config["edge_holdout_quantile"],
        n_train=config["n_train"], n_val=config["n_val"], n_test=config["n_test"],
        n_edge_holdout=config["n_edge_holdout"],
        seed=config.get("seed", 0),
    )

    counts = {}
    qc_summary = _new_qc_summary()
    for split_name, split_samples in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for i, sample in enumerate(split_samples):
            result = _run_one_sample(sample, physio_base, geometry_mesh_cfg, end_time_s, dt_s, grid_size)
            np.savez(os.path.join(split_dir, f"sample_{i:04d}.npz"), **result)
            _update_qc_summary(qc_summary, result)
        counts[split_name] = len(split_samples)
    return counts, qc_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=str, default="configs/training.yaml")
    parser.add_argument("--physio-config", type=str, default="configs/physio_params.yaml")
    parser.add_argument("--geometry-config", type=str, default="configs/geometry.yaml")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--end-time-s", type=float, default=1.0)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--target-num-elements", type=int, default=800)
    parser.add_argument("--grid-size", type=int, default=32)
    args = parser.parse_args()

    with open(args.training_config) as f:
        training_cfg = yaml.safe_load(f)
    with open(args.physio_config) as f:
        physio_base = yaml.safe_load(f)
    with open(args.geometry_config) as f:
        geometry_yaml = yaml.safe_load(f)
    mesh_cfg = dict(geometry_yaml["mesh"])
    mesh_cfg["target_num_elements"] = args.target_num_elements

    counts, qc = generate_dataset(
        training_cfg["data"], physio_base, mesh_cfg, args.output_dir,
        end_time_s=args.end_time_s, dt_s=args.dt_s, grid_size=(args.grid_size, args.grid_size),
    )
    print(f"Wrote dataset to {args.output_dir}: {counts}")
    _print_qc_summary(qc)


def _print_qc_summary(qc: dict) -> None:
    n = qc["n_samples"]
    print(f"\nQC summary ({n} samples total):")
    print(f"  {qc['n_converged']}/{n} samples converged (flow_solver Picard iteration).")
    print(
        f"  {n - qc['n_thrombin_fibrin_reliable']}/{n} samples hit the concentration cap for "
        "[T] or [FI] at some point (thrombin_fibrin_reliable=False -- see README.md "
        '"Known limitations").'
    )
    clip_hit = {name: count for name, count in qc["n_samples_clip_hit"].items() if count > 0}
    if clip_hit:
        per_species = ", ".join(f"{name}: {count}/{n}" for name, count in clip_hit.items())
        print(f"  Samples that hit the concentration cap, per species: {per_species}.")
    else:
        print("  No sample hit the concentration cap for any species.")


if __name__ == "__main__":
    main()
