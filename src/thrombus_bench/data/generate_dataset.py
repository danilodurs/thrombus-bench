"""Batch-run the mechanistic solver over sampled parameters to build the surrogate dataset.

Responsibility
---------------
For each parameter sample from `sampler.py`:

1. Build the mesh (`mechanistic/mesh.py`) for the sampled geometry.
2. Run the full coupled mechanistic simulation (`mechanistic/coupled_solver.py`)
   for the sampled physiological parameters and inlet velocity, checkpointed
   at `n_snapshots` points in time (see `_output_every_n_steps_for_snapshots`;
   `n_snapshots=1` reproduces the original final-checkpoint-only behavior).
3. Save each checkpoint's mesh node coordinates and per-node field values
   directly -- **no rasterization** -- as the point-cloud schema
   (`_build_pointcloud_sample`, finalized in `docs/
   continuous_surrogate_design.md` Phase 1): the FEM mesh's own node
   coordinates are the ground truth with zero interpolation error, unlike
   the legacy `_rasterize`/`griddata(method="nearest")` path (see below),
   and different sampled geometries mesh differently anyway, so a fixed
   grid was always an approximation. Also includes per-sample QC scalars
   (`max_M_at`, `thrombosed_fraction`, `converged`, `flow_n_iterations`/
   `flow_residual`, `thrombin_fibrin_reliable` -- see
   `mechanistic/coupled_solver.CoupledSimulationHistory` docstring --
   `clip_count_{species}`, `conc_{species}_min`/`max`) and the new
   per-checkpoint `thrombin_fibrin_reliable_at_checkpoint` array -- see
   `_build_pointcloud_sample`'s docstring for the full schema. This is
   this project's only per-sample QC signal beyond visual inspection; none
   of it is consumed by training/evaluation automatically -- see
   `data/dataset.py`'s docstring for how to read it back.

   `--also-save-raster` (default off) additionally computes and merges in
   the legacy rasterized representation (`_build_raster_sample`: fixed-grid
   `FIELD_NAMES` channels via `_rasterize`, `fluid_mask` via `_fluid_mask`,
   and the wall-band `M_at_wall` raster via `_rasterize_wall_band`) for the
   *final* checkpoint only -- needed to produce data
   `data/dataset.ThrombusSurrogateDataset`/the legacy grid-projection
   comparison baseline (`neural/model.ThrombusSurrogate`) can consume, or
   for development/debugging visualization. Off by default specifically
   because it is the expensive part -- `griddata`/mask-building cost is
   skipped entirely when not requested (see this module's test suite for a
   measured timing comparison).
4. Route each sample to train/val/test/edge_holdout per
   `sampler.split_train_val_test_edge_holdout`.

Scope note: runs serially (no multiprocessing) -- scikit-fem `Basis`/`Mesh`
objects are not trivially picklable across a `multiprocessing.Pool`, and
given this project's reduced-scale demo dataset (see README.md "Project
status"), serial execution is fast enough in practice.

Opt-in genuine-extrapolation variant
--------------------------------------
`generate_extrapolation_dataset` (CLI: `--extrapolation-param`) is a
separate, opt-in mode -- not part of the default train/val/test/edge_holdout
dataset above -- that restricts one parameter's train/val/test range to a
sub-interval and draws a fourth "extrapolation" split from the withheld
remainder, so a model trained on it has genuinely never seen that
parameter's extrapolation range (unlike the edge-of-domain holdout, which
is still drawn from the same sampled box). See `sampler.
sample_with_extrapolation_holdout`'s docstring.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import yaml
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from thrombus_bench.mechanistic.coupled_solver import CoupledSimulationHistory, run_coupled_simulation
from thrombus_bench.mechanistic.mesh import GeometryConfig, MeshConfig, TaggedMesh, build_aneurysm_mesh

from .dataset import FIELD_NAMES
from .sampler import (
    DEFAULT_RANGES,
    ParameterSpace,
    latin_hypercube_sample,
    sample_with_extrapolation_holdout,
    split_train_val_test_edge_holdout,
)

_ALL_SPECIES = ("RP", "AP", "APR", "APS", "T", "AT", "PT", "FG", "FI")
PARAM_ORDER = (
    "aneurysm_diameter_mm", "vessel_diameter_mm", "inlet_velocity_cm_s", "platelet_conc_plt_ml",
    "heparin_conc_uM", "prothrombin_uM", "antithrombin_uM", "fibrinogen_uM",
)

# Default checkpoint count for the point-cloud storage path (see
# `_output_every_n_steps_for_snapshots`'s docstring and `_run_one_sample`)
# -- `docs/continuous_surrogate_design.md` Phase 1/3.
DEFAULT_N_SNAPSHOTS = 5


def _output_every_n_steps_for_snapshots(n_steps: int, n_snapshots: int) -> int:
    """Derive the `output_every_n_steps` stride (`coupled_solver.
    run_coupled_simulation`'s checkpoint cadence) that yields approximately
    `n_snapshots` checkpoints, roughly evenly spaced, across a run of
    `n_steps` macro time steps.

    `n_snapshots <= 1` reproduces the original `_run_one_sample` formula
    (`max(1, n_steps)` -- final-checkpoint-only) exactly; this is a
    regression guard, not just a convenient default, since every dataset
    generated before this function existed relied on exactly that formula
    for its one saved checkpoint.

    For `n_snapshots > 1`: `run_coupled_simulation`'s own recording
    condition (`(step + 1) % output_every_n_steps == 0 or step == n_steps
    - 1`) already guarantees the *final* step is always captured
    regardless of stride, so this only needs to control the spacing of the
    remaining `n_snapshots - 1` checkpoints. Note this does **not** produce a true
    `t=0` checkpoint -- `run_coupled_simulation` only ever records state
    *after* a macro step's update, so the earliest possible checkpoint is
    the state after step 0 (t=dt_s), not the initial condition; capturing
    a genuine t=0 snapshot would need an explicit pre-loop-state recording
    added to `run_coupled_simulation` itself, which nothing in this project
    currently needs (`data.dataset.PointCloudThrombusDataset`'s time
    normalization accounts for this -- see that class's docstring).
    """

    if n_snapshots <= 1:
        return max(1, n_steps)
    return max(1, int(round(n_steps / (n_snapshots - 1))))


def _rasterize(node_coords: np.ndarray, values: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor rasterization of a nodal field onto a regular grid
    covering the node coordinates' bounding box."""

    xmin, ymin = node_coords.min(axis=1)
    xmax, ymax = node_coords.max(axis=1)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, grid_size[1]), np.linspace(ymin, ymax, grid_size[0]))
    grid = griddata(node_coords.T, values, (gx, gy), method="nearest")
    return grid


def _fluid_mask(node_coords: np.ndarray, triangles: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    """Boolean-as-float32 raster mask, True (1.0) where a grid cell center
    falls inside the actual FEM mesh domain, False (0.0) otherwise.

    The vessel+aneurysm domain is an L/T-shaped union (a long thin vessel
    rectangle with a circular sac bulging from the top) -- `_rasterize`'s
    regular grid spans the mesh nodes' axis-aligned bounding box, which
    contains genuine exterior cells (e.g. above the vessel walls, outside
    the sac footprint). `griddata(method="nearest")` silently fills those
    exterior cells with whatever in-domain node happens to be nearest, which
    is not a physically meaningful value; this mask flags which raster
    cells are trustworthy.

    Built from the mesh's own triangulation (`matplotlib.tri.Triangulation`
    + `get_trifinder`, a point-in-triangle test) rather than reconstructing
    the analytic domain outline, so it stays correct for any mesh this
    project can produce (the fill/thinning in `mesh.py` means the actual
    triangulated domain isn't perfectly identical to the analytic polygon).
    Uses the same grid convention as `_rasterize` (bounding box of
    `node_coords`, `grid_size = (n_rows, n_cols)`).
    """

    xmin, ymin = node_coords.min(axis=1)
    xmax, ymax = node_coords.max(axis=1)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, grid_size[1]), np.linspace(ymin, ymax, grid_size[0]))

    triangulation = Triangulation(node_coords[0], node_coords[1], triangles.T)
    membership = triangulation.get_trifinder()(gx.ravel(), gy.ravel())
    return (membership >= 0).astype(np.float32).reshape(grid_size)


def _rasterize_wall_band(
    node_coords: np.ndarray,
    wall_node_coords: np.ndarray,
    wall_values: np.ndarray,
    grid_size: tuple[int, int],
    band_width_cells: float = 1.5,
) -> np.ndarray:
    """Rasterize a wall-only nodal field (e.g. `surface_ode.SurfaceState.M_at`,
    defined only on wall DOFs -- unlike the bulk species fields `_rasterize`
    handles) into a narrow band around the wall, on the same fixed
    bounding-box grid as `_rasterize`/`_fluid_mask`.

    Deliberately NOT a nearest-neighbor fill of the whole domain (that
    would smear a *surface* density across the entire fluid interior,
    which `_rasterize` is fine with for bulk concentrations but would be
    physically wrong here). Instead: for every grid cell, find the nearest
    wall node (`scipy.spatial.cKDTree`) and assign its value only if within
    `band_width_cells` grid cells of the wall; all other cells are 0.

    The threshold is expressed in grid cells (`band_width_cells *
    min(dx, dy)`, using the *finer* of the grid's two axis spacings) rather
    than a fixed physical length, so the band stays a similarly-narrow
    number of cells thick regardless of `grid_size` or how elongated the
    domain's bounding box is (this domain is a long, thin vessel, so
    `dx` and `dy` can differ substantially -- using the coarser axis would
    make the band unnecessarily thick along the finer one).
    """

    xmin, ymin = node_coords.min(axis=1)
    xmax, ymax = node_coords.max(axis=1)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, grid_size[1]), np.linspace(ymin, ymax, grid_size[0]))
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])

    dx = (xmax - xmin) / max(grid_size[1] - 1, 1)
    dy = (ymax - ymin) / max(grid_size[0] - 1, 1)
    threshold = band_width_cells * min(dx, dy)

    tree = cKDTree(wall_node_coords.T)
    distances, nearest_idx = tree.query(grid_points)

    raster = np.where(distances <= threshold, wall_values[nearest_idx], 0.0)
    return raster.reshape(grid_size).astype(np.float32)


def _build_pointcloud_sample(
    tagged_mesh: TaggedMesh,
    history: CoupledSimulationHistory,
    params: np.ndarray,
    m_at_critical_plt_cm2: float,
) -> dict:
    """Point-cloud (ragged, node-native) sample builder -- the default
    `_run_one_sample` save path since Phase 3 (`docs/
    continuous_surrogate_design.md`); `data.dataset.PointCloudThrombusDataset`
    is the corresponding `Dataset`.

    Motivation: `_rasterize`'s `griddata(method="nearest")` step introduces
    interpolation error and wastes the FEM mesh's own resolution (which
    varies per sample -- different sampled geometries mesh differently).
    Training the continuous decoder (Stage 2) directly against the mesh's
    own node coordinates uses the ground-truth solution with zero
    interpolation error, and needs no fixed grid at all.

    Schema (one `.npz` per sample, mirroring `_run_one_sample`'s dict):

    - ``params``: ``(8,)`` float64 -- unchanged, `PARAM_ORDER`.
    - ``node_coords``: ``(n_nodes, 2)`` float64 -- FEM mesh vertex
      coordinates (meters), `tagged_mesh.mesh.p.T`.
    - ``triangles``: ``(n_triangles, 3)`` int32 -- mesh connectivity
      (`tagged_mesh.mesh.t.T`), kept only so a post-hoc legacy
      visualization utility can still rasterize this sample the old way
      (e.g. via `matplotlib.tri.Triangulation`) without needing the
      original mesh object -- not used in training.
    - ``fields``: ``(n_snapshots, n_nodes, 11)`` float32 -- physical field
      values at every node, at every checkpoint, in `dataset.FIELD_NAMES`
      channel order (velocity_x, velocity_y, then conc_{species} in
      `_ALL_SPECIES` order -- the two orderings already agree today).
    - ``wall_dofs``: ``(n_wall_nodes,)`` int64 -- indices into
      ``node_coords``/``fields``'s first axis identifying which bulk mesh
      nodes are wall nodes (fixed across checkpoints within a sample,
      since the mesh/wall_dofs don't change during a run); i.e.
      ``node_coords[wall_dofs] == wall_node_coords`` exactly.
      `data.dataset.PointCloudThrombusDataset` uses this to build a
      per-point M_at training target (Phase 4) without a separate
      geometric point set.
    - ``wall_node_coords``: ``(n_wall_nodes, 2)`` float64 -- wall DOF
      coordinates (fixed across checkpoints within a sample, since the
      mesh/wall_dofs don't change during a run).
    - ``M_at_wall_values``: ``(n_snapshots, n_wall_nodes)`` float32 --
      `surface_ode.SurfaceState.M_at` at each wall node, per checkpoint --
      the exact surface-density values `_rasterize_wall_band` currently
      only approximates via a nearest-neighbor grid band.
    - ``time_s``: ``(n_snapshots,)`` float64 -- checkpoint times.
    - ``thrombin_fibrin_reliable_at_checkpoint``: ``(n_snapshots,)`` bool
      -- from `CoupledSimulationHistory` (see that field's docstring);
      this is genuinely new per-checkpoint information the raster schema
      never had.
    - Per-run QC keys, unchanged in meaning from `_run_one_sample`:
      ``converged``, ``flow_n_iterations``, ``flow_residual`` (from the
      *final* checkpoint's flow, matching today's convention),
      ``thrombin_fibrin_reliable``, ``clip_count_{species}``,
      ``conc_{species}_min``/``max`` (final-checkpoint nodal extrema, same
      convention as today), ``max_M_at``, ``thrombosed_fraction`` (the
      latter needs `physio["adhesion_aggregation"]["M_at_critical_plt_cm2"]`,
      passed in explicitly as `m_at_critical_plt_cm2` since this function
      takes no `physio` dict).

    Deliberately dropped from this schema: ``fluid_mask`` (only meaningful
    relative to a fixed raster grid; point-cloud training has no exterior
    cells to flag) and the rasterized ``M_at_wall``/bulk-field grids
    themselves (superseded by the node-native arrays above; the legacy
    raster path can still be reconstructed post-hoc from ``node_coords``
    + ``triangles`` + ``fields`` when needed for visualization).

    Ragged-ness (deliberately not handled here): ``n_nodes`` and
    ``n_wall_nodes`` vary per sample -- different sampled geometries mesh
    differently (`mesh.py::build_aneurysm_mesh`). A single sample's `.npz`
    has no problem with that (each file just has its own shapes); it only
    becomes a real problem when a `DataLoader` tries to batch multiple
    samples' variable-length arrays together, which is Phase 3's concern
    (e.g. padding + a mask, or a custom `collate_fn`), not this function's.
    """

    states = history.states
    n_snapshots = len(states)
    node_coords = tagged_mesh.mesh.p.T
    n_vertices = node_coords.shape[0]
    wall_dofs = states[0].wall_dofs
    wall_node_coords = history.basis_c.doflocs[:, wall_dofs].T

    fields = np.empty((n_snapshots, n_vertices, len(FIELD_NAMES)), dtype=np.float32)
    m_at_wall_values = np.empty((n_snapshots, len(wall_dofs)), dtype=np.float32)
    time_s = np.empty(n_snapshots, dtype=np.float64)
    for i, state in enumerate(states):
        ux = state.flow.u[0 : 2 * n_vertices : 2]
        uy = state.flow.u[1 : 2 * n_vertices : 2]
        channels = [ux, uy] + [state.concentrations[name][:n_vertices] for name in _ALL_SPECIES]
        fields[i] = np.stack(channels, axis=-1)
        m_at_wall_values[i] = state.surface.M_at
        time_s[i] = state.time_s

    final = states[-1]
    conc_min_max = {}
    for name in _ALL_SPECIES:
        nodal = final.concentrations[name][:n_vertices]
        conc_min_max[f"conc_{name}_min"] = float(np.min(nodal))
        conc_min_max[f"conc_{name}_max"] = float(np.max(nodal))
    clip_counts = {f"clip_count_{name}": int(history.clip_event_counts[name]) for name in _ALL_SPECIES}

    return {
        "params": np.asarray(params, dtype=np.float64),
        "node_coords": node_coords.astype(np.float64),
        "triangles": tagged_mesh.mesh.t.T.astype(np.int32),
        "fields": fields,
        "wall_dofs": np.asarray(wall_dofs, dtype=np.int64),
        "wall_node_coords": wall_node_coords.astype(np.float64),
        "M_at_wall_values": m_at_wall_values,
        "time_s": time_s,
        "thrombin_fibrin_reliable_at_checkpoint": history.thrombin_fibrin_reliable_at_checkpoint,
        "converged": bool(final.flow.converged),
        "flow_n_iterations": int(final.flow.n_iterations),
        "flow_residual": float(final.flow.residual),
        "thrombin_fibrin_reliable": bool(history.thrombin_fibrin_reliable),
        "max_M_at": float(final.surface.M_at.max()),
        "thrombosed_fraction": float(np.mean(final.surface.M_at >= m_at_critical_plt_cm2)),
        **clip_counts,
        **conc_min_max,
    }


def _build_raster_sample(
    tagged_mesh: TaggedMesh, history: CoupledSimulationHistory, grid_size: tuple[int, int]
) -> dict:
    """Legacy rasterized representation of a run's *final* checkpoint only
    (`_rasterize`/`_fluid_mask`/`_rasterize_wall_band`) -- opt-in via
    `_run_one_sample`'s `also_save_raster` (see that function and this
    module's docstring), kept as a comparison/visualization utility
    (`ThrombusSurrogateDataset`/`neural.model.ThrombusSurrogate`'s
    grid-projection baseline, `docs/continuous_surrogate_design.md`), not
    part of the default point-cloud training-data path.

    Returns only the raster-specific keys (`fluid_mask`, `M_at_wall`, and
    `fields`'s `FIELD_NAMES` grid keys) -- `_run_one_sample` merges this
    into `_build_pointcloud_sample`'s result, whose QC scalars
    (`converged`, `clip_count_*`, etc., also derived from the final
    checkpoint) are already identical and don't need recomputing here.
    """

    final = history.states[-1]
    node_coords = tagged_mesh.mesh.p
    n_vertices = node_coords.shape[1]
    ux = final.flow.u[0 : 2 * n_vertices : 2]
    uy = final.flow.u[1 : 2 * n_vertices : 2]

    fields = {
        "velocity_x": _rasterize(node_coords, ux, grid_size),
        "velocity_y": _rasterize(node_coords, uy, grid_size),
    }
    fluid_mask = _fluid_mask(node_coords, tagged_mesh.mesh.t, grid_size)
    wall_node_coords = history.basis_c.doflocs[:, final.wall_dofs]
    M_at_wall = _rasterize_wall_band(node_coords, wall_node_coords, final.surface.M_at, grid_size)
    for name in _ALL_SPECIES:
        nodal = final.concentrations[name][:n_vertices]
        fields[f"conc_{name}"] = _rasterize(node_coords, nodal, grid_size)

    return {"fluid_mask": fluid_mask, "M_at_wall": M_at_wall, **fields}


def _run_one_sample(
    sample: dict,
    physio_base: dict,
    mesh_cfg: dict,
    end_time_s: float,
    dt_s: float,
    grid_size: tuple[int, int],
    n_snapshots: int = DEFAULT_N_SNAPSHOTS,
    also_save_raster: bool = False,
) -> dict:
    """Build one sample's mechanistic run and dataset record.

    Default (point-cloud) path: `n_snapshots` checkpoints (see
    `_output_every_n_steps_for_snapshots`; `n_snapshots<=1` reproduces
    today's original final-checkpoint-only formula exactly, so existing
    single-checkpoint behavior needs no special-casing) saved via
    `_build_pointcloud_sample` -- no `griddata`/mask-building cost. Passing
    `also_save_raster=True` additionally computes and merges in the legacy
    rasterized representation (`_build_raster_sample`) for the *final*
    checkpoint, e.g. for development/debugging or for still generating
    data the legacy grid-projection baseline (`ThrombusSurrogateDataset`)
    can consume; this is opt-in specifically because it is the expensive
    part (see this module's docstring for measured timing).

    `flow_resolve_every_n_steps` is set equal to the derived
    `output_every_n_steps` (not a separate, coarser cadence) so that each
    saved checkpoint's flow field is actually resolved at roughly that
    checkpoint's own time, rather than every checkpoint before the last
    reusing a stale, much-earlier flow solve -- for `n_snapshots<=1` this
    is exactly today's single value, preserving the regression guarantee.
    """

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
    n_steps = max(1, int(round(end_time_s / dt_s)))
    output_every_n_steps = _output_every_n_steps_for_snapshots(n_steps, n_snapshots)
    history = run_coupled_simulation(
        tagged_mesh, inlet_velocity_m_s=inlet_velocity_m_s, physio=physio,
        end_time_s=end_time_s, dt_s=dt_s, output_every_n_steps=output_every_n_steps,
        flow_resolve_every_n_steps=output_every_n_steps,
    )

    params = np.array([sample[name] for name in PARAM_ORDER], dtype=np.float64)
    m_at_critical = physio["adhesion_aggregation"]["M_at_critical_plt_cm2"]
    result = _build_pointcloud_sample(tagged_mesh, history, params, m_at_critical)

    if also_save_raster:
        result.update(_build_raster_sample(tagged_mesh, history, grid_size))

    return result


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


def _generate_from_splits(
    splits: dict[str, list[dict]],
    physio_base: dict,
    geometry_mesh_cfg: dict,
    output_dir: str,
    end_time_s: float,
    dt_s: float,
    grid_size: tuple[int, int],
    n_snapshots: int = DEFAULT_N_SNAPSHOTS,
    also_save_raster: bool = False,
) -> tuple[dict[str, int], dict]:
    """Shared by `generate_dataset` and `generate_extrapolation_dataset`:
    batch-run the mechanistic solver over every sample in `splits` (a
    `{split_name: [sample, ...]}` dict, from either
    `sampler.split_train_val_test_edge_holdout` or
    `sampler.sample_with_extrapolation_holdout`), writing results under
    `output_dir/{split_name}/`. Returns `(counts, qc_summary)` -- see
    `generate_dataset`'s docstring."""

    counts = {}
    qc_summary = _new_qc_summary()
    for split_name, split_samples in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for i, sample in enumerate(split_samples):
            result = _run_one_sample(
                sample, physio_base, geometry_mesh_cfg, end_time_s, dt_s, grid_size,
                n_snapshots=n_snapshots, also_save_raster=also_save_raster,
            )
            np.savez(os.path.join(split_dir, f"sample_{i:04d}.npz"), **result)
            _update_qc_summary(qc_summary, result)
        counts[split_name] = len(split_samples)
    return counts, qc_summary


def generate_dataset(
    config: dict,
    physio_base: dict,
    geometry_mesh_cfg: dict,
    output_dir: str,
    end_time_s: float = 1.0,
    dt_s: float = 0.1,
    grid_size: tuple[int, int] = (32, 32),
    n_snapshots: int = DEFAULT_N_SNAPSHOTS,
    also_save_raster: bool = False,
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
    return _generate_from_splits(
        splits, physio_base, geometry_mesh_cfg, output_dir, end_time_s, dt_s, grid_size,
        n_snapshots=n_snapshots, also_save_raster=also_save_raster,
    )


# Split point for generate_extrapolation_dataset's default --extrapolation-param
# choices: fraction of the parameter's full sampler.DEFAULT_RANGES span
# routed to train/val/test (the remainder, upper end, is the withheld
# "extrapolation" range) -- see generate_extrapolation_dataset's docstring.
DEFAULT_EXTRAPOLATION_TRAIN_FRACTION = 0.7


def generate_extrapolation_dataset(
    config: dict,
    physio_base: dict,
    geometry_mesh_cfg: dict,
    output_dir: str,
    extrapolate_param: str,
    train_range: tuple[float, float],
    extrapolate_range: tuple[float, float],
    n_extrapolate: int,
    end_time_s: float = 1.0,
    dt_s: float = 0.1,
    grid_size: tuple[int, int] = (32, 32),
    n_snapshots: int = DEFAULT_N_SNAPSHOTS,
    also_save_raster: bool = False,
) -> tuple[dict[str, int], dict]:
    """Opt-in, genuinely-extrapolative counterpart to `generate_dataset`
    (see `sampler.sample_with_extrapolation_holdout`'s docstring for how
    this differs from the default edge-of-domain holdout): train/val/test
    are drawn with `extrapolate_param` restricted to `train_range` (all
    other parameters from their full `sampler.DEFAULT_RANGES`, as usual);
    a fourth "extrapolation" split is drawn with `extrapolate_param`
    restricted to the withheld `extrapolate_range` instead. Writes results
    under `output_dir/{train,val,test,extrapolation}/`; same return value
    and per-sample QC fields as `generate_dataset`.

    `config` supplies `n_train`/`n_val`/`n_test` (same keys as
    `generate_dataset`'s `config` -- `n_edge_holdout`/`edge_holdout_quantile`
    are not used here)."""

    space = ParameterSpace()
    splits = sample_with_extrapolation_holdout(
        space, extrapolate_param, train_range, extrapolate_range,
        n_train=config["n_train"], n_val=config["n_val"], n_test=config["n_test"],
        n_extrapolate=n_extrapolate, seed=config.get("seed", 0),
    )
    return _generate_from_splits(
        splits, physio_base, geometry_mesh_cfg, output_dir, end_time_s, dt_s, grid_size,
        n_snapshots=n_snapshots, also_save_raster=also_save_raster,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=str, default="configs/demo_cpu.yaml")
    parser.add_argument("--physio-config", type=str, default="configs/physio_params.yaml")
    parser.add_argument("--geometry-config", type=str, default="configs/geometry.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Default: data/processed, or data/processed_extrap_{param} if --extrapolation-param is given.")
    parser.add_argument("--end-time-s", type=float, default=1.0)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--target-num-elements", type=int, default=800)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument(
        "--extrapolation-param", type=str, default=None, choices=["heparin_conc_uM"],
        help="Opt-in: generate a genuinely-extrapolative split for this parameter instead of the "
        "default edge-of-domain-holdout dataset -- see generate_extrapolation_dataset's docstring.",
    )
    parser.add_argument(
        "--extrapolation-split-fraction", type=float, default=DEFAULT_EXTRAPOLATION_TRAIN_FRACTION,
        help="Only used with --extrapolation-param: fraction of that parameter's full "
        "sampler.DEFAULT_RANGES span routed to train/val/test (lower end); the remainder "
        "(upper end) is the withheld extrapolation range.",
    )
    parser.add_argument(
        "--n-extrapolate", type=int, default=6,
        help="Only used with --extrapolation-param: number of samples in the extrapolation split.",
    )
    parser.add_argument(
        "--n-snapshots", type=int, default=None,
        help="Number of checkpoints per sample to save (point-cloud path), via "
        "`_output_every_n_steps_for_snapshots`; 1 reproduces the original final-checkpoint-only "
        "behavior. Defaults to --training-config's data.n_snapshots if set there (e.g. "
        "configs/continuous.yaml), else DEFAULT_N_SNAPSHOTS. See that function's docstring.",
    )
    parser.add_argument(
        "--also-save-raster", action="store_true",
        help="Additionally compute and save the legacy rasterized representation "
        "(_build_raster_sample: fluid_mask/M_at_wall/FIELD_NAMES grids for the final checkpoint) "
        "alongside the default point-cloud data -- needed to generate data "
        "ThrombusSurrogateDataset/the grid-projection baseline can consume. Off by default since "
        "it is the expensive part (griddata/mask-building); see this module's docstring.",
    )
    args = parser.parse_args()

    with open(args.training_config) as f:
        training_cfg = yaml.safe_load(f)
    with open(args.physio_config) as f:
        physio_base = yaml.safe_load(f)
    with open(args.geometry_config) as f:
        geometry_yaml = yaml.safe_load(f)
    mesh_cfg = dict(geometry_yaml["mesh"])
    mesh_cfg["target_num_elements"] = args.target_num_elements
    n_snapshots = args.n_snapshots if args.n_snapshots is not None else training_cfg["data"].get(
        "n_snapshots", DEFAULT_N_SNAPSHOTS
    )

    if args.extrapolation_param:
        output_dir = args.output_dir or f"data/processed_extrap_{args.extrapolation_param.removesuffix('_uM').removesuffix('_conc')}"
        lo, hi = DEFAULT_RANGES[args.extrapolation_param]
        split_point = lo + args.extrapolation_split_fraction * (hi - lo)
        train_range, extrapolate_range = (lo, split_point), (split_point, hi)
        counts, qc = generate_extrapolation_dataset(
            training_cfg["data"], physio_base, mesh_cfg, output_dir,
            extrapolate_param=args.extrapolation_param, train_range=train_range, extrapolate_range=extrapolate_range,
            n_extrapolate=args.n_extrapolate,
            end_time_s=args.end_time_s, dt_s=args.dt_s, grid_size=(args.grid_size, args.grid_size),
            n_snapshots=n_snapshots, also_save_raster=args.also_save_raster,
        )
        print(
            f"Wrote extrapolation dataset ({args.extrapolation_param}: train/val/test in "
            f"{train_range}, extrapolation in {extrapolate_range}) to {output_dir}: {counts}"
        )
    else:
        output_dir = args.output_dir or "data/processed"
        counts, qc = generate_dataset(
            training_cfg["data"], physio_base, mesh_cfg, output_dir,
            end_time_s=args.end_time_s, dt_s=args.dt_s, grid_size=(args.grid_size, args.grid_size),
            n_snapshots=n_snapshots, also_save_raster=args.also_save_raster,
        )
        print(f"Wrote dataset to {output_dir}: {counts}")
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
