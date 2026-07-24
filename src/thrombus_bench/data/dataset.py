"""PyTorch Dataset/DataLoader wrapping mechanistic simulation outputs.

Responsibility
---------------
Load the `.npz` sample records written by `generate_dataset.py` (already
rasterized onto a fixed-resolution grid there -- see its docstring) and
expose them as a `torch.utils.data.Dataset` yielding, per sample:

* `params`: the sampled parameter vector (geometry + physiology + inlet
  velocity), shape `(8,)`, order matching `generate_dataset.PARAM_ORDER`.
  Min-max normalized to `[-1, 1]` per `sampler.normalize_params` (based on
  `sampler.DEFAULT_RANGES`, the same physical ranges used for sampling) --
  raw values span wildly different scales (e.g. `platelet_conc_plt_ml`
  ~1e8 vs. `heparin_conc_uM` ~0.1-0.5), which a `nn.Linear` handles poorly
  un-normalized. Use `sampler.denormalize_params` to invert back to
  physical units (e.g. for plotting).
* `fields`: stacked target field channels (velocity x/y + 9 species
  concentrations), shape `(11, H, W)`, **log-compressed**: `field_to_log`
  applies `sign(x) * log1p(|x|)` per element. Raw field magnitudes span
  ~20 orders of magnitude across channels (platelet concentrations ~1e8
  PLT/mL vs. picomolar-scale agonists), which would otherwise make MSE
  training loss dominated entirely by the largest-magnitude channel(s) and
  give near-zero effective gradient to the rest. Use `log_to_field` to
  invert back to physical units (e.g. in `benchmark/metrics.py`).
* `max_M_at`, `thrombosed_fraction`: scalar summary targets used by
  `benchmark/metrics.py` (left in physical units, not log-compressed).
* `thrombin_fibrin_reliable`: bool scalar, False whenever
  `mechanistic/coupled_solver.py`'s [T]/[FI] concentration-cap safety clip
  actually bound during that sample's run (see
  `CoupledSimulationHistory`/`generate_dataset.py` docstrings) -- this
  sample's `conc_T`/`conc_FI` channels (and `FI`-derived quantities) should
  not be treated as physically meaningful without filtering/weighting by
  this flag downstream.
* `flow_n_iterations`, `flow_residual`: the final-checkpoint flow solve's
  Picard-iteration diagnostics (`mechanistic/flow_solver.FlowSolution`),
  alongside `converged`'s pass/fail summary of the same thing (not currently
  exposed here -- see `generate_dataset.py`'s saved `converged` key if
  needed).
* `clip_counts`, `conc_min`, `conc_max`: shape-`(9,)` QC arrays, one entry
  per species in the same order as `fields`'s species channels (`FIELD_
  NAMES[2:]`, i.e. RP/AP/APR/APS/T/AT/PT/FG/FI). `clip_counts` is each
  species' cumulative concentration-cap clip-event count over the whole run
  (`CoupledSimulationHistory.clip_event_counts`); `conc_min`/`conc_max` are
  each species' raw (pre-rasterization, pre-log-compression) nodal field
  extrema -- a cheap way to spot NaN/Inf/out-of-range values without
  decoding `fields`.
* `fluid_mask`: shape-`(H, W)` float32 0/1 raster, True where a grid cell
  center falls inside the actual FEM mesh domain (`generate_dataset.
  _fluid_mask`) -- the vessel+aneurysm domain is an L/T-shaped union, so
  `fields`'s bounding-box grid contains genuine exterior cells that
  `griddata(method="nearest")` silently filled with a nearby in-domain
  node's value. Kept out of `fields`/`FIELD_NAMES` deliberately: it's not a
  physical quantity to log-compress or predict, just context for
  interpreting the other channels (e.g. masking the loss/metrics to
  in-domain cells only).
* `M_at_wall`: shape-`(H, W)` float32 raster, a spatial representation of
  `surface_ode.SurfaceState.M_at` (PLT/cm^2) rasterized into a narrow band
  around the wall (`generate_dataset._rasterize_wall_band`; 0 elsewhere,
  including in-domain fluid cells away from the wall -- `M_at` is a
  *surface* density, not a bulk one). **Log-compressed** the same way as
  `fields` (`field_to_log`, for the same reason: `M_at` spans a huge range
  including exact zeros almost everywhere except the wall band) -- use
  `log_to_field` to invert. Kept as its own key rather than a 12th `fields`
  channel: different physical unit/support (surface density on a thin
  band vs. bulk concentration/velocity over the whole domain) and a
  different, much sparser statistic (mostly zeros) that would dilute a
  shared per-channel loss/metric if stacked in.

Train/val/test/edge-of-domain splits are read from the corresponding
subdirectory written by `generate_dataset.py`
(`sampler.split_train_val_test_edge_holdout`). An `"extrapolation"` split
also exists as an opt-in alternative, written by `generate_dataset.
generate_extrapolation_dataset` (`sampler.sample_with_extrapolation_holdout`)
-- unlike edge-of-domain, genuinely never-seen-during-training parameter
values for one chosen parameter; see `benchmark/extrapolation_eval.py`.

`PointCloudThrombusDataset` + `pointcloud_collate_fn`
----------------------------------------------------------
The point-cloud counterpart of `ThrombusSurrogateDataset`, reading the
node-native `.npz` schema `generate_dataset._build_pointcloud_sample`
writes by default since Phase 3 (`docs/continuous_surrogate_design.md`) --
see that function's docstring for the schema. Feeds `neural.
coordinate_decoder.ContinuousThrombusSurrogate`; `ThrombusSurrogateDataset`
above is unmodified and stays in use for the legacy grid-projection
comparison baseline (which needs samples generated with
`--also-save-raster`).
"""

from __future__ import annotations

import glob
import os
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .sampler import ParameterSpace, normalize_params

FIELD_NAMES = ("velocity_x", "velocity_y", "conc_RP", "conc_AP", "conc_APR", "conc_APS", "conc_T", "conc_AT", "conc_PT", "conc_FG", "conc_FI")
# Species name order for the `clip_counts`/`conc_min`/`conc_max` QC arrays,
# matching `FIELD_NAMES`'s species channels (derived rather than
# re-declared, so the two can't drift apart).
_SPECIES_NAMES = tuple(name[len("conc_") :] for name in FIELD_NAMES if name.startswith("conc_"))


def field_to_log(x):
    """sign(x) * log1p(|x|); see module docstring."""

    return np.sign(x) * np.log1p(np.abs(x)) if isinstance(x, np.ndarray) else torch.sign(x) * torch.log1p(torch.abs(x))


def log_to_field(x):
    """Inverse of `field_to_log`: sign(x) * expm1(|x|)."""

    return np.sign(x) * np.expm1(np.abs(x)) if isinstance(x, np.ndarray) else torch.sign(x) * torch.expm1(torch.abs(x))


class ThrombusSurrogateDataset(Dataset):
    def __init__(self, dataset_dir: str, split: Literal["train", "val", "test", "edge_holdout", "extrapolation"]):
        self.dataset_dir = dataset_dir
        self.split = split
        self.files = sorted(glob.glob(os.path.join(dataset_dir, split, "sample_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No samples found in {os.path.join(dataset_dir, split)}")
        self.param_space = ParameterSpace()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        fields = np.stack([data[name] for name in FIELD_NAMES], axis=0).astype(np.float32)
        fields = field_to_log(fields)
        params_normalized = normalize_params(data["params"], self.param_space).astype(np.float32)
        return {
            "params": torch.from_numpy(params_normalized),
            "fields": torch.from_numpy(fields),
            "max_M_at": torch.tensor(float(data["max_M_at"]), dtype=torch.float32),
            "thrombosed_fraction": torch.tensor(float(data["thrombosed_fraction"]), dtype=torch.float32),
            "thrombin_fibrin_reliable": torch.tensor(bool(data["thrombin_fibrin_reliable"]), dtype=torch.bool),
            "flow_n_iterations": torch.tensor(int(data["flow_n_iterations"]), dtype=torch.int64),
            "flow_residual": torch.tensor(float(data["flow_residual"]), dtype=torch.float32),
            "clip_counts": torch.tensor(
                [int(data[f"clip_count_{name}"]) for name in _SPECIES_NAMES], dtype=torch.int64
            ),
            "conc_min": torch.tensor(
                [float(data[f"conc_{name}_min"]) for name in _SPECIES_NAMES], dtype=torch.float32
            ),
            "conc_max": torch.tensor(
                [float(data[f"conc_{name}_max"]) for name in _SPECIES_NAMES], dtype=torch.float32
            ),
            "fluid_mask": torch.from_numpy(data["fluid_mask"].astype(np.float32)),
            "M_at_wall": torch.from_numpy(field_to_log(data["M_at_wall"].astype(np.float32))),
        }


class PointCloudThrombusDataset(Dataset):
    """One item per `(sample, checkpoint)` pair -- see module docstring.

    `__getitem__` returns, for a single checkpoint of a single sample:

    * `params_with_time`: `(9,)` float32 -- the existing 8 parameters,
      normalized exactly like `ThrombusSurrogateDataset` (`sampler.
      normalize_params`, reused directly, not reimplemented), plus a
      9th normalized-time scalar `t_norm = 2 * (t / t_final) - 1` where
      `t_final` is *this sample's own* last saved checkpoint time
      (`time_s[-1]`) -- the same min-max-to-`[-1, 1]` convention as the
      other 8, just using each sample's own run duration as its range
      rather than a shared physical scale (there is no single fixed
      "maximum simulated time" across samples the way there is a fixed
      physical range for e.g. `heparin_conc_uM`). Note `generate_dataset.
      _output_every_n_steps_for_snapshots` never records a true `t=0`
      checkpoint (see that function's docstring), so `t_norm` for a
      multi-checkpoint sample's first entry is strictly greater than -1,
      not exactly -1; for an `n_snapshots=1` sample it is always exactly
      `+1` (the one checkpoint *is* the final time).
    * `node_coords`: `(n_points, 2)` float32 -- that checkpoint's mesh
      node coordinates (meters), or a random subsample of size
      `points_per_sample` if that constructor argument is set and smaller
      than the checkpoint's actual node count.
    * `fields`: `(n_points, 11)` float32 -- field values at `node_coords`,
      in `FIELD_NAMES` order, subsampled identically to `node_coords`
      (same random indices). **Log-compressed** (`field_to_log`), same as
      `ThrombusSurrogateDataset.fields` and for the same reason (see that
      class's docstring) -- use `log_to_field` to invert.
    * `M_at_target`: `(n_points,)` float32 -- `surface_ode.
      SurfaceState.M_at` at each point, 0 for non-wall points (see
      `generate_dataset._build_pointcloud_sample`'s `wall_dofs` field and
      `docs/continuous_surrogate_design.md` Phase 4 "Finalized M_at design
      choice") -- the training target for `ContinuousThrombusSurrogate`'s
      optional 12th (`predict_M_at_wall`) output channel. Also
      log-compressed (`field_to_log(0) == 0`, so the off-wall fill value
      is unaffected).
    * `is_wall`: `(n_points,)` bool -- which points are actual wall nodes
      (kept alongside `M_at_target` for diagnostics; the training loss
      itself does not need it, since 0 is already the correct off-wall
      target -- mirrors `ThrombusSurrogateDataset`'s `M_at_wall` raster,
      which is 0-filled off the wall band with no separate mask either).
    * `geometry_mm`: `(2,)` float32 -- raw (unnormalized)
      `[aneurysm_diameter_mm, vessel_diameter_mm]` for this sample (i.e.
      `data["params"][:2]`, `PARAM_ORDER`'s first two entries) -- needed
      by `neural.coordinate_decoder.ContinuousThrombusSurrogate` for its
      analytic-SDF and coordinate-normalization terms, which require
      physical geometry rather than `params_with_time`'s normalized form.
    * `thrombin_fibrin_reliable`: bool scalar -- this checkpoint's entry
      from `thrombin_fibrin_reliable_at_checkpoint` (Phase 1), for the
      per-checkpoint channel-exclusion-by-default mechanism (`docs/
      continuous_surrogate_design.md`'s "Channel-exclusion caveat").

    Subsampling (`points_per_sample`) happens in `__getitem__`, using a
    freshly-constructed `np.random.default_rng()` (no fixed seed) on every
    call rather than an RNG stored on `self` -- this both re-draws a
    different random subsample each epoch (a `self`-stored RNG advanced
    once at construction would not) and is safe under a multi-worker
    `DataLoader` (`np.random.default_rng()` with no seed pulls fresh OS
    entropy per call, avoiding the classic bug where forked worker
    processes inherit and share identical global RNG state).
    """

    def __init__(
        self,
        dataset_dir: str,
        split: Literal["train", "val", "test", "edge_holdout", "extrapolation"],
        points_per_sample: int | None = None,
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.points_per_sample = points_per_sample
        self.files = sorted(glob.glob(os.path.join(dataset_dir, split, "sample_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No samples found in {os.path.join(dataset_dir, split)}")
        self.param_space = ParameterSpace()

        # (file_idx, checkpoint_idx) for every checkpoint of every sample --
        # only `time_s` (tiny) is read per file here, not `node_coords`/
        # `fields`, to keep dataset construction cheap.
        self._index: list[tuple[int, int]] = []
        for file_idx, path in enumerate(self.files):
            with np.load(path) as data:
                n_snapshots = data["time_s"].shape[0]
            self._index.extend((file_idx, checkpoint_idx) for checkpoint_idx in range(n_snapshots))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, checkpoint_idx = self._index[idx]
        data = np.load(self.files[file_idx])

        params_raw = data["params"]
        params_normalized = normalize_params(params_raw, self.param_space).astype(np.float32)
        time_s = data["time_s"]
        t_norm = np.float32(2.0 * (time_s[checkpoint_idx] / time_s[-1]) - 1.0)
        params_with_time = np.concatenate([params_normalized, [t_norm]]).astype(np.float32)

        node_coords = data["node_coords"].astype(np.float32)
        # Log-compressed (field_to_log: sign(x) * log1p(|x|)), same as
        # ThrombusSurrogateDataset's `fields` -- raw field magnitudes span
        # ~20 orders of magnitude across channels (see that class's
        # docstring), which otherwise makes a plain per-point MSE loss
        # dominated entirely by the largest-magnitude channel(s) (e.g.
        # platelet concentrations ~1e8) with near-zero effective gradient
        # for the rest; confirmed empirically running train_continuous
        # without this (loss was completely flat across 10 epochs).
        fields = field_to_log(data["fields"][checkpoint_idx].astype(np.float32))

        # M_at (surface_ode.SurfaceState.M_at) as a 12th per-point target,
        # 0-filled off the wall -- see docs/continuous_surrogate_design.md
        # Phase 4 "Finalized M_at design choice": wall_dofs are indices
        # into these same bulk node_coords/fields (not a separate
        # geometric point set), so this needs no extra query points, just
        # a target array the training loop concatenates as channel 12
        # when `model.predict_M_at_wall` is set (mirroring
        # ThrombusSurrogate's existing plain-MSE, 0-filled-off-wall
        # convention for the same quantity on the raster path). Also
        # log-compressed (field_to_log(0) == 0, so the off-wall fill value
        # is unaffected).
        n_nodes = node_coords.shape[0]
        wall_dofs = data["wall_dofs"]
        m_at_target = np.zeros(n_nodes, dtype=np.float32)
        m_at_target[wall_dofs] = data["M_at_wall_values"][checkpoint_idx]
        m_at_target = field_to_log(m_at_target)
        is_wall = np.zeros(n_nodes, dtype=bool)
        is_wall[wall_dofs] = True

        if self.points_per_sample is not None and self.points_per_sample < n_nodes:
            rng = np.random.default_rng()
            chosen = rng.choice(n_nodes, size=self.points_per_sample, replace=False)
            node_coords = node_coords[chosen]
            fields = fields[chosen]
            m_at_target = m_at_target[chosen]
            is_wall = is_wall[chosen]

        geometry_mm = params_raw[:2].astype(np.float32)
        reliable = bool(data["thrombin_fibrin_reliable_at_checkpoint"][checkpoint_idx])

        return {
            "params_with_time": torch.from_numpy(params_with_time),
            "node_coords": torch.from_numpy(node_coords),
            "fields": torch.from_numpy(fields),
            "M_at_target": torch.from_numpy(m_at_target),
            "is_wall": torch.from_numpy(is_wall),
            "geometry_mm": torch.from_numpy(geometry_mm),
            "thrombin_fibrin_reliable": torch.tensor(reliable, dtype=torch.bool),
        }


def pointcloud_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Custom `collate_fn` for `PointCloudThrombusDataset`, implementing
    the flat-points + `batch_index` ragged-batching convention from
    `neural.coordinate_decoder.py` (see that module's docstring) --
    PyTorch's default collation requires every sample's tensors to already
    share the same shape, which different samples' (generally different)
    node counts violate.

    Returns `params_with_time`/`geometry_mm`/`thrombin_fibrin_reliable`
    stacked to `(batch, ...)` as usual, and `node_coords`/`fields`/
    `M_at_target`/`is_wall` concatenated to flat `(total_points, ...)`
    tensors plus a `batch_index: (total_points,)` long tensor mapping each
    point back to its sample's position in the batch -- directly
    consumable as `ContinuousThrombusSurrogate.forward`'s
    `query_points_m`/`batch_index` (with `fields`/`M_at_target` as the
    training targets at those same points).
    """

    params_with_time = torch.stack([item["params_with_time"] for item in batch], dim=0)
    geometry_mm = torch.stack([item["geometry_mm"] for item in batch], dim=0)
    thrombin_fibrin_reliable = torch.stack([item["thrombin_fibrin_reliable"] for item in batch], dim=0)

    node_coords = torch.cat([item["node_coords"] for item in batch], dim=0)
    fields = torch.cat([item["fields"] for item in batch], dim=0)
    m_at_target = torch.cat([item["M_at_target"] for item in batch], dim=0)
    is_wall = torch.cat([item["is_wall"] for item in batch], dim=0)
    batch_index = torch.cat(
        [torch.full((item["node_coords"].shape[0],), i, dtype=torch.long) for i, item in enumerate(batch)]
    )

    return {
        "params_with_time": params_with_time,
        "geometry_mm": geometry_mm,
        "thrombin_fibrin_reliable": thrombin_fibrin_reliable,
        "node_coords": node_coords,
        "fields": fields,
        "M_at_target": m_at_target,
        "is_wall": is_wall,
        "batch_index": batch_index,
    }
