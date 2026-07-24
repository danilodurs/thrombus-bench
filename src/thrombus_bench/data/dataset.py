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
