"""PyTorch Dataset/DataLoader wrapping mechanistic simulation outputs.

Responsibility
---------------
Load the `.npz` sample records written by `generate_dataset.py` (already
rasterized onto a fixed-resolution grid there -- see its docstring) and
expose them as a `torch.utils.data.Dataset` yielding, per sample:

* `params`: the sampled parameter vector (geometry + physiology + inlet
  velocity), shape `(8,)`, order matching `generate_dataset.PARAM_ORDER`.
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

Train/val/test/OOD splits are read from the corresponding subdirectory
written by `generate_dataset.py` (`sampler.split_train_val_test_ood`).
"""

from __future__ import annotations

import glob
import os
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

FIELD_NAMES = ("velocity_x", "velocity_y", "conc_RP", "conc_AP", "conc_APR", "conc_APS", "conc_T", "conc_AT", "conc_PT", "conc_FG", "conc_FI")


def field_to_log(x):
    """sign(x) * log1p(|x|); see module docstring."""

    return np.sign(x) * np.log1p(np.abs(x)) if isinstance(x, np.ndarray) else torch.sign(x) * torch.log1p(torch.abs(x))


def log_to_field(x):
    """Inverse of `field_to_log`: sign(x) * expm1(|x|)."""

    return np.sign(x) * np.expm1(np.abs(x)) if isinstance(x, np.ndarray) else torch.sign(x) * torch.expm1(torch.abs(x))


class ThrombusSurrogateDataset(Dataset):
    def __init__(self, dataset_dir: str, split: Literal["train", "val", "test", "ood"]):
        self.dataset_dir = dataset_dir
        self.split = split
        self.files = sorted(glob.glob(os.path.join(dataset_dir, split, "sample_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No samples found in {os.path.join(dataset_dir, split)}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        fields = np.stack([data[name] for name in FIELD_NAMES], axis=0).astype(np.float32)
        fields = field_to_log(fields)
        return {
            "params": torch.from_numpy(data["params"].astype(np.float32)),
            "fields": torch.from_numpy(fields),
            "max_M_at": torch.tensor(float(data["max_M_at"]), dtype=torch.float32),
            "thrombosed_fraction": torch.tensor(float(data["thrombosed_fraction"]), dtype=torch.float32),
        }
