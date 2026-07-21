"""PyTorch Dataset/DataLoader wrapping mechanistic simulation outputs.

Responsibility
---------------
Load the `.npz`/`.h5` simulation records written by `generate_dataset.py`
and expose them as a `torch.utils.data.Dataset` yielding, per sample:

* `params`: the sampled parameter vector (geometry + physiology + inlet
  velocity), matching `configs/training.yaml` `model.encoder.param_dim`.
* `mesh_fields`: mesh node coordinates and connectivity, resampled onto the
  fixed-resolution latent grid used by the neural operator
  (`neural/encoder.py`), via nearest-neighbor or linear interpolation.
* `target_fields`: time series of velocity, viscosity, species
  concentrations, and surface coverage fields (the neural surrogate's
  prediction targets), matching `configs/training.yaml` `model.output_channels`.

Train/val/test/OOD splits are read from separate subdirectories written by
`generate_dataset.py` (see `sampler.split_train_val_test_ood`); a
`split: Literal["train", "val", "test", "ood"]` constructor argument selects
among them.

Not yet implemented -- this is a scaffolding stub. Depends on
`generate_dataset.py` for its input data format.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.utils.data import Dataset


class ThrombusSurrogateDataset(Dataset):
    def __init__(self, dataset_dir: str, split: Literal["train", "val", "test", "ood"]):
        self.dataset_dir = dataset_dir
        self.split = split
        raise NotImplementedError("dataset.ThrombusSurrogateDataset: not yet implemented")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError
