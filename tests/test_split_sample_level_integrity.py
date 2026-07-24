"""Confirms `data/sampler.split_train_val_test_edge_holdout`'s sample-level
partition guarantee still holds end-to-end for the point-cloud data path
(Phase 7, `docs/continuous_surrogate_design.md`): a whole simulation
(every checkpoint, every mesh node together) must land in exactly one
split, never spread across two.

The sampler itself partitions *samples* (parameter dicts) before any
mechanistic run happens, and `data/generate_dataset._generate_from_splits`
writes each sample's entire `.npz` (all its checkpoints) to exactly one
split directory -- so this is largely a structural guarantee already. This
test checks it against the *actual generated files* for the multi-
checkpoint point-cloud schema specifically, rather than only at the
abstract sampler-partition level (which the existing
`test_extrapolation_holdout.py` already covers for
`sample_with_extrapolation_holdout`, but not by reading real generated
files back through `PointCloudThrombusDataset`)."""

from __future__ import annotations

import numpy as np
import yaml

from thrombus_bench.data.dataset import PointCloudThrombusDataset
from thrombus_bench.data.generate_dataset import generate_dataset

PHYSIO_PATH = "configs/physio_params.yaml"
GEOMETRY_PATH = "configs/geometry.yaml"


def test_no_sample_appears_in_more_than_one_split(tmp_path):
    with open(PHYSIO_PATH) as f:
        physio_base = yaml.safe_load(f)
    with open(GEOMETRY_PATH) as f:
        geometry_yaml = yaml.safe_load(f)
    mesh_cfg = dict(geometry_yaml["mesh"])
    mesh_cfg["target_num_elements"] = 150

    config = {
        "n_train": 3, "n_val": 2, "n_test": 2, "n_edge_holdout": 2,
        "edge_holdout_quantile": 0.9, "seed": 0,
    }
    output_dir = str(tmp_path / "data")
    counts, _ = generate_dataset(
        config, physio_base, mesh_cfg, output_dir,
        end_time_s=0.3, dt_s=0.1, grid_size=(8, 8), n_snapshots=3,
    )
    assert sum(counts.values()) == 9

    split_names = ["train", "val", "test", "edge_holdout"]
    datasets = {name: PointCloudThrombusDataset(output_dir, name) for name in split_names}

    # Each dataset item's underlying sample is identified by its raw
    # `params` vector (an LHS draw -- astronomically unlikely to collide
    # across genuinely different samples). Collect every (split, params)
    # pair and confirm no params vector is associated with more than one
    # split -- this would only happen if a sample's checkpoints/nodes got
    # split across directories.
    params_by_split: dict[str, set[tuple]] = {}
    for name, ds in datasets.items():
        seen = set()
        for file_idx, _ in ds._index:
            data = np.load(ds.files[file_idx])
            seen.add(tuple(np.round(data["params"], 8).tolist()))
        params_by_split[name] = seen

    all_pairs = [(name, params) for name, seen in params_by_split.items() for params in seen]
    all_params = [params for _, params in all_pairs]
    assert len(all_params) == len(set(all_params)), (
        "a sample's params vector appears under more than one split -- "
        "sample-level partitioning was violated for the point-cloud path"
    )

    # Also confirm every checkpoint within one sample's own .npz file is
    # entirely attributed to that same file/split (not, e.g., accidentally
    # counted under two different file_idx values) -- one dataset item per
    # checkpoint, all sharing the same file.
    for name, ds in datasets.items():
        file_idx_seen = {file_idx for file_idx, _ in ds._index}
        assert file_idx_seen == set(range(len(ds.files))), (
            f"{name}: not every file in the split directory is represented in the dataset index"
        )


def test_multi_checkpoint_sample_stays_whole_within_its_split(tmp_path):
    """A single sample generated with n_snapshots > 1 must contribute ALL
    of its checkpoints to the SAME split's PointCloudThrombusDataset --
    i.e. len(dataset) for that split's one file equals n_snapshots, not a
    partial subset."""

    with open(PHYSIO_PATH) as f:
        physio_base = yaml.safe_load(f)
    with open(GEOMETRY_PATH) as f:
        geometry_yaml = yaml.safe_load(f)
    mesh_cfg = dict(geometry_yaml["mesh"])
    mesh_cfg["target_num_elements"] = 150

    config = {"n_train": 1, "n_val": 0, "n_test": 0, "n_edge_holdout": 0, "edge_holdout_quantile": 0.9, "seed": 1}
    output_dir = str(tmp_path / "data")
    generate_dataset(config, physio_base, mesh_cfg, output_dir, end_time_s=0.3, dt_s=0.1, grid_size=(8, 8), n_snapshots=3)

    train_ds = PointCloudThrombusDataset(output_dir, "train")
    data = np.load(train_ds.files[0])
    n_snapshots_saved = data["time_s"].shape[0]

    checkpoints_for_file_0 = [checkpoint_idx for file_idx, checkpoint_idx in train_ds._index if file_idx == 0]
    assert len(checkpoints_for_file_0) == n_snapshots_saved
    assert sorted(checkpoints_for_file_0) == list(range(n_snapshots_saved))
