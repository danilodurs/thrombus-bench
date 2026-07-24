"""Tests for `neural/train.py`'s `train_continuous` (Phase 4, `docs/
continuous_surrogate_design.md`): a basic "is the training loop actually
learning something" sanity check on a tiny synthetic point-cloud dataset
(constructed directly, bypassing the mechanistic solver entirely for
speed) -- not a real convergence test. Also covers the channel-exclusion
weighting helper directly."""

from __future__ import annotations

import csv

import numpy as np
import torch

from thrombus_bench.data.dataset import FIELD_NAMES
from thrombus_bench.data.generate_dataset import PARAM_ORDER, _ALL_SPECIES
from thrombus_bench.neural.train import DEFAULT_EXCLUDED_TEMPORAL_CHANNELS, _channel_weight_mask, train_continuous


def _write_synthetic_sample(path, seed: int, target_value: float, n_nodes: int = 20) -> None:
    """A minimal, schema-valid point-cloud `.npz` sample with an easy,
    constant regression target (every point/channel == `target_value`) --
    deliberately trivial so a handful of optimizer steps can visibly
    reduce the loss, without needing a real mechanistic run."""

    rng = np.random.default_rng(seed)
    params = {
        "aneurysm_diameter_mm": 7.0, "vessel_diameter_mm": 3.2, "inlet_velocity_cm_s": 47.0,
        "platelet_conc_plt_ml": 3.5e8, "heparin_conc_uM": 2.0, "prothrombin_uM": 1.1,
        "antithrombin_uM": 2.844, "fibrinogen_uM": 7.0,
    }
    node_coords = np.column_stack([rng.uniform(0.0, 0.05, n_nodes), rng.uniform(0.0, 0.0032, n_nodes)])
    n_wall = 3
    wall_dofs = np.arange(n_wall, dtype=np.int64)

    result = {
        "params": np.array([params[name] for name in PARAM_ORDER], dtype=np.float64),
        "node_coords": node_coords.astype(np.float64),
        "triangles": np.zeros((0, 3), dtype=np.int32),
        "fields": np.full((1, n_nodes, len(FIELD_NAMES)), target_value, dtype=np.float32),
        "wall_dofs": wall_dofs,
        "wall_node_coords": node_coords[wall_dofs].astype(np.float64),
        "M_at_wall_values": np.full((1, n_wall), target_value, dtype=np.float32),
        "time_s": np.array([0.3], dtype=np.float64),
        "thrombin_fibrin_reliable_at_checkpoint": np.array([True]),
        "converged": True,
        "flow_n_iterations": 1,
        "flow_residual": 1e-8,
        "thrombin_fibrin_reliable": True,
        "max_M_at": 0.0,
        "thrombosed_fraction": 0.0,
        **{f"clip_count_{name}": 0 for name in _ALL_SPECIES},
        **{f"conc_{name}_min": 0.0 for name in _ALL_SPECIES},
        **{f"conc_{name}_max": 0.0 for name in _ALL_SPECIES},
    }
    np.savez(path, **result)


def _tiny_continuous_cfg() -> dict:
    return {
        "seed": 0,
        "data": {"points_per_sample": None, "excluded_temporal_channels": list(DEFAULT_EXCLUDED_TEMPORAL_CHANNELS)},
        "model": {
            "encoder": {"param_dim": 9, "latent_grid_size": (8, 8), "hidden_channels": 8, "n_layers": 1},
            "operator_core": {"type": "fno", "fno": {"modes": 2, "hidden_channels": 8, "n_layers": 1}},
            "coordinate_encoding": {"num_frequency_bands": 4},
            "coordinate_decoder": {"mlp_hidden": 16, "n_residual_blocks": 1},
            "output_channels": 11,
            "predict_M_at_wall": True,
            "uncertainty": {"mc_dropout_rate": 0.0},
        },
        "optim": {
            "epochs": 30, "batch_size": 2, "lr": 5.0e-3, "weight_decay": 0.0,
            "grad_clip_norm": 10.0, "early_stopping_patience": 1000,
        },
    }


def test_train_continuous_reduces_loss_on_tiny_synthetic_batch(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    for i in range(3):
        _write_synthetic_sample(train_dir / f"sample_{i:04d}.npz", seed=i, target_value=0.5)
    _write_synthetic_sample(val_dir / "sample_0000.npz", seed=99, target_value=0.5)

    checkpoint_path = tmp_path / "model.pt"
    log_path = tmp_path / "log.csv"

    final = train_continuous(_tiny_continuous_cfg(), str(tmp_path), str(checkpoint_path), str(log_path))
    assert "train_loss" in final and "val_loss" in final
    assert torch.isfinite(torch.tensor(final["train_loss"]))

    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    train_losses = [float(r["train_loss"]) for r in rows]

    assert len(train_losses) >= 2
    # Basic sanity check, not a convergence test: loss on this trivial
    # constant-target task should end up well below where it started.
    assert train_losses[-1] < train_losses[0] * 0.5
    assert checkpoint_path.exists()


def test_channel_weight_mask_zeroes_excluded_channels_only():
    mask = _channel_weight_mask(["conc_T", "conc_PT", "conc_FI"], predict_M_at_wall=False)
    assert mask.shape == (len(FIELD_NAMES),)
    for name, weight in zip(FIELD_NAMES, mask.tolist()):
        expected = 0.0 if name in ("conc_T", "conc_PT", "conc_FI") else 1.0
        assert weight == expected


def test_channel_weight_mask_appends_unexcludable_m_at_channel():
    mask = _channel_weight_mask(["conc_T"], predict_M_at_wall=True)
    assert mask.shape == (len(FIELD_NAMES) + 1,)
    assert mask[-1].item() == 1.0
