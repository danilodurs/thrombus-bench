"""Training loop: data loss + weighted physics losses, CSV logging.

Responsibility
---------------
Standard supervised training loop over `data/dataset.py`
`ThrombusSurrogateDataset(split="train"/"val")`, optimizing
`neural/model.py` `ThrombusSurrogate` against a combination of:

* Data loss: MSE between predicted and mechanistic-solver ground-truth
  field grids -- plus `batch["M_at_wall"]` too, concatenated on as an extra
  target channel, if `cfg["model"]["predict_M_at_wall"]` is set (see
  `neural/model.py`'s docstring for this opt-in 12th output channel).
* Physics losses: `neural/physics_losses.py` `total_physics_loss`
  (mass-conservation + non-negativity penalties), weighted per
  `configs/training.yaml` `physics_loss.weights`. Evaluated with
  `batch["fluid_mask"]` (`data/dataset.py`) so the exterior (non-fluid)
  raster cells don't dilute either penalty -- see `physics_losses.py`'s
  "Fluid-domain masking" docstring section. Only ever given the first
  `cfg["model"]["output_channels"]` channels of `pred` (a no-op slice when
  `predict_M_at_wall` is off) -- `M_at_wall` is a different physical
  quantity/support region and isn't part of what these penalties reason
  about.

Hyperparameters (optimizer, LR schedule, epochs, batch size, early
stopping) come from `configs/training.yaml` `optim`; only CSV logging is
implemented (`configs/training.yaml` `logging.backend: tensorboard` is not
supported -- this project does not add a tensorboard dependency).

`train_continuous`: point-cloud training loop
--------------------------------------------------
`train_continuous` is the `ContinuousThrombusSurrogate`/
`PointCloudThrombusDataset` counterpart of `train` above (`docs/
continuous_surrogate_design.md` Phase 4/5), selected via `main`'s
`--continuous` flag (e.g. with `--config configs/continuous.yaml`) rather
than a separate console-script entry point, mirroring `generate_dataset.
main`'s `--extrapolation-param`-style mode-flag dispatch. It reuses `train`'s
optimizer/checkpointing/CSV-logging scaffolding (structurally identical);
batch consumption (ragged flat-points + `batch_index`) and loss
computation (per-point MSE with `data.excluded_temporal_channels` zeroed
out) are new, since the grid path's fixed-shape-batch/whole-grid-MSE
assumptions don't apply.

Physics loss: when `physics_loss.enabled` and `physics_loss.residual_mode:
autograd` (the only mode this path supports -- `finite_difference` only
applies to `train`'s grid path, see `physics_losses.py`'s module
docstring), `physics_loss.weights.mass_conservation` (if nonzero) adds
`physics_losses.continuous_mass_conservation_loss` -- PINN-style
collocation points (`physics_loss.n_collocation_points` per sample) plus a
true `torch.autograd.grad` divergence residual on the model's own
continuous output, per Phase 5. Only added to the *train* loss, mirroring
`train`'s own physics-loss-only-during-training convention -- val loss
here stays pure data loss. `benchmark.run_benchmark.run_benchmark_continuous`
(Phase 7) reports a separate accuracy metric (`benchmark.metrics.
field_rmse_pointwise`, not this module's masked-MSE training loss) on the
test split, so the two numbers are not directly comparable to each other
either. `nonnegativity` (finite-difference-only, operates on a raster) has
no continuous-path counterpart yet.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from thrombus_bench.data.dataset import (
    FIELD_NAMES,
    PointCloudThrombusDataset,
    ThrombusSurrogateDataset,
    pointcloud_collate_fn,
)
from thrombus_bench.neural.coordinate_decoder import VESSEL_LENGTH_MM, ContinuousThrombusSurrogate
from thrombus_bench.neural.model import ThrombusSurrogate
from thrombus_bench.neural.physics_losses import continuous_mass_conservation_loss, total_physics_loss

# Species whose concentration-cap clip rate makes them not physically
# trustworthy (see mechanistic/coupled_solver.py's "Known limitation" and
# README.md "Known limitations"): checked against actual clip_count_* QC
# data across 18 LHS-sampled runs (docs/continuous_surrogate_design.md
# Phase 4), not assumed -- T/PT/FI all clip in 100% of samples checked,
# every other species in 0%. Excluded from train_continuous's data loss by
# default via `data.excluded_temporal_channels` (FIELD_NAMES channel
# names) -- still loaded/predicted, just zero-weighted, so diagnostics can
# still inspect them.
DEFAULT_EXCLUDED_TEMPORAL_CHANNELS = ("conc_T", "conc_PT", "conc_FI")


def train(cfg: dict, dataset_dir: str, checkpoint_path: str, log_path: str) -> dict:
    """Train a `ThrombusSurrogate` per `cfg` (configs/training.yaml).
    Returns the final train/val loss dict."""

    torch.manual_seed(cfg.get("seed", 0))

    train_ds = ThrombusSurrogateDataset(dataset_dir, "train")
    val_ds = ThrombusSurrogateDataset(dataset_dir, "val")
    batch_size = min(cfg["optim"]["batch_size"], len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(cfg["optim"]["batch_size"], len(val_ds)))

    model = ThrombusSurrogate(cfg["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])

    physics_cfg = cfg["physics_loss"]
    weights = physics_cfg["weights"]
    predict_M_at_wall = bool(cfg["model"].get("predict_M_at_wall", False))
    n_field_channels = cfg["model"]["output_channels"]

    def _target(batch: dict) -> torch.Tensor:
        if predict_M_at_wall:
            return torch.cat([batch["fields"], batch["M_at_wall"].unsqueeze(1)], dim=1)
        return batch["fields"]

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_rows = []
    best_val = float("inf")
    epochs_since_improvement = 0

    for epoch in range(cfg["optim"]["epochs"]):
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["params"])
            data_loss = torch.nn.functional.mse_loss(pred, _target(batch))
            loss = weights["data"] * data_loss
            if physics_cfg["enabled"]:
                phys = total_physics_loss(
                    pred[:, :n_field_channels], weights, physics_cfg["residual_mode"], mask=batch["fluid_mask"]
                )
                for name, value in phys.items():
                    loss = loss + weights.get(name, 0.0) * value
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip_norm"])
            optimizer.step()
            train_loss_sum += float(loss.detach())
            n_batches += 1
        train_loss = train_loss_sum / max(1, n_batches)

        model.eval()
        val_loss_sum, n_val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["params"])
                val_loss_sum += float(torch.nn.functional.mse_loss(pred, _target(batch)))
                n_val_batches += 1
        val_loss = val_loss_sum / max(1, n_val_batches)

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            epochs_since_improvement = 0
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save({"model_state": model.state_dict(), "cfg": cfg["model"]}, checkpoint_path)
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= cfg["optim"]["early_stopping_patience"]:
            break

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)

    return log_rows[-1] if log_rows else {}


def _channel_weight_mask(excluded_channels: list[str], predict_M_at_wall: bool) -> torch.Tensor:
    """`(len(FIELD_NAMES) [+1 if predict_M_at_wall],)` tensor: `0.0` for
    each name in `excluded_channels` (`data/dataset.FIELD_NAMES` channel
    names, e.g. `"conc_T"`), `1.0` otherwise. The M_at channel (when
    present) is never excludable through this mechanism -- it's a
    different physical quantity (`predict_M_at_wall` opts it in
    separately), not one of the species-reliability channels
    `excluded_temporal_channels` is about."""

    weights = [0.0 if name in excluded_channels else 1.0 for name in FIELD_NAMES]
    if predict_M_at_wall:
        weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, channel_weight: torch.Tensor) -> torch.Tensor:
    """Mean squared error over `(n_points, n_channels)` tensors, restricted
    to channels with nonzero `channel_weight` -- i.e. the mean is over
    included channels only, not diluted by how many channels happen to be
    excluded (unlike a plain `((pred - target) * mask).pow(2).mean()`,
    which would shrink as more channels are zeroed regardless of the
    included channels' actual error)."""

    squared_error = (pred - target).pow(2) * channel_weight
    denom = channel_weight.sum() * pred.shape[0]
    return squared_error.sum() / denom.clamp_min(1.0)


def train_continuous(cfg: dict, dataset_dir: str, checkpoint_path: str, log_path: str) -> dict:
    """Train a `ContinuousThrombusSurrogate` per `cfg` (e.g.
    `configs/continuous.yaml`) -- see module docstring. Returns the final
    train/val loss dict, same shape as `train`'s."""

    torch.manual_seed(cfg.get("seed", 0))

    data_cfg = cfg["data"]
    points_per_sample = data_cfg.get("points_per_sample")
    train_ds = PointCloudThrombusDataset(dataset_dir, "train", points_per_sample=points_per_sample)
    val_ds = PointCloudThrombusDataset(dataset_dir, "val", points_per_sample=points_per_sample)
    batch_size = min(cfg["optim"]["batch_size"], len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pointcloud_collate_fn)
    val_loader = DataLoader(
        val_ds, batch_size=min(cfg["optim"]["batch_size"], len(val_ds)), collate_fn=pointcloud_collate_fn
    )

    model = ContinuousThrombusSurrogate(cfg["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])

    predict_M_at_wall = bool(cfg["model"].get("predict_M_at_wall", False))
    excluded_channels = data_cfg.get("excluded_temporal_channels", list(DEFAULT_EXCLUDED_TEMPORAL_CHANNELS))
    channel_weight = _channel_weight_mask(excluded_channels, predict_M_at_wall)

    physics_cfg = cfg.get("physics_loss", {})
    physics_enabled = bool(physics_cfg.get("enabled", False))
    if physics_enabled and physics_cfg.get("residual_mode") != "autograd":
        raise ValueError(
            f"train_continuous: physics_loss.enabled requires residual_mode='autograd' for this path "
            f"(got {physics_cfg.get('residual_mode')!r}) -- 'finite_difference' only applies to train()'s "
            "grid-projection path, see physics_losses.py's module docstring."
        )
    mass_conservation_weight = physics_cfg.get("weights", {}).get("mass_conservation", 0.0)
    n_collocation_points = physics_cfg.get("n_collocation_points", 32)
    vessel_length_mm = cfg["model"].get("vessel_length_mm", VESSEL_LENGTH_MM)
    collocation_rng = np.random.default_rng(cfg.get("seed", 0))

    def _target(batch: dict) -> torch.Tensor:
        if predict_M_at_wall:
            return torch.cat([batch["fields"], batch["M_at_target"].unsqueeze(-1)], dim=-1)
        return batch["fields"]

    def _forward(batch: dict) -> torch.Tensor:
        return model(batch["params_with_time"], batch["node_coords"], batch["batch_index"], batch["geometry_mm"])

    def _train_loss(batch: dict) -> torch.Tensor:
        loss = _masked_mse(_forward(batch), _target(batch), channel_weight)
        if physics_enabled and mass_conservation_weight:
            loss = loss + mass_conservation_weight * continuous_mass_conservation_loss(
                model, batch["params_with_time"], batch["geometry_mm"], n_collocation_points,
                vessel_length_mm, rng=collocation_rng,
            )
        return loss

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_rows = []
    best_val = float("inf")
    epochs_since_improvement = 0

    for epoch in range(cfg["optim"]["epochs"]):
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = _train_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip_norm"])
            optimizer.step()
            train_loss_sum += float(loss.detach())
            n_batches += 1
        train_loss = train_loss_sum / max(1, n_batches)

        model.eval()
        val_loss_sum, n_val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                val_loss_sum += float(_masked_mse(_forward(batch), _target(batch), channel_weight))
                n_val_batches += 1
        val_loss = val_loss_sum / max(1, n_val_batches)

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            epochs_since_improvement = 0
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save({"model_state": model.state_dict(), "cfg": cfg["model"]}, checkpoint_path)
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= cfg["optim"]["early_stopping_patience"]:
            break

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)

    return log_rows[-1] if log_rows else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/demo_cpu.yaml")
    parser.add_argument("--dataset-dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--log", type=str, default="runs/train_log.csv")
    parser.add_argument(
        "--continuous", action="store_true",
        help="Train ContinuousThrombusSurrogate on PointCloudThrombusDataset (train_continuous) instead of "
        "the default grid-projection ThrombusSurrogate -- use with e.g. --config configs/continuous.yaml "
        "and a dataset generated without --also-save-raster.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.continuous:
        final = train_continuous(cfg, args.dataset_dir, args.checkpoint, args.log)
    else:
        final = train(cfg, args.dataset_dir, args.checkpoint, args.log)
    print(f"Final: {final}")


if __name__ == "__main__":
    main()
