"""Training loop: data loss + weighted physics losses, CSV logging.

Responsibility
---------------
Standard supervised training loop over `data/dataset.py`
`ThrombusSurrogateDataset(split="train"/"val")`, optimizing
`neural/model.py` `ThrombusSurrogate` against a combination of:

* Data loss: MSE between predicted and mechanistic-solver ground-truth
  field grids.
* Physics losses: `neural/physics_losses.py` `total_physics_loss`
  (mass-conservation + non-negativity penalties), weighted per
  `configs/training.yaml` `physics_loss.weights`. Evaluated with
  `batch["fluid_mask"]` (`data/dataset.py`) so the exterior (non-fluid)
  raster cells don't dilute either penalty -- see `physics_losses.py`'s
  "Fluid-domain masking" docstring section.

Hyperparameters (optimizer, LR schedule, epochs, batch size, early
stopping) come from `configs/training.yaml` `optim`; only CSV logging is
implemented (`configs/training.yaml` `logging.backend: tensorboard` is not
supported -- this project does not add a tensorboard dependency).
"""

from __future__ import annotations

import argparse
import csv
import os

import torch
import yaml
from torch.utils.data import DataLoader

from thrombus_bench.data.dataset import ThrombusSurrogateDataset
from thrombus_bench.neural.model import ThrombusSurrogate
from thrombus_bench.neural.physics_losses import total_physics_loss


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
            data_loss = torch.nn.functional.mse_loss(pred, batch["fields"])
            loss = weights["data"] * data_loss
            if physics_cfg["enabled"]:
                phys = total_physics_loss(pred, weights, physics_cfg["residual_mode"], mask=batch["fluid_mask"])
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
                val_loss_sum += float(torch.nn.functional.mse_loss(pred, batch["fields"]))
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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    final = train(cfg, args.dataset_dir, args.checkpoint, args.log)
    print(f"Final: {final}")


if __name__ == "__main__":
    main()
