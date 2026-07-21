"""Training loop: data loss + weighted physics losses, CSV/tensorboard logging.

Responsibility
---------------
Standard supervised training loop over `data/dataset.py`
`ThrombusSurrogateDataset(split="train"/"val")`, optimizing
`neural/model.py` `ThrombusSurrogate` against a combination of:

* Data loss: MSE (or similar) between predicted and mechanistic-solver
  ground-truth fields.
* Physics losses: `neural/physics_losses.py` `total_physics_loss`, weighted
  per `configs/training.yaml` `physics_loss.weights`.

Hyperparameters (optimizer, LR schedule, epochs, batch size, early
stopping) come from `configs/training.yaml` `optim`; logging backend
(csv/tensorboard) and checkpoint directory from `configs/training.yaml`
`logging`.

CLI: `thrombus-train training=default`.

Not yet implemented -- this is a scaffolding stub. Depends on
`data/dataset.py`, `neural/model.py`, and `neural/physics_losses.py`.
"""

from __future__ import annotations

import argparse


def train(cfg: dict) -> None:
    raise NotImplementedError("train.train: not yet implemented")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.parse_args()
    raise NotImplementedError("train CLI: pending data/dataset.py and neural/model.py implementation")


if __name__ == "__main__":
    main()
