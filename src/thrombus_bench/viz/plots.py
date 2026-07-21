"""Visualization: thrombus maps, shear-gradient overlays, viscosity fields, error heatmaps.

Responsibility
---------------
Matplotlib plotting functions used by `mechanistic/run_simulation.py`
(exploratory single-run plots) and `benchmark/run_benchmark.py` (the
`results/report.md` bundle):

* `plot_thrombus_map`: mesh-triangulated view of the thrombosed region
  (M_at/FI threshold mask, `benchmark/metrics.thrombus_mask`), styled after
  the paper's Fig. 4-6 viscosity/thrombus maps.
* `plot_shear_gradient_overlay`: wall shear rate and its axial gradient
  along the vessel/aneurysm wall, highlighting negative-gradient regions
  (paper Fig. 3c, 9a, 10d/h).
* `plot_viscosity_field`: mesh-triangulated viscosity field (Eq. 18),
  matching paper Fig. 4b/c/5a/11.
* `plot_error_heatmap`: per-node error between neural and mechanistic
  predictions, mesh-triangulated.
* `plot_speed_vs_accuracy`: scatter of runtime vs. accuracy metric across
  test-set samples, mechanistic vs. neural (the benchmark's headline plot).
* `plot_ood_degradation`: bar/line comparison of in-distribution vs. OOD
  metrics (`benchmark/ood_eval.py`).
* `plot_calibration`: reliability diagram (`benchmark/calibration.py`).

Not yet implemented -- this is a scaffolding stub.
"""

from __future__ import annotations

import numpy as np


def plot_thrombus_map(node_coords: np.ndarray, elements: np.ndarray, mask: np.ndarray, ax=None):
    raise NotImplementedError("plots.plot_thrombus_map: not yet implemented")


def plot_shear_gradient_overlay(wall_coords: np.ndarray, shear_rate: np.ndarray, shear_gradient: np.ndarray, ax=None):
    raise NotImplementedError("plots.plot_shear_gradient_overlay: not yet implemented")


def plot_viscosity_field(node_coords: np.ndarray, elements: np.ndarray, viscosity: np.ndarray, ax=None):
    raise NotImplementedError("plots.plot_viscosity_field: not yet implemented")


def plot_error_heatmap(node_coords: np.ndarray, elements: np.ndarray, error: np.ndarray, ax=None):
    raise NotImplementedError("plots.plot_error_heatmap: not yet implemented")


def plot_speed_vs_accuracy(runtimes: dict, accuracies: dict, ax=None):
    raise NotImplementedError("plots.plot_speed_vs_accuracy: not yet implemented")


def plot_ood_degradation(degradation: dict, ax=None):
    raise NotImplementedError("plots.plot_ood_degradation: not yet implemented")


def plot_calibration(reliability_data: dict, ax=None):
    raise NotImplementedError("plots.plot_calibration: not yet implemented")
