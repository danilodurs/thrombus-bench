"""Visualization: thrombus maps, shear-gradient overlays, viscosity fields, error heatmaps.

Responsibility
---------------
Matplotlib plotting functions used by `mechanistic/run_simulation.py`
(exploratory single-run plots, see `notebooks/01_explore_mechanistic_baseline.ipynb`)
and `benchmark/run_benchmark.py` (the `results/report.md` bundle):

* `plot_mesh_field`: mesh-triangulated view of a nodal scalar field (shared
  by thrombus maps / viscosity fields / error heatmaps -- they differ only
  in which field and colormap are passed).
* `plot_speed_vs_accuracy`: scatter of runtime vs. accuracy metric across
  test-set samples, mechanistic vs. neural (the benchmark's headline plot).
* `plot_edge_holdout_degradation`: bar comparison of core-range vs.
  edge-of-domain RMSE (`benchmark/edge_holdout_eval.py`).
* `plot_calibration`: reliability diagram (`benchmark/calibration.py`).
"""

from __future__ import annotations

import numpy as np


def plot_mesh_field(node_coords: np.ndarray, elements: np.ndarray, field: np.ndarray, ax, title: str = "", cmap: str = "viridis"):
    """Triangulated pseudocolor plot of a nodal scalar field. Shared
    implementation for thrombus maps, viscosity fields, and error heatmaps
    -- these differ only in which `field`/`cmap` is passed."""

    n_vertices = node_coords.shape[1]
    tpc = ax.tripcolor(node_coords[0], node_coords[1], elements.T, field[:n_vertices], shading="gouraud", cmap=cmap)
    ax.set_aspect("equal")
    ax.set_title(title)
    return tpc


def plot_thrombus_map(node_coords: np.ndarray, elements: np.ndarray, mask: np.ndarray, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    return plot_mesh_field(node_coords, elements, mask.astype(float), ax, title="Thrombosed region", cmap="Reds")


def plot_viscosity_field(node_coords: np.ndarray, elements: np.ndarray, viscosity: np.ndarray, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    return plot_mesh_field(node_coords, elements, viscosity, ax, title="Viscosity [Pa.s]", cmap="viridis")


def plot_error_heatmap(node_coords: np.ndarray, elements: np.ndarray, error: np.ndarray, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    return plot_mesh_field(node_coords, elements, error, ax, title="Prediction error", cmap="magma")


def plot_shear_gradient_overlay(wall_coords: np.ndarray, shear_rate: np.ndarray, shear_gradient: np.ndarray, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    order = np.argsort(wall_coords)
    ax2 = ax.twinx()
    ax.plot(wall_coords[order], shear_rate[order], color="tab:blue", label="shear rate")
    ax2.plot(wall_coords[order], shear_gradient[order], color="tab:red", label="d(shear)/dx")
    ax2.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("wall shear rate [1/s]", color="tab:blue")
    ax2.set_ylabel("d(shear rate)/dx", color="tab:red")
    return ax


def plot_speed_vs_accuracy(runtimes: dict, accuracies: dict, ax=None):
    """runtimes/accuracies: {"mechanistic": array, "neural": array}."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    for name, color in (("mechanistic", "tab:blue"), ("neural", "tab:orange")):
        ax.scatter(runtimes[name], accuracies[name], label=name, color=color, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("runtime [s]")
    ax.set_ylabel("error (RMSE)")
    ax.legend()
    return ax


def plot_edge_holdout_degradation(degradation: dict, ax=None):
    """degradation: output of
    `benchmark.edge_holdout_eval.evaluate_edge_holdout_degradation`."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    labels = ["core-range (test)", "edge-of-domain (edge_holdout)"]
    values = [degradation["test"]["overall"], degradation["edge_holdout"]["overall"]]
    ax.bar(labels, values, color=["tab:blue", "tab:red"])
    ax.set_ylabel("overall field RMSE")
    ax.set_title(f"Edge-holdout degradation ratio: {degradation['degradation_ratio']:.2f}x")
    return ax


def plot_calibration(reliability_data: dict, ax=None):
    """reliability_data: output of `benchmark.calibration.reliability_diagram_data`."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    mv, mse = reliability_data["mean_variance"], reliability_data["mean_squared_error"]
    max_val = max(mv.max(), mse.max()) if len(mv) else 1.0
    ax.plot([0, max_val], [0, max_val], "k--", label="perfect calibration")
    ax.scatter(mv, mse, color="tab:purple")
    ax.set_xlabel("predicted variance")
    ax.set_ylabel("observed squared error")
    ax.legend()
    return ax
