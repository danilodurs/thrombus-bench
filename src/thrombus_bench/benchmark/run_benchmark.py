"""CLI: run both models on the test set, produce the comparison report + plots.

Responsibility
---------------
End-to-end benchmark entrypoint:

1. Load the trained neural surrogate (`neural/model.py`, from a checkpoint
   written by `neural/train.py`) and run it on `data/dataset.py`
   `split="test"` and `split="edge_holdout"`.
2. Compute `benchmark/metrics.py` accuracy metrics (field RMSE -- both
   all-cells and fluid-only via `data/dataset.py`'s `fluid_mask`, max-M_at
   and thrombosed-fraction summary errors), `benchmark/edge_holdout_eval.py`
   edge-of-domain degradation, `benchmark/calibration.py` UQ calibration
   (MC-dropout or deep-ensemble, per `configs/training.yaml`
   `model.uncertainty.method` -- see `_build_uncertainty_wrapper`'s
   docstring for the deep-ensemble path's known limitation), and a runtime
   comparison (neural forward-pass time vs. a freshly re-timed mechanistic
   solve on a handful of test-set parameter vectors).
3. Fit `neural/baselines.py`'s `MeanFieldBaseline`/`NearestNeighborBaseline`
   on the train split and compute the same field RMSE (all-cells and
   fluid-only) for them alongside the FNO surrogate, on both the test and
   edge-holdout splits -- without this, there's no way to tell whether the
   FNO is adding value over something trivial.
4. Render plots via `viz/plots.py` and assemble a single Markdown + PNG
   bundle at `results/report.md`.
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.dataset import ThrombusSurrogateDataset
from ..mechanistic.coupled_solver import run_coupled_simulation
from ..mechanistic.mesh import GeometryConfig, build_aneurysm_mesh
from ..neural.baselines import MeanFieldBaseline, NearestNeighborBaseline
from ..neural.model import ThrombusSurrogate
from ..neural.uncertainty import DeepEnsemble, MCDropoutWrapper
from . import calibration as calibration_mod
from . import edge_holdout_eval as edge_holdout_eval_mod
from .metrics import field_rmse, runtime_comparison
from ..data.generate_dataset import PARAM_ORDER
from ..data.sampler import ParameterSpace, denormalize_params
from ..viz import plots as plots_mod


def _time_mechanistic_solve(params_row: np.ndarray, physio_base: dict, mesh_cfg: dict, end_time_s: float, dt_s: float) -> float:
    """`params_row` must be in physical units (`data/sampler.denormalize_params`'d
    back out of `data/dataset.py`'s [-1, 1]-normalized `params`) -- this
    rebuilds the actual mesh/physio for a re-timed mechanistic solve, not a
    model input."""

    sample = dict(zip(PARAM_ORDER, params_row))
    geom = GeometryConfig(
        vessel_diameter_mm=float(sample["vessel_diameter_mm"]),
        aneurysm_diameter_mm=float(sample["aneurysm_diameter_mm"]),
        vessel_length_mm=50.0,
    )
    tagged_mesh = build_aneurysm_mesh(geom, mesh_cfg)
    physio = {k: (dict(v) if isinstance(v, dict) else v) for k, v in physio_base.items()}
    t0 = time.perf_counter()
    run_coupled_simulation(
        tagged_mesh, inlet_velocity_m_s=float(sample["inlet_velocity_cm_s"]) / 100.0, physio=physio,
        end_time_s=end_time_s, dt_s=dt_s, output_every_n_steps=int(round(end_time_s / dt_s)),
        flow_resolve_every_n_steps=int(round(end_time_s / dt_s)),
    )
    return time.perf_counter() - t0


def _build_uncertainty_wrapper(model: ThrombusSurrogate, model_cfg: dict, uncertainty_cfg: dict):
    """Dispatch to `DeepEnsemble` or `MCDropoutWrapper` per
    `configs/training.yaml` `model.uncertainty.method`.

    Known limitation: `train.py` only ever saves a single trained seed, so
    the `deep_ensemble` path here cannot load independently trained
    checkpoints per member. `model_factory` below instead returns freshly,
    randomly initialized copies of the architecture from `checkpoint["cfg"]`
    -- none of them are the trained weights. This exercises the DeepEnsemble
    mechanics (variance across members) but is NOT a trained ensemble and
    should not be read as one; treat any calibration numbers produced via
    this path as a placeholder until `train.py` supports saving multiple
    independently trained seeds.
    """

    method = uncertainty_cfg["method"]
    if method == "deep_ensemble":
        model_factory = lambda: ThrombusSurrogate(model_cfg).eval()
        return DeepEnsemble(model_factory, n_members=uncertainty_cfg["n_ensemble_members"])
    if method == "mc_dropout":
        return MCDropoutWrapper(
            model,
            n_samples=uncertainty_cfg["mc_dropout_n_samples"],
            dropout_rate=uncertainty_cfg["mc_dropout_rate"],
        )
    raise ValueError(f"Unknown model.uncertainty.method: {method!r}")


def run_benchmark(
    training_cfg: dict,
    physio_base: dict,
    mesh_cfg: dict,
    checkpoint_path: str,
    dataset_dir: str,
    output_dir: str = "results",
    n_runtime_samples: int = 2,
    mechanistic_end_time_s: float = 0.4,
    mechanistic_dt_s: float = 0.1,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = ThrombusSurrogate(checkpoint["cfg"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    train_ds = ThrombusSurrogateDataset(dataset_dir, "train")
    test_ds = ThrombusSurrogateDataset(dataset_dir, "test")
    edge_holdout_ds = ThrombusSurrogateDataset(dataset_dir, "edge_holdout")

    test_loader = DataLoader(test_ds, batch_size=len(test_ds))
    test_batch = next(iter(test_loader))
    edge_holdout_loader = DataLoader(edge_holdout_ds, batch_size=len(edge_holdout_ds))
    edge_holdout_batch = next(iter(edge_holdout_loader))

    with torch.no_grad():
        t0 = time.perf_counter()
        pred_fields = model(test_batch["params"])
        neural_time_per_sample = (time.perf_counter() - t0) / len(test_ds)

    accuracy = field_rmse(pred_fields.numpy(), test_batch["fields"].numpy(), mask=test_batch["fluid_mask"].numpy())

    # Baseline comparison (Task 2.4): without these, there's no way to tell
    # whether the FNO is adding value over something trivial. Both are fit
    # on the train split only, then evaluated with the same field_rmse
    # (masked + unmasked) as the FNO, on both test and edge-holdout.
    mean_field_baseline = MeanFieldBaseline().fit(train_ds).eval()
    nearest_neighbor_baseline = NearestNeighborBaseline().fit(train_ds).eval()

    def _field_rmse_for(candidate_model, batch: dict) -> dict:
        with torch.no_grad():
            pred = candidate_model(batch["params"])
        return field_rmse(pred.numpy(), batch["fields"].numpy(), mask=batch["fluid_mask"].numpy())

    model_comparison = {
        "Mean-field baseline": {
            "test": _field_rmse_for(mean_field_baseline, test_batch),
            "edge_holdout": _field_rmse_for(mean_field_baseline, edge_holdout_batch),
        },
        "Nearest-neighbor baseline": {
            "test": _field_rmse_for(nearest_neighbor_baseline, test_batch),
            "edge_holdout": _field_rmse_for(nearest_neighbor_baseline, edge_holdout_batch),
        },
        "FNO surrogate": {
            "test": accuracy,
            "edge_holdout": _field_rmse_for(model, edge_holdout_batch),
        },
    }

    # Runtime comparison: re-time the mechanistic solver on a few test-set
    # parameter vectors (short window, matching generate_dataset.py's
    # scope), vs. the neural forward pass just timed above.
    n_runtime = min(n_runtime_samples, len(test_ds))
    param_space = ParameterSpace()
    mech_times = np.array(
        [
            _time_mechanistic_solve(
                denormalize_params(test_batch["params"][i].numpy(), param_space),
                physio_base, mesh_cfg, mechanistic_end_time_s, mechanistic_dt_s,
            )
            for i in range(n_runtime)
        ]
    )
    neural_times = np.full(n_runtime, neural_time_per_sample)
    runtime = runtime_comparison(mech_times, neural_times)

    edge_holdout_degradation = edge_holdout_eval_mod.evaluate_edge_holdout_degradation(model, test_ds, edge_holdout_ds)

    uq_model = _build_uncertainty_wrapper(model, checkpoint["cfg"], training_cfg["model"]["uncertainty"])
    with torch.no_grad():
        pred_mean, pred_var = uq_model.predict(test_batch["params"])
    reliability = calibration_mod.reliability_diagram_data(
        pred_mean.numpy(), pred_var.numpy(), test_batch["fields"].numpy(), n_bins=5
    )
    ece = calibration_mod.expected_calibration_error(pred_mean.numpy(), pred_var.numpy(), test_batch["fields"].numpy(), n_bins=5)

    fig, ax = plt.subplots()
    plots_mod.plot_edge_holdout_degradation(edge_holdout_degradation, ax=ax)
    fig.savefig(os.path.join(output_dir, "edge_holdout_degradation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plots_mod.plot_calibration(reliability, ax=ax)
    fig.savefig(os.path.join(output_dir, "calibration.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    _write_report(
        output_dir, accuracy, runtime, edge_holdout_degradation, ece, model_comparison,
        n_test=len(test_ds), n_edge_holdout=len(edge_holdout_ds),
    )


def _write_report(
    output_dir: str, accuracy: dict, runtime: dict, edge_holdout_degradation: dict, ece: float,
    model_comparison: dict,
    n_test: int, n_edge_holdout: int,
) -> None:
    comparison_rows = [
        f"| {name} | {m['test']['overall']:.4f} | {m['test']['fluid_only']:.4f} | "
        f"{m['edge_holdout']['overall']:.4f} | {m['edge_holdout']['fluid_only']:.4f} |"
        for name, m in model_comparison.items()
    ]

    lines = [
        "# thrombus-bench benchmark report",
        "",
        f"Test set: {n_test} samples. Edge holdout set: {n_edge_holdout} samples.",
        "",
        "## Accuracy",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Overall field RMSE (log-space, all cells) | {accuracy['overall']:.4f} |",
        f"| Overall field RMSE (log-space, fluid cells only) | {accuracy['fluid_only']:.4f} |",
        "",
        "Fluid-only excludes the rasterization bounding box's exterior "
        "(non-fluid) cells -- see `data/generate_dataset._fluid_mask` -- so "
        "it isn't inflated by easy, mostly-constant background reconstruction; "
        "the all-cells number is kept alongside it for comparison.",
        "",
        "## Model comparison (field RMSE, log-space)",
        "",
        "| Model | Test (all cells) | Test (fluid only) | Edge holdout (all cells) | Edge holdout (fluid only) |",
        "|---|---|---|---|---|",
        *comparison_rows,
        "",
        "`Mean-field baseline`/`Nearest-neighbor baseline` (`neural/baselines.py`) "
        "are trivial predictors fit on the train split only -- a constant "
        "(per-pixel training mean) and a lookup table (nearest training "
        "sample by normalized parameter distance), respectively -- included "
        "so the FNO surrogate's numbers above can be judged against a floor "
        "rather than in isolation.",
        "",
        "## Runtime",
        "",
        "| Model | Mean time per sample (s) |",
        "|---|---|",
        f"| Mechanistic | {runtime['mechanistic_mean_s']:.3f} |",
        f"| Neural | {runtime['neural_mean_s']:.6f} |",
        f"| Speedup | {runtime['speedup_factor']:.1f}x |",
        "",
        "## Edge-of-domain holdout degradation",
        "",
        f"![Edge holdout degradation](edge_holdout_degradation.png)",
        "",
        f"Core-range (test) RMSE: {edge_holdout_degradation['test']['overall']:.4f}. "
        f"Edge-holdout RMSE: {edge_holdout_degradation['edge_holdout']['overall']:.4f}. "
        f"Degradation ratio: {edge_holdout_degradation['degradation_ratio']:.2f}x.",
        "",
        "## Uncertainty calibration",
        "",
        f"![Calibration](calibration.png)",
        "",
        f"Expected calibration error (predicted variance vs. observed squared error): {ece:.4f}.",
        "",
        "---",
        "",
        "*Generated by `thrombus-benchmark` (`benchmark/run_benchmark.py`). "
        "Field RMSE is computed in the log-compressed space used for training "
        "(`data/dataset.field_to_log`), not physical units -- see that module's "
        "docstring. This is a reduced-scale demonstration run; see README.md "
        '"Project status" for dataset/model scale caveats.*',
    ]
    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=str, default="configs/demo_cpu.yaml")
    parser.add_argument("--physio-config", type=str, default="configs/physio_params.yaml")
    parser.add_argument("--geometry-config", type=str, default="configs/geometry.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--dataset-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--target-num-elements", type=int, default=800)
    args = parser.parse_args()

    with open(args.training_config) as f:
        training_cfg = yaml.safe_load(f)
    with open(args.physio_config) as f:
        physio_base = yaml.safe_load(f)
    with open(args.geometry_config) as f:
        geometry_yaml = yaml.safe_load(f)
    mesh_cfg = dict(geometry_yaml["mesh"])
    mesh_cfg["target_num_elements"] = args.target_num_elements

    run_benchmark(training_cfg, physio_base, mesh_cfg, args.checkpoint, args.dataset_dir, args.output_dir)
    print(f"Wrote benchmark report to {os.path.join(args.output_dir, 'report.md')}")


if __name__ == "__main__":
    main()
