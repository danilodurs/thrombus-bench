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
   solve on a handful of test-set parameter vectors). If the checkpoint was
   trained with `model.predict_M_at_wall: true` (`neural/model.py`), also
   computes `benchmark/metrics.thrombus_mask`/`thrombus_iou` between
   predicted and reference thrombosed-region masks on the test split (see
   `_FieldChannelsOnly` for how the resulting 12-channel predictions are
   kept from leaking into the metrics above, which only know about the
   original 11 physical field channels).
3. Fit `neural/baselines.py`'s `MeanFieldBaseline`/`NearestNeighborBaseline`
   on the train split and compute the same field RMSE (all-cells and
   fluid-only) for them alongside the FNO surrogate, on both the test and
   edge-holdout splits -- without this, there's no way to tell whether the
   FNO is adding value over something trivial.
4. Render plots via `viz/plots.py` and assemble a single Markdown + PNG
   bundle at `results/report.md`.

`--continuous` dispatches instead to `run_benchmark_continuous` (`docs/
continuous_surrogate_design.md` Phase 7 -- supersedes Phase 4's
`benchmark_continuous_placeholder`, a test-loss-only stub, now removed):
the same shape of report as steps 1-4 above, but for
`ContinuousThrombusSurrogate` and its point-cloud data --

1. `ContinuousThrombusSurrogate` + both continuous baselines
   (`neural.baselines.Continuous*`, fit on the train split) evaluated via
   `benchmark.metrics.field_rmse_pointwise` (the primary number: exact
   mesh-node ground truth, no rasterization) on `test`/`edge_holdout`.
2. If `--grid-checkpoint` is also given (a separately-trained
   `ThrombusSurrogate`, on a dataset directory generated with
   `--also-save-raster` so `ThrombusSurrogateDataset` also has raster data
   to read): its legacy grid `field_rmse`, in its own clearly-separate
   report section -- **not directly comparable in magnitude** to the
   point-query numbers above (different evaluation domains: interpolated
   raster vs. exact mesh nodes; see that section's caveat in the report).
3. `edge_holdout_eval.evaluate_edge_holdout_degradation_continuous`,
   `calibration.py` UQ calibration (unchanged -- both wrappers work with
   `ContinuousThrombusSurrogate`'s multi-argument `forward()` without
   modification, confirmed in `tests/test_uncertainty.py`), and a runtime
   comparison, all for the continuous model.
4. The opt-in genuine-extrapolation split is NOT wired in here, matching
   `run_benchmark`'s own grid-path precedent (`scripts/
   evaluate_extrapolation.py` is a separate script, not part of the main
   report) -- see `scripts/evaluate_extrapolation_continuous.py`.

Report written to `results/report_continuous.md` (a different filename
from the grid path's `report.md`, since they cover different checkpoints/
models and running both against the same `--output-dir` shouldn't clobber
either).
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
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from ..data.dataset import FIELD_NAMES, PointCloudThrombusDataset, ThrombusSurrogateDataset, log_to_field, pointcloud_collate_fn
from ..mechanistic.coupled_solver import run_coupled_simulation
from ..mechanistic.mesh import GeometryConfig, build_aneurysm_mesh
from ..neural.baselines import (
    ContinuousMeanFieldBaseline,
    ContinuousNearestNeighborBaseline,
    MeanFieldBaseline,
    NearestNeighborBaseline,
)
from ..neural.coordinate_decoder import ContinuousThrombusSurrogate
from ..neural.model import ThrombusSurrogate
from ..neural.train import DEFAULT_EXCLUDED_TEMPORAL_CHANNELS
from ..neural.uncertainty import DeepEnsemble, MCDropoutWrapper
from . import calibration as calibration_mod
from . import edge_holdout_eval as edge_holdout_eval_mod
from .metrics import field_rmse, field_rmse_pointwise, runtime_comparison, thrombus_iou, thrombus_mask
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


class _FieldChannelsOnly(nn.Module):
    """Wraps a model so `forward()` only ever returns its first
    `n_field_channels` channels -- a no-op when the model wasn't built
    with `predict_M_at_wall` (its output already has exactly that many
    channels), but necessary when it was: `edge_holdout_eval.py`,
    `calibration.py`, and the accuracy/baseline-comparison metrics below
    all compare a model's prediction directly against `batch["fields"]`
    (always `len(FIELD_NAMES)` channels) and have no notion of an extra
    `M_at_wall`/`M_at_target` channel -- passing them the raw 12-channel
    prediction would be a shape mismatch. `.eval()`/`.modules()` (used by
    `neural.uncertainty._enable_mc_dropout`) still reach the wrapped
    model's submodules normally, since `self.model` is a regular
    registered submodule.

    `forward(*args)`, not `forward(params)`: works for both
    `ThrombusSurrogate` (single `params` tensor) and
    `ContinuousThrombusSurrogate` (`params_with_time, query_points_m,
    batch_index, geometry_mm`) -- this wrapper only needs to slice the
    *output*, regardless of how many arguments produced it."""

    def __init__(self, model: nn.Module, n_field_channels: int):
        super().__init__()
        self.model = model
        self.n_field_channels = n_field_channels

    def forward(self, *args) -> torch.Tensor:
        return self.model(*args)[:, : self.n_field_channels]


def _build_uncertainty_wrapper(
    model: nn.Module, model_cfg: dict, uncertainty_cfg: dict, n_field_channels: int, model_cls: type = ThrombusSurrogate
):
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

    `model` (for `mc_dropout`) is expected to already be field-channels-only
    (see `_FieldChannelsOnly`) if `model_cfg` has `predict_M_at_wall` set;
    freshly-constructed `deep_ensemble` members are wrapped here directly
    since `model_factory` builds them itself. `model_cls` (default
    `ThrombusSurrogate`) is the class `model_factory` constructs -- pass
    `ContinuousThrombusSurrogate` for the continuous path (Phase 7); both
    take the same single `cfg` positional argument, so no other change is
    needed here, matching `neural.uncertainty`'s own `*args`-generic design.
    """

    method = uncertainty_cfg["method"]
    predict_M_at_wall = bool(model_cfg.get("predict_M_at_wall", False))
    if method == "deep_ensemble":
        def model_factory():
            member = model_cls(model_cfg).eval()
            return _FieldChannelsOnly(member, n_field_channels) if predict_M_at_wall else member

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
    predict_M_at_wall = bool(checkpoint["cfg"].get("predict_M_at_wall", False))
    n_field_channels = checkpoint["cfg"]["output_channels"]
    # See _FieldChannelsOnly: everything below except the M_at_wall/IoU step
    # itself only knows about the original n_field_channels physical fields.
    field_model = _FieldChannelsOnly(model, n_field_channels) if predict_M_at_wall else model

    train_ds = ThrombusSurrogateDataset(dataset_dir, "train")
    test_ds = ThrombusSurrogateDataset(dataset_dir, "test")
    edge_holdout_ds = ThrombusSurrogateDataset(dataset_dir, "edge_holdout")

    test_loader = DataLoader(test_ds, batch_size=len(test_ds))
    test_batch = next(iter(test_loader))
    edge_holdout_loader = DataLoader(edge_holdout_ds, batch_size=len(edge_holdout_ds))
    edge_holdout_batch = next(iter(edge_holdout_loader))

    with torch.no_grad():
        t0 = time.perf_counter()
        raw_pred = model(test_batch["params"])
        neural_time_per_sample = (time.perf_counter() - t0) / len(test_ds)
    pred_fields = raw_pred[:, :n_field_channels]

    accuracy = field_rmse(pred_fields.numpy(), test_batch["fields"].numpy(), mask=test_batch["fluid_mask"].numpy())

    # Thrombosed-region mask IoU (Task 5.1's M_at_wall channel): only
    # available if this checkpoint was trained with predict_M_at_wall.
    thrombus_overlap = None
    if predict_M_at_wall:
        M_at_critical = physio_base["adhesion_aggregation"]["M_at_critical_plt_cm2"]
        fibrin_critical = physio_base["fibrin"]["fibrin_critical_uM"]
        fi_idx = FIELD_NAMES.index("conc_FI")

        pred_M_at_wall = log_to_field(raw_pred[:, n_field_channels]).numpy()
        true_M_at_wall = log_to_field(test_batch["M_at_wall"]).numpy()
        pred_FI = log_to_field(pred_fields[:, fi_idx]).numpy()
        true_FI = log_to_field(test_batch["fields"][:, fi_idx]).numpy()

        pred_thrombus_mask = thrombus_mask(pred_M_at_wall, pred_FI, M_at_critical, fibrin_critical)
        true_thrombus_mask = thrombus_mask(true_M_at_wall, true_FI, M_at_critical, fibrin_critical)
        thrombus_overlap = {
            "iou": thrombus_iou(pred_thrombus_mask, true_thrombus_mask),
            "pred_thrombosed_fraction": float(pred_thrombus_mask.mean()),
            "true_thrombosed_fraction": float(true_thrombus_mask.mean()),
        }

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
            "edge_holdout": _field_rmse_for(field_model, edge_holdout_batch),
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

    edge_holdout_degradation = edge_holdout_eval_mod.evaluate_edge_holdout_degradation(field_model, test_ds, edge_holdout_ds)

    uq_model = _build_uncertainty_wrapper(field_model, checkpoint["cfg"], training_cfg["model"]["uncertainty"], n_field_channels)
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
        output_dir, accuracy, runtime, edge_holdout_degradation, ece, model_comparison, thrombus_overlap,
        n_test=len(test_ds), n_edge_holdout=len(edge_holdout_ds),
    )


def _write_report(
    output_dir: str, accuracy: dict, runtime: dict, edge_holdout_degradation: dict, ece: float,
    model_comparison: dict, thrombus_overlap: dict | None,
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
        "## Thrombosed-region overlap (IoU)",
        "",
    ]
    if thrombus_overlap is None:
        lines += [
            "Not computed for this checkpoint -- requires `model.predict_M_at_wall: true` "
            "at training time (see `neural/model.py`'s docstring). Retrain with that flag "
            "set to get this section.",
            "",
        ]
    else:
        lines += [
            "| Metric | Value |",
            "|---|---|",
            f"| IoU (predicted vs. reference thrombosed mask) | {thrombus_overlap['iou']:.4f} |",
            f"| Predicted thrombosed fraction (of raster cells) | {thrombus_overlap['pred_thrombosed_fraction']:.4f} |",
            f"| Reference thrombosed fraction (of raster cells) | {thrombus_overlap['true_thrombosed_fraction']:.4f} |",
            "",
            "Thrombosed-region mask per `benchmark/metrics.thrombus_mask` (paper Sec. 2.6: "
            "`M_at >= M_at_critical` OR `[FI] >= fibrin_critical`, evaluated per raster cell "
            "using the predicted/reference `M_at_wall` and `conc_FI` fields). `[FI]` is "
            'unreliable whenever a sample hit the concentration-cap safety clip (see README.md '
            '"Known limitations"); if every test-split sample did, this IoU is effectively '
            "driven by `M_at_wall` alone, not a genuine test of the fibrin threshold term.",
            "",
        ]
    lines += [
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


def _evaluate_continuous_pointwise(model: nn.Module, dataset: PointCloudThrombusDataset) -> dict:
    """`benchmark.metrics.field_rmse_pointwise` over an entire
    `PointCloudThrombusDataset` split (every checkpoint, every mesh node --
    not a training-time `points_per_sample` subsample; a benchmark should
    use all available ground truth). Works for `ContinuousThrombusSurrogate`
    and both `neural.baselines.Continuous*` baselines (same call
    signature). Always slices to `len(FIELD_NAMES)` output channels first
    -- a no-op unless the model has an extra `predict_M_at_wall` channel
    (in which case the caller should already have wrapped it in
    `_FieldChannelsOnly`, but this slice is harmless either way)."""

    loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=pointcloud_collate_fn)
    batch = next(iter(loader))
    with torch.no_grad():
        pred = model(batch["params_with_time"], batch["node_coords"], batch["batch_index"], batch["geometry_mm"])
    pred_fields = pred[:, : len(FIELD_NAMES)]
    return field_rmse_pointwise(pred_fields.numpy(), batch["fields"].numpy())


def run_benchmark_continuous(
    training_cfg: dict,
    physio_base: dict,
    mesh_cfg: dict,
    checkpoint_path: str,
    dataset_dir: str,
    output_dir: str = "results",
    grid_checkpoint_path: str | None = None,
    n_runtime_samples: int = 2,
    mechanistic_end_time_s: float = 0.4,
    mechanistic_dt_s: float = 0.1,
) -> None:
    """Full continuous-path benchmark report -- see module docstring's
    `--continuous` section. Supersedes Phase 4's
    `benchmark_continuous_placeholder` (removed)."""

    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_cfg = checkpoint["cfg"]
    model = ContinuousThrombusSurrogate(model_cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predict_M_at_wall = bool(model_cfg.get("predict_M_at_wall", False))
    n_field_channels = len(FIELD_NAMES)
    # See _FieldChannelsOnly: everything below except the M_at/IoU step
    # itself only knows about the original n_field_channels physical fields.
    field_model = _FieldChannelsOnly(model, n_field_channels) if predict_M_at_wall else model

    data_cfg = training_cfg["data"]
    train_ds = PointCloudThrombusDataset(dataset_dir, "train")
    test_ds = PointCloudThrombusDataset(dataset_dir, "test")
    edge_holdout_ds = PointCloudThrombusDataset(dataset_dir, "edge_holdout")

    test_loader = DataLoader(test_ds, batch_size=len(test_ds), collate_fn=pointcloud_collate_fn)
    test_batch = next(iter(test_loader))

    with torch.no_grad():
        t0 = time.perf_counter()
        raw_pred = model(
            test_batch["params_with_time"], test_batch["node_coords"], test_batch["batch_index"], test_batch["geometry_mm"]
        )
        # Per (sample, checkpoint) item, not per query point -- matches
        # PointCloudThrombusDataset's own item granularity (one row of
        # params_with_time per item), the continuous analogue of the grid
        # path's "per sample" runtime unit.
        neural_time_per_item = (time.perf_counter() - t0) / len(test_ds)
    pred_fields = raw_pred[:, :n_field_channels]

    accuracy = field_rmse_pointwise(pred_fields.numpy(), test_batch["fields"].numpy())

    # Thrombosed-region mask IoU: only available if this checkpoint was
    # trained with predict_M_at_wall -- mirrors run_benchmark's grid-path
    # section, using M_at_target (0 off-wall, Phase 4) instead of the
    # rasterized M_at_wall band.
    thrombus_overlap = None
    if predict_M_at_wall:
        M_at_critical = physio_base["adhesion_aggregation"]["M_at_critical_plt_cm2"]
        fibrin_critical = physio_base["fibrin"]["fibrin_critical_uM"]
        fi_idx = FIELD_NAMES.index("conc_FI")

        pred_M_at = log_to_field(raw_pred[:, n_field_channels]).numpy()
        true_M_at = log_to_field(test_batch["M_at_target"]).numpy()
        pred_FI = log_to_field(pred_fields[:, fi_idx]).numpy()
        true_FI = log_to_field(test_batch["fields"][:, fi_idx]).numpy()

        pred_thrombus_mask = thrombus_mask(pred_M_at, pred_FI, M_at_critical, fibrin_critical)
        true_thrombus_mask = thrombus_mask(true_M_at, true_FI, M_at_critical, fibrin_critical)
        thrombus_overlap = {
            "iou": thrombus_iou(pred_thrombus_mask, true_thrombus_mask),
            "pred_thrombosed_fraction": float(pred_thrombus_mask.mean()),
            "true_thrombosed_fraction": float(true_thrombus_mask.mean()),
        }

    # Baseline comparison (Phase 6): fit on train, evaluated the same way
    # as the continuous model -- without this, there's no way to tell
    # whether ContinuousThrombusSurrogate is adding value over something
    # trivial.
    mean_field_baseline = ContinuousMeanFieldBaseline().fit(train_ds)
    nearest_neighbor_baseline = ContinuousNearestNeighborBaseline().fit(train_ds)

    model_comparison = {
        "Continuous mean-field baseline": {
            "test": _evaluate_continuous_pointwise(mean_field_baseline, test_ds),
            "edge_holdout": _evaluate_continuous_pointwise(mean_field_baseline, edge_holdout_ds),
        },
        "Continuous nearest-neighbor baseline": {
            "test": _evaluate_continuous_pointwise(nearest_neighbor_baseline, test_ds),
            "edge_holdout": _evaluate_continuous_pointwise(nearest_neighbor_baseline, edge_holdout_ds),
        },
        "ContinuousThrombusSurrogate": {
            "test": accuracy,
            "edge_holdout": _evaluate_continuous_pointwise(field_model, edge_holdout_ds),
        },
    }

    # Legacy grid-projection FNO, kept alongside for comparison per the
    # design summary -- opt-in (a separately-trained checkpoint, and
    # dataset_dir must have been generated with --also-save-raster so
    # ThrombusSurrogateDataset has raster keys to read). Its numbers use
    # field_rmse (grid RMSE), NOT field_rmse_pointwise -- see module
    # docstring: the two are evaluated on different ground truth
    # (interpolated raster vs. exact mesh nodes) and are not directly
    # comparable in magnitude.
    grid_comparison = None
    if grid_checkpoint_path is not None:
        grid_checkpoint = torch.load(grid_checkpoint_path, weights_only=False)
        grid_model = ThrombusSurrogate(grid_checkpoint["cfg"])
        grid_model.load_state_dict(grid_checkpoint["model_state"])
        grid_model.eval()
        grid_predict_M_at_wall = bool(grid_checkpoint["cfg"].get("predict_M_at_wall", False))
        grid_n_field_channels = grid_checkpoint["cfg"]["output_channels"]
        grid_field_model = (
            _FieldChannelsOnly(grid_model, grid_n_field_channels) if grid_predict_M_at_wall else grid_model
        )

        grid_test_ds = ThrombusSurrogateDataset(dataset_dir, "test")
        grid_edge_holdout_ds = ThrombusSurrogateDataset(dataset_dir, "edge_holdout")

        def _grid_field_rmse_for(batch: dict) -> dict:
            with torch.no_grad():
                pred = grid_field_model(batch["params"])
            return field_rmse(pred.numpy(), batch["fields"].numpy(), mask=batch["fluid_mask"].numpy())

        grid_test_batch = next(iter(DataLoader(grid_test_ds, batch_size=len(grid_test_ds))))
        grid_edge_holdout_batch = next(iter(DataLoader(grid_edge_holdout_ds, batch_size=len(grid_edge_holdout_ds))))
        grid_comparison = {
            "test": _grid_field_rmse_for(grid_test_batch),
            "edge_holdout": _grid_field_rmse_for(grid_edge_holdout_batch),
            "n_test": len(grid_test_ds),
            "n_edge_holdout": len(grid_edge_holdout_ds),
        }

    # Runtime comparison: re-time the mechanistic solver on a few test-set
    # parameter vectors vs. the neural forward pass just timed above.
    # params_with_time's first 8 entries are the normalized PARAM_ORDER
    # scalars (the 9th is normalized time, not a mechanistic-solver input).
    n_runtime = min(n_runtime_samples, len(test_ds))
    param_space = ParameterSpace()
    mech_times = np.array(
        [
            _time_mechanistic_solve(
                denormalize_params(test_batch["params_with_time"][i, :8].numpy(), param_space),
                physio_base, mesh_cfg, mechanistic_end_time_s, mechanistic_dt_s,
            )
            for i in range(n_runtime)
        ]
    )
    neural_times = np.full(n_runtime, neural_time_per_item)
    runtime = runtime_comparison(mech_times, neural_times)

    edge_holdout_degradation = edge_holdout_eval_mod.evaluate_edge_holdout_degradation_continuous(
        field_model, test_ds, edge_holdout_ds
    )

    uq_model = _build_uncertainty_wrapper(
        field_model, model_cfg, training_cfg["model"]["uncertainty"], n_field_channels,
        model_cls=ContinuousThrombusSurrogate,
    )
    with torch.no_grad():
        pred_mean, pred_var = uq_model.predict(
            test_batch["params_with_time"], test_batch["node_coords"], test_batch["batch_index"], test_batch["geometry_mm"]
        )
    reliability = calibration_mod.reliability_diagram_data(
        pred_mean.numpy(), pred_var.numpy(), test_batch["fields"].numpy(), n_bins=5
    )
    ece = calibration_mod.expected_calibration_error(
        pred_mean.numpy(), pred_var.numpy(), test_batch["fields"].numpy(), n_bins=5
    )

    fig, ax = plt.subplots()
    plots_mod.plot_edge_holdout_degradation(edge_holdout_degradation, ax=ax)
    fig.savefig(os.path.join(output_dir, "edge_holdout_degradation_continuous.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plots_mod.plot_calibration(reliability, ax=ax)
    fig.savefig(os.path.join(output_dir, "calibration_continuous.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    _write_report_continuous(
        output_dir, accuracy, runtime, edge_holdout_degradation, ece, model_comparison, grid_comparison,
        thrombus_overlap, n_test=len(test_ds), n_edge_holdout=len(edge_holdout_ds),
        excluded_temporal_channels=data_cfg.get("excluded_temporal_channels", list(DEFAULT_EXCLUDED_TEMPORAL_CHANNELS)),
    )


def _write_report_continuous(
    output_dir: str,
    accuracy: dict,
    runtime: dict,
    edge_holdout_degradation: dict,
    ece: float,
    model_comparison: dict,
    grid_comparison: dict | None,
    thrombus_overlap: dict | None,
    n_test: int,
    n_edge_holdout: int,
    excluded_temporal_channels: list[str],
) -> None:
    comparison_rows = [
        f"| {name} | {m['test']['overall']:.4f} | {m['edge_holdout']['overall']:.4f} |"
        for name, m in model_comparison.items()
    ]

    lines = [
        "# thrombus-bench continuous-surrogate benchmark report",
        "",
        f"Test set: {n_test} (sample, checkpoint) items. Edge holdout set: {n_edge_holdout} (sample, "
        "checkpoint) items -- each item is one checkpoint of one held-out simulation; a single "
        "simulation with multiple saved checkpoints (`data.n_snapshots`) contributes multiple items.",
        "",
        "## Accuracy (point-query RMSE, primary metric)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Overall field RMSE (log-space, exact mesh nodes) | {accuracy['overall']:.4f} |",
        "",
        "Computed by `benchmark.metrics.field_rmse_pointwise` directly against each held-out "
        "simulation's own mesh node coordinates -- no rasterization/interpolation involved (unlike "
        "the legacy grid path below), so this is the ground truth at exactly the points it's "
        "evaluated at.",
        "",
        "## Model comparison (point-query field RMSE, log-space)",
        "",
        "| Model | Test | Edge holdout |",
        "|---|---|---|",
        *comparison_rows,
        "",
        "`Continuous mean-field baseline`/`Continuous nearest-neighbor baseline` "
        "(`neural/baselines.py`) are trivial point-query predictors fit on the train split only "
        "-- see that module's docstring for how each is defined -- included so "
        "`ContinuousThrombusSurrogate`'s numbers above can be judged against a floor rather than "
        "in isolation.",
        "",
        f"Species excluded from this checkpoint's training loss by default due to known "
        f"concentration-cap unreliability (`data.excluded_temporal_channels`, see README.md "
        f'"Known limitations"): {", ".join(excluded_temporal_channels) if excluded_temporal_channels else "none"}. '
        "Their RMSE is still included in the overall number above (not masked out here) -- treat it "
        "as a diagnostic, not a fully trustworthy accuracy figure for those channels specifically.",
        "",
    ]

    if grid_comparison is not None:
        lines += [
            "## Legacy grid-projection FNO (comparison baseline, different evaluation domain)",
            "",
            "**Not directly comparable in magnitude to the point-query numbers above** -- this model "
            "is evaluated on a rasterized grid (`benchmark.metrics.field_rmse`, "
            "`griddata`-interpolated onto a fixed grid, `data/generate_dataset._rasterize`), not "
            "exact mesh nodes, so differences in these numbers reflect the evaluation domain as "
            "much as (or more than) genuine model accuracy differences.",
            "",
            f"Test set: {grid_comparison['n_test']} samples. Edge holdout set: "
            f"{grid_comparison['n_edge_holdout']} samples (one item per sample -- the grid path only "
            "ever saves the final checkpoint).",
            "",
            "| Metric | Test (all cells) | Test (fluid only) | Edge holdout (all cells) | Edge holdout (fluid only) |",
            "|---|---|---|---|---|",
            f"| FNO surrogate (grid RMSE) | {grid_comparison['test']['overall']:.4f} | "
            f"{grid_comparison['test']['fluid_only']:.4f} | {grid_comparison['edge_holdout']['overall']:.4f} | "
            f"{grid_comparison['edge_holdout']['fluid_only']:.4f} |",
            "",
        ]
    else:
        lines += [
            "## Legacy grid-projection FNO",
            "",
            "Not included in this report -- pass `--grid-checkpoint` (a separately-trained "
            "`ThrombusSurrogate`) to compare against it. The dataset directory must also have been "
            "generated with `--also-save-raster` for its `ThrombusSurrogateDataset` raster data to "
            "exist.",
            "",
        ]

    lines += [
        "## Runtime",
        "",
        "| Model | Mean time (s) |",
        "|---|---|",
        f"| Mechanistic (per simulation) | {runtime['mechanistic_mean_s']:.3f} |",
        f"| Neural (per (sample, checkpoint) item) | {runtime['neural_mean_s']:.6f} |",
        f"| Speedup | {runtime['speedup_factor']:.1f}x |",
        "",
        "## Edge-of-domain holdout degradation",
        "",
        "![Edge holdout degradation](edge_holdout_degradation_continuous.png)",
        "",
        f"Core-range (test) RMSE: {edge_holdout_degradation['test']['overall']:.4f}. "
        f"Edge-holdout RMSE: {edge_holdout_degradation['edge_holdout']['overall']:.4f}. "
        f"Degradation ratio: {edge_holdout_degradation['degradation_ratio']:.2f}x.",
        "",
        "## Uncertainty calibration",
        "",
        "![Calibration](calibration_continuous.png)",
        "",
        f"Expected calibration error (predicted variance vs. observed squared error): {ece:.4f}.",
        "",
        "## Thrombosed-region overlap (IoU)",
        "",
    ]
    if thrombus_overlap is None:
        lines += [
            "Not computed for this checkpoint -- requires `model.predict_M_at_wall: true` at "
            "training time. Retrain with that flag set to get this section.",
            "",
        ]
    else:
        lines += [
            "| Metric | Value |",
            "|---|---|",
            f"| IoU (predicted vs. reference thrombosed mask) | {thrombus_overlap['iou']:.4f} |",
            f"| Predicted thrombosed fraction (of query points) | {thrombus_overlap['pred_thrombosed_fraction']:.4f} |",
            f"| Reference thrombosed fraction (of query points) | {thrombus_overlap['true_thrombosed_fraction']:.4f} |",
            "",
            "Thrombosed-region mask per `benchmark/metrics.thrombus_mask` (paper Sec. 2.6: "
            "`M_at >= M_at_critical` OR `[FI] >= fibrin_critical`), evaluated per query point using "
            "the predicted/reference `M_at`/`conc_FI` values -- most points are far from the wall, "
            "where both predicted and reference M_at are ~0 by construction (see Phase 4's "
            '"Finalized M_at design choice"), so this IoU is necessarily dominated by the (sparser) '
            "near-wall points. `[FI]` is unreliable whenever a sample hit the concentration-cap "
            'safety clip (see README.md "Known limitations").',
            "",
        ]
    lines += [
        "---",
        "",
        "*Generated by `thrombus-benchmark --continuous` (`benchmark/run_benchmark.py`, "
        "`run_benchmark_continuous`). Field RMSE is computed in the log-compressed space used for "
        "training (`data/dataset.field_to_log`), not physical units. This is a reduced-scale "
        'demonstration run; see README.md "Project status" for dataset/model scale caveats.*',
    ]
    with open(os.path.join(output_dir, "report_continuous.md"), "w") as f:
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
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run run_benchmark_continuous (ContinuousThrombusSurrogate + both continuous baselines, "
        "point-query RMSE) instead of the grid-path run_benchmark -- see that function's docstring.",
    )
    parser.add_argument(
        "--grid-checkpoint", type=str, default=None,
        help="Only used with --continuous: a separately-trained ThrombusSurrogate checkpoint to include "
        "as a 4th, legacy grid-RMSE comparison row -- --dataset-dir must have been generated with "
        "--also-save-raster for this to work. Omit to skip the grid-FNO comparison section.",
    )
    args = parser.parse_args()

    with open(args.training_config) as f:
        training_cfg = yaml.safe_load(f)
    with open(args.physio_config) as f:
        physio_base = yaml.safe_load(f)
    with open(args.geometry_config) as f:
        geometry_yaml = yaml.safe_load(f)
    mesh_cfg = dict(geometry_yaml["mesh"])
    mesh_cfg["target_num_elements"] = args.target_num_elements

    if args.continuous:
        run_benchmark_continuous(
            training_cfg, physio_base, mesh_cfg, args.checkpoint, args.dataset_dir, args.output_dir,
            grid_checkpoint_path=args.grid_checkpoint,
        )
        print(f"Wrote continuous benchmark report to {os.path.join(args.output_dir, 'report_continuous.md')}")
        return

    run_benchmark(training_cfg, physio_base, mesh_cfg, args.checkpoint, args.dataset_dir, args.output_dir)
    print(f"Wrote benchmark report to {os.path.join(args.output_dir, 'report.md')}")


if __name__ == "__main__":
    main()
