"""End-to-end tests for `benchmark.run_benchmark.run_benchmark_continuous`
(Phase 7, `docs/continuous_surrogate_design.md`): real (small) generated
data -> real (briefly) trained checkpoints -> the actual report-writing
function, both with and without the optional grid-FNO comparison row."""

from __future__ import annotations

import yaml

from thrombus_bench.benchmark.run_benchmark import run_benchmark_continuous
from thrombus_bench.data.generate_dataset import generate_dataset
from thrombus_bench.neural.train import train, train_continuous

PHYSIO_PATH = "configs/physio_params.yaml"
GEOMETRY_PATH = "configs/geometry.yaml"


def _tiny_continuous_training_cfg() -> dict:
    return {
        "seed": 0,
        "data": {"points_per_sample": None, "excluded_temporal_channels": ["conc_T", "conc_PT", "conc_FI"]},
        "model": {
            "encoder": {"param_dim": 9, "latent_grid_size": (8, 8), "hidden_channels": 8, "n_layers": 1},
            "operator_core": {"type": "fno", "fno": {"modes": 2, "hidden_channels": 8, "n_layers": 1}},
            "coordinate_encoding": {"num_frequency_bands": 4},
            "coordinate_decoder": {"mlp_hidden": 16, "n_residual_blocks": 1},
            "output_channels": 11,
            "predict_M_at_wall": True,
            "uncertainty": {"method": "mc_dropout", "mc_dropout_rate": 0.1, "mc_dropout_n_samples": 4},
        },
        "optim": {
            "epochs": 2, "batch_size": 2, "lr": 5.0e-3, "weight_decay": 0.0,
            "grad_clip_norm": 10.0, "early_stopping_patience": 1000,
        },
    }


def _tiny_grid_training_cfg() -> dict:
    return {
        "seed": 0,
        "model": {
            "encoder": {"param_dim": 8, "latent_grid_size": (16, 16), "hidden_channels": 8, "n_layers": 1},
            "operator_core": {"type": "fno", "fno": {"modes": 2, "hidden_channels": 8, "n_layers": 1}},
            "output_channels": 11,
            "predict_M_at_wall": False,
            "uncertainty": {"mc_dropout_rate": 0.1},
        },
        "physics_loss": {"enabled": False, "residual_mode": "finite_difference", "weights": {"data": 1.0}},
        "optim": {
            "epochs": 2, "batch_size": 2, "lr": 5.0e-3, "weight_decay": 0.0,
            "grad_clip_norm": 10.0, "early_stopping_patience": 1000,
        },
    }


def _generate_tiny_dataset(tmp_path, also_save_raster: bool) -> str:
    with open(PHYSIO_PATH) as f:
        physio_base = yaml.safe_load(f)
    with open(GEOMETRY_PATH) as f:
        geometry_yaml = yaml.safe_load(f)
    mesh_cfg = dict(geometry_yaml["mesh"])
    mesh_cfg["target_num_elements"] = 150

    config = {"n_train": 2, "n_val": 1, "n_test": 1, "n_edge_holdout": 1, "edge_holdout_quantile": 0.9, "seed": 0}
    output_dir = str(tmp_path / "data")
    grid_size = (16, 16) if also_save_raster else (8, 8)
    generate_dataset(
        config, physio_base, mesh_cfg, output_dir, end_time_s=0.2, dt_s=0.1, grid_size=grid_size,
        n_snapshots=2, also_save_raster=also_save_raster,
    )
    return output_dir


def test_run_benchmark_continuous_without_grid_checkpoint(tmp_path):
    dataset_dir = _generate_tiny_dataset(tmp_path, also_save_raster=False)
    training_cfg = _tiny_continuous_training_cfg()
    checkpoint_path = str(tmp_path / "model.pt")
    log_path = str(tmp_path / "log.csv")

    train_continuous(training_cfg, dataset_dir, checkpoint_path, log_path)

    with open(PHYSIO_PATH) as f:
        physio_base = yaml.safe_load(f)
    with open(GEOMETRY_PATH) as f:
        geometry_yaml = yaml.safe_load(f)
    mesh_cfg = dict(geometry_yaml["mesh"])
    mesh_cfg["target_num_elements"] = 150

    output_dir = str(tmp_path / "results")
    run_benchmark_continuous(training_cfg, physio_base, mesh_cfg, checkpoint_path, dataset_dir, output_dir)

    report_path = tmp_path / "results" / "report_continuous.md"
    assert report_path.exists()
    report = report_path.read_text()
    assert "ContinuousThrombusSurrogate" in report
    assert "Continuous mean-field baseline" in report
    assert "Continuous nearest-neighbor baseline" in report
    assert "Not included in this report" in report  # grid FNO section, since no --grid-checkpoint
    assert "Thrombosed-region overlap" in report
    assert (tmp_path / "results" / "edge_holdout_degradation_continuous.png").exists()
    assert (tmp_path / "results" / "calibration_continuous.png").exists()


def test_run_benchmark_continuous_with_grid_checkpoint(tmp_path):
    dataset_dir = _generate_tiny_dataset(tmp_path, also_save_raster=True)

    continuous_cfg = _tiny_continuous_training_cfg()
    continuous_checkpoint = str(tmp_path / "continuous_model.pt")
    train_continuous(continuous_cfg, dataset_dir, continuous_checkpoint, str(tmp_path / "continuous_log.csv"))

    grid_cfg = _tiny_grid_training_cfg()
    grid_checkpoint = str(tmp_path / "grid_model.pt")
    train(grid_cfg, dataset_dir, grid_checkpoint, str(tmp_path / "grid_log.csv"))

    with open(PHYSIO_PATH) as f:
        physio_base = yaml.safe_load(f)
    with open(GEOMETRY_PATH) as f:
        geometry_yaml = yaml.safe_load(f)
    mesh_cfg = dict(geometry_yaml["mesh"])
    mesh_cfg["target_num_elements"] = 150

    output_dir = str(tmp_path / "results")
    run_benchmark_continuous(
        continuous_cfg, physio_base, mesh_cfg, continuous_checkpoint, dataset_dir, output_dir,
        grid_checkpoint_path=grid_checkpoint,
    )

    report = (tmp_path / "results" / "report_continuous.md").read_text()
    assert "FNO surrogate (grid RMSE)" in report
    assert "Not directly comparable in magnitude" in report
    assert "Not included in this report" not in report
