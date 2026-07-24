"""Tests for benchmark/run_benchmark.py's Task 5.2 wiring: _FieldChannelsOnly
(so a 12-channel predict_M_at_wall model stays compatible with metrics that
only know about the original 11 physical field channels) and the
thrombosed-region IoU report section."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from thrombus_bench.benchmark.metrics import thrombus_iou, thrombus_mask
from thrombus_bench.benchmark.run_benchmark import _FieldChannelsOnly, _write_report


class _FakeModel(nn.Module):
    """Returns a fixed, distinguishable (batch, 12, H, W) tensor regardless
    of input -- channel i has every element equal to i, so slicing can be
    checked exactly."""

    def __init__(self, n_channels: int = 12, h: int = 4, w: int = 4):
        super().__init__()
        self.n_channels, self.h, self.w = n_channels, h, w

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        batch = params.shape[0]
        channel_values = torch.arange(self.n_channels, dtype=torch.float32).view(1, self.n_channels, 1, 1)
        return channel_values.expand(batch, self.n_channels, self.h, self.w).clone()


def test_field_channels_only_slices_to_requested_count():
    wrapped = _FieldChannelsOnly(_FakeModel(n_channels=12), n_field_channels=11)
    out = wrapped(torch.zeros(2, 8))

    assert out.shape == (2, 11, 4, 4)
    # Channel 11 (the M_at_wall channel) must be gone; channels 0-10 kept.
    for c in range(11):
        assert torch.all(out[:, c] == c)


def test_field_channels_only_is_noop_when_already_correct_size():
    wrapped = _FieldChannelsOnly(_FakeModel(n_channels=11), n_field_channels=11)
    out = wrapped(torch.zeros(2, 8))
    assert out.shape == (2, 11, 4, 4)


def test_field_channels_only_reaches_submodules_for_eval_and_modules():
    """neural.uncertainty._enable_mc_dropout walks model.modules() looking
    for nn.Dropout instances -- the wrapper must not hide the wrapped
    model's submodules."""

    class _ModelWithDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout2d(p=0.3)

        def forward(self, params):
            return torch.zeros(params.shape[0], 12, 4, 4)

    wrapped = _FieldChannelsOnly(_ModelWithDropout(), n_field_channels=11)
    dropouts = [m for m in wrapped.modules() if isinstance(m, nn.Dropout2d)]
    assert len(dropouts) == 1

    wrapped.eval()
    assert wrapped.model.dropout.training is False


def test_thrombus_mask_and_iou_composition_matches_manual_computation():
    """Mirrors run_benchmark.py's actual composition: combine M_at_wall +
    conc_FI into a thrombosed mask via thrombus_mask, then compare
    predicted vs. reference masks via thrombus_iou."""

    M_at_critical, fibrin_critical = 2.0e7, 0.6

    pred_M_at = np.array([[3.0e7, 0.0], [1.0e7, 1.0e7]])
    pred_FI = np.array([[0.0, 0.7], [0.0, 0.0]])
    true_M_at = np.array([[3.0e7, 0.0], [0.0, 1.0e7]])
    true_FI = np.array([[0.0, 0.7], [0.0, 0.0]])

    pred_mask = thrombus_mask(pred_M_at, pred_FI, M_at_critical, fibrin_critical)
    true_mask = thrombus_mask(true_M_at, true_FI, M_at_critical, fibrin_critical)

    # pred_mask: [[T, T], [F, F]]; true_mask: [[T, T], [F, F]] -- identical.
    assert np.array_equal(pred_mask, np.array([[True, True], [False, False]]))
    assert np.array_equal(true_mask, np.array([[True, True], [False, False]]))
    assert thrombus_iou(pred_mask, true_mask) == 1.0


def test_write_report_thrombus_overlap_none_shows_not_computed(tmp_path):
    _write_report(
        str(tmp_path),
        accuracy={"overall": 1.0, "fluid_only": 1.0},
        runtime={"mechanistic_mean_s": 1.0, "neural_mean_s": 0.01, "speedup_factor": 100.0},
        edge_holdout_degradation={"test": {"overall": 1.0}, "edge_holdout": {"overall": 1.0}, "degradation_ratio": 1.0},
        ece=0.1,
        model_comparison={"FNO surrogate": {"test": {"overall": 1.0, "fluid_only": 1.0}, "edge_holdout": {"overall": 1.0, "fluid_only": 1.0}}},
        thrombus_overlap=None,
        n_test=6, n_edge_holdout=6,
    )
    report = (tmp_path / "report.md").read_text()
    assert "Thrombosed-region overlap (IoU)" in report
    assert "predict_M_at_wall: true" in report
    assert "Not computed" in report


def test_write_report_thrombus_overlap_present_shows_iou(tmp_path):
    _write_report(
        str(tmp_path),
        accuracy={"overall": 1.0, "fluid_only": 1.0},
        runtime={"mechanistic_mean_s": 1.0, "neural_mean_s": 0.01, "speedup_factor": 100.0},
        edge_holdout_degradation={"test": {"overall": 1.0}, "edge_holdout": {"overall": 1.0}, "degradation_ratio": 1.0},
        ece=0.1,
        model_comparison={"FNO surrogate": {"test": {"overall": 1.0, "fluid_only": 1.0}, "edge_holdout": {"overall": 1.0, "fluid_only": 1.0}}},
        thrombus_overlap={"iou": 0.7321, "pred_thrombosed_fraction": 0.25, "true_thrombosed_fraction": 0.20},
        n_test=6, n_edge_holdout=6,
    )
    report = (tmp_path / "report.md").read_text()
    assert "0.7321" in report
    assert "0.2500" in report
    assert "0.2000" in report
    assert "Not computed" not in report
