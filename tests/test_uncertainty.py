"""Tests for neural/uncertainty.py's MC-dropout/deep-ensemble UQ wrappers.

Regression coverage for a bug found while testing Task 5.2's
_FieldChannelsOnly wrapper: _enable_mc_dropout checked
`isinstance(module, nn.Dropout)`, but neural/model.py's ThrombusSurrogate
uses `nn.Dropout2d`, which is NOT a subclass of `nn.Dropout` (both are
siblings under the private `_DropoutNd` base) -- so the check silently
never matched, and MCDropoutWrapper.predict's "stochastic" samples were
all identical (pred_var ~ 0) regardless of the model's true uncertainty.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from thrombus_bench.neural.coordinate_decoder import ContinuousThrombusSurrogate
from thrombus_bench.neural.uncertainty import DeepEnsemble, MCDropoutWrapper, _enable_mc_dropout


class _DropoutModel(nn.Module):
    """Mirrors ThrombusSurrogate's structure: an nn.Dropout2d between two
    linear-ish stages, high enough dropout rate that stochastic samples
    are virtually guaranteed to differ."""

    def __init__(self):
        super().__init__()
        self.pre = nn.Linear(4, 16)
        self.dropout = nn.Dropout2d(p=0.9)
        self.post = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        h = self.dropout(h.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return self.post(h)


def test_enable_mc_dropout_sets_dropout2d_to_training_mode():
    model = _DropoutModel()
    model.eval()
    assert model.dropout.training is False

    _enable_mc_dropout(model)

    assert model.dropout.training is True


def test_mc_dropout_wrapper_produces_nonzero_variance():
    """End-to-end consequence of the fix: with dropout actually active
    during sampling, repeated forward passes on identical input must
    differ, giving nonzero predictive variance."""

    torch.manual_seed(0)
    model = _DropoutModel()
    wrapper = MCDropoutWrapper(model, n_samples=20, dropout_rate=0.9)

    mean, var = wrapper.predict(torch.ones(3, 4))

    assert mean.shape == (3, 4)
    assert var.shape == (3, 4)
    assert torch.all(var > 0), "MC-dropout variance is zero -- dropout is not actually active during sampling"


def test_mc_dropout_wrapper_leaves_non_dropout_layers_in_eval_mode():
    """predict() re-enables only the dropout submodule(s) for sampling --
    every other layer should stay in eval mode throughout."""

    model = _DropoutModel()
    wrapper = MCDropoutWrapper(model, n_samples=3, dropout_rate=0.9)
    wrapper.predict(torch.ones(2, 4))

    assert model.pre.training is False
    assert model.post.training is False


def _tiny_continuous_cfg(mc_dropout_rate: float = 0.5) -> dict:
    return {
        "encoder": {"param_dim": 9, "latent_grid_size": (8, 8), "hidden_channels": 8, "n_layers": 1},
        "operator_core": {"type": "fno", "fno": {"modes": 2, "hidden_channels": 8, "n_layers": 1}},
        "coordinate_decoder": {"mlp_hidden": 16, "n_residual_blocks": 1},
        "output_channels": 11,
        "uncertainty": {"mc_dropout_rate": mc_dropout_rate},
    }


def _continuous_call_args():
    """A ragged 2-sample batch, matching ContinuousThrombusSurrogate.forward's
    (params_with_time, query_points_m, batch_index, geometry_mm) signature."""

    params_with_time = torch.randn(2, 9)
    geometry_mm = torch.tensor([[7.0, 3.2], [10.0, 4.0]])
    counts = [4, 6]
    batch_index = torch.cat([torch.full((n,), b, dtype=torch.long) for b, n in enumerate(counts)])
    query_points_m = torch.rand(sum(counts), 2) * torch.tensor([0.05, 0.0067])
    return params_with_time, query_points_m, batch_index, geometry_mm


def test_mc_dropout_wrapper_works_unmodified_with_continuous_model():
    """MCDropoutWrapper.predict is `*args, **kwargs`-generic -- confirms it
    actually works (not just "should work") against
    ContinuousThrombusSurrogate's multi-argument, ragged-batch forward
    signature, not just ThrombusSurrogate's single-tensor one."""

    torch.manual_seed(0)
    model = ContinuousThrombusSurrogate(_tiny_continuous_cfg(mc_dropout_rate=0.5))
    args = _continuous_call_args()
    total_points = args[1].shape[0]

    wrapper = MCDropoutWrapper(model, n_samples=20, dropout_rate=0.5)
    mean, var = wrapper.predict(*args)

    assert mean.shape == (total_points, 11)
    assert var.shape == (total_points, 11)
    assert torch.all(torch.isfinite(mean))
    assert torch.all(torch.isfinite(var))
    assert torch.all(var >= 0.0)
    assert var.sum() > 0.0, "MC-dropout variance is entirely zero for the continuous model"


def test_deep_ensemble_works_unmodified_with_continuous_model():
    """Same confirmation for DeepEnsemble -- and that independently
    initialized ensemble members produce meaningfully larger variance than
    a single model's dropout perturbations (a basic "is this sensible"
    sanity check, not just "is it nonzero")."""

    torch.manual_seed(1)
    cfg = _tiny_continuous_cfg(mc_dropout_rate=0.5)
    args = _continuous_call_args()
    total_points = args[1].shape[0]

    ensemble = DeepEnsemble(lambda: ContinuousThrombusSurrogate(cfg).eval(), n_members=4)
    mean, var = ensemble.predict(*args)

    assert mean.shape == (total_points, 11)
    assert var.shape == (total_points, 11)
    assert torch.all(torch.isfinite(mean))
    assert torch.all(var >= 0.0)
    assert var.sum() > 0.0, "DeepEnsemble variance is entirely zero across independently-initialized members"

    mc_wrapper = MCDropoutWrapper(ContinuousThrombusSurrogate(cfg), n_samples=20, dropout_rate=0.5)
    _, mc_var = mc_wrapper.predict(*args)
    assert var.mean().item() > mc_var.mean().item(), (
        "independently-initialized ensemble members should disagree more than dropout perturbations "
        "of a single model's weights -- if not, something is off (e.g. members aren't actually varying)"
    )
