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

from thrombus_bench.neural.uncertainty import MCDropoutWrapper, _enable_mc_dropout


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
