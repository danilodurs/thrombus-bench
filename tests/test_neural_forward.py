"""Shape/dtype/gradient checks on the neural surrogate's forward pass."""

from __future__ import annotations

import pytest
import torch

from thrombus_bench.neural.encoder import GeometryParamEncoder
from thrombus_bench.neural.model import ThrombusSurrogate
from thrombus_bench.neural.operator_core import build_operator_core


def test_encoder_output_shape():
    encoder = GeometryParamEncoder(param_dim=8, latent_grid_size=(16, 16), hidden_channels=8, n_layers=2)
    params = torch.randn(4, 8)
    out = encoder(params)
    assert out.shape == (4, 8, 16, 16)


def _model_cfg():
    return {
        "encoder": {"param_dim": 8, "latent_grid_size": (16, 16), "hidden_channels": 8, "n_layers": 2},
        "operator_core": {"type": "fno", "fno": {"modes": 4, "hidden_channels": 8, "n_layers": 2}},
        "output_channels": 11,
        "uncertainty": {"mc_dropout_rate": 0.1},
    }


def test_model_forward_output_shape_and_dtype():
    model = ThrombusSurrogate(_model_cfg())
    params = torch.randn(3, 8)
    out = model(params)
    assert out.shape == (3, 11, 16, 16)
    assert out.dtype == torch.float32


def test_model_gradient_flows_to_all_parameters():
    model = ThrombusSurrogate(_model_cfg())
    params = torch.randn(2, 8)
    out = model(params)
    out.pow(2).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"


def test_gnn_operator_core_not_implemented():
    with pytest.raises(NotImplementedError):
        build_operator_core({"type": "gnn", "gnn": {"hidden_channels": 8, "n_message_passing_steps": 2}}, out_channels=11)


def test_predict_m_at_wall_defaults_to_false_and_output_channels_unchanged():
    """Existing configs (no predict_M_at_wall key) must keep producing
    exactly output_channels channels -- the opt-in 12th channel is off by
    default so existing checkpoints/configs keep working."""

    model = ThrombusSurrogate(_model_cfg())
    assert model.predict_M_at_wall is False
    out = model(torch.randn(3, 8))
    assert out.shape == (3, 11, 16, 16)


def test_predict_m_at_wall_true_adds_one_output_channel():
    cfg = _model_cfg()
    cfg["predict_M_at_wall"] = True
    model = ThrombusSurrogate(cfg)
    assert model.predict_M_at_wall is True
    out = model(torch.randn(3, 8))
    assert out.shape == (3, 12, 16, 16)
