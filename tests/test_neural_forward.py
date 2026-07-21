"""Shape/dtype checks on the neural surrogate's forward pass.

Depends on `neural/model.py`, `neural/encoder.py`, `neural/operator_core.py`,
which are currently scaffolding stubs -- marked `xfail` until implemented.
Intended coverage once implemented:

* `ThrombusSurrogate.forward` output dict has one entry per field named in
  `configs/training.yaml` `model.output_channels` (13 channels: velocity x2,
  pressure, viscosity, 9 species concentrations... exact split TBD at
  implementation time), each shaped `(batch, latent_grid_size[0],
  latent_grid_size[1])` or `(batch, 2, *latent_grid_size)` for vector fields.
* Output dtype matches input dtype (float32 by default).
* Both `operator_core.type in {"fno", "gnn"}` configurations produce
  matching output shapes for the same input.
* Gradient flows to all parameters (a basic "not detached" sanity check)
  after a single backward pass on a dummy scalar loss.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.xfail(
    reason="neural/model.py, encoder.py, operator_core.py not yet implemented", strict=False
)


def test_encoder_output_shape():
    from thrombus_bench.neural.encoder import GeometryParamEncoder

    encoder = GeometryParamEncoder(param_dim=8, latent_grid_size=(64, 64), hidden_channels=64, n_layers=3)
    params = torch.randn(4, 8)
    sdf = torch.randn(4, 1, 64, 64)
    out = encoder(params, sdf)
    assert out.shape == (4, 64, 64, 64)


def test_model_forward_output_keys_and_shapes():
    from thrombus_bench.neural.model import ThrombusSurrogate

    cfg = {
        "encoder": {"param_dim": 8, "latent_grid_size": (64, 64), "hidden_channels": 64, "n_layers": 3},
        "operator_core": {"type": "fno", "fno": {"modes": 16, "hidden_channels": 64, "n_layers": 4}},
        "output_channels": 13,
    }
    model = ThrombusSurrogate(cfg)
    params = torch.randn(2, 8)
    sdf = torch.randn(2, 1, 64, 64)
    out = model(params, sdf)
    assert isinstance(out, dict)
    for tensor in out.values():
        assert tensor.shape[0] == 2
        assert tensor.dtype == torch.float32
