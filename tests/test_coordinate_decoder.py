"""Tests for `neural/coordinate_decoder.py`: `CoordinateDecoder` (Stage 2)
and `ContinuousThrombusSurrogate` (Stage 1 + Stage 2 wired together), plus
a regression check that the legacy grid-projection path
(`neural/model.ThrombusSurrogate`) is unaffected by the Stage 1/Stage 2
split introduced alongside it (see `test_neural_forward.py`/
`test_uncertainty.py` for that path's own pre-existing tests, which this
file does not duplicate)."""

from __future__ import annotations

import torch

from thrombus_bench.data.dataset import FIELD_NAMES
from thrombus_bench.neural.coordinate_decoder import ContinuousThrombusSurrogate, _grid_sample_per_sample
from thrombus_bench.neural.model import ThrombusSurrogate


def _continuous_model_cfg():
    # param_dim=9: the existing 8 (data/generate_dataset.PARAM_ORDER) +
    # normalized time, per docs/continuous_surrogate_design.md.
    return {
        "encoder": {"param_dim": 9, "latent_grid_size": (16, 16), "hidden_channels": 8, "n_layers": 2},
        "operator_core": {"type": "fno", "fno": {"modes": 4, "hidden_channels": 8, "n_layers": 2}},
        "coordinate_decoder": {"mlp_hidden": 32, "n_residual_blocks": 2},
        "output_channels": 11,
        "uncertainty": {"mc_dropout_rate": 0.1},
    }


def _legacy_model_cfg():
    return {
        "encoder": {"param_dim": 8, "latent_grid_size": (16, 16), "hidden_channels": 8, "n_layers": 2},
        "operator_core": {"type": "fno", "fno": {"modes": 4, "hidden_channels": 8, "n_layers": 2}},
        "output_channels": 11,
        "uncertainty": {"mc_dropout_rate": 0.1},
    }


def test_continuous_surrogate_output_shape_with_ragged_query_counts():
    model = ContinuousThrombusSurrogate(_continuous_model_cfg())

    batch = 3
    params = torch.randn(batch, 9)
    # aneurysm_diameter_mm, vessel_diameter_mm per sample.
    geometry_mm = torch.tensor([[7.0, 3.2], [10.0, 4.0], [8.5, 3.6]])

    # Ragged: sample 0 has 5 points, sample 1 has 1 point, sample 2 has 12.
    counts = [5, 1, 12]
    batch_index = torch.cat([torch.full((n,), b, dtype=torch.long) for b, n in enumerate(counts)])
    total_points = sum(counts)
    torch.manual_seed(0)
    query_points_m = torch.rand(total_points, 2) * torch.tensor([0.05, 0.01])

    out = model(params, query_points_m, batch_index, geometry_mm)

    assert out.shape == (total_points, len(FIELD_NAMES))
    assert out.dtype == torch.float32
    assert torch.all(torch.isfinite(out))


def test_continuous_surrogate_single_sample_all_points():
    """A degenerate but common case: batch size 1, e.g. querying every mesh
    node of a single sample at once."""

    model = ContinuousThrombusSurrogate(_continuous_model_cfg())
    params = torch.randn(1, 9)
    geometry_mm = torch.tensor([[7.0, 3.2]])
    query_points_m = torch.rand(37, 2) * torch.tensor([0.05, 0.01])
    batch_index = torch.zeros(37, dtype=torch.long)

    out = model(params, query_points_m, batch_index, geometry_mm)
    assert out.shape == (37, len(FIELD_NAMES))


def test_gradient_flows_to_all_parameters_through_grid_sample_and_sdf():
    """Smoke test: `loss.backward()` runs without error and produces
    finite gradients for every learnable parameter, including through
    `grid_sample`'s bilinear interpolation. The SDF feature itself has no
    learnable parameters and is explicitly detached (see
    `ContinuousThrombusSurrogate.forward`), so it should not itself receive
    a gradient, but must not block gradient flow to the rest of the
    network either."""

    model = ContinuousThrombusSurrogate(_continuous_model_cfg())
    batch = 2
    params = torch.randn(batch, 9, requires_grad=True)
    geometry_mm = torch.tensor([[7.0, 3.2], [10.0, 4.0]])
    counts = [4, 6]
    batch_index = torch.cat([torch.full((n,), b, dtype=torch.long) for b, n in enumerate(counts)])
    query_points_m = torch.rand(sum(counts), 2) * torch.tensor([0.05, 0.01])

    out = model(params, query_points_m, batch_index, geometry_mm)
    loss = out.pow(2).sum()
    loss.backward()

    n_params = 0
    for name, p in model.named_parameters():
        n_params += 1
        assert p.grad is not None, f"parameter {name} received no gradient"
        assert torch.all(torch.isfinite(p.grad)), f"parameter {name} has non-finite gradient"
    assert n_params > 0

    # params (the encoder input) should also receive a gradient -- confirms
    # the whole chain (encoder -> FNO backbone -> grid_sample -> decoder)
    # is differentiable end-to-end, not just the decoder's own weights.
    assert params.grad is not None
    assert torch.all(torch.isfinite(params.grad))


def test_grid_sample_per_sample_uses_each_points_own_sample_grid():
    """Two samples with distinguishable constant latent grids: a query
    point assigned to sample 0 should read sample 0's value, never
    sample 1's, regardless of point ordering in the flat/ragged batch."""

    latent_grid = torch.zeros(2, 1, 4, 4)
    latent_grid[0] = 1.0
    latent_grid[1] = -5.0

    query_points_norm = torch.tensor([[0.0, 0.0], [0.5, -0.5], [0.0, 0.0], [-0.3, 0.2]])
    batch_index = torch.tensor([1, 0, 1, 0], dtype=torch.long)

    sampled = _grid_sample_per_sample(latent_grid, query_points_norm, batch_index)
    assert sampled.shape == (4, 1)
    torch.testing.assert_close(sampled[batch_index == 0], torch.full((2, 1), 1.0))
    torch.testing.assert_close(sampled[batch_index == 1], torch.full((2, 1), -5.0))


def test_legacy_grid_projection_path_unaffected_by_stage_split():
    """Regression guard for the Phase 2 refactor: `ThrombusSurrogate`
    (backed internally by the now-shared `SurrogateBackbone` + its own
    projection head) must still behave exactly like a plain grid-projection
    model -- fixed output shape, no ragged/query-point machinery involved."""

    model = ThrombusSurrogate(_legacy_model_cfg())
    params = torch.randn(3, 8)
    out = model(params)
    assert out.shape == (3, 11, 16, 16)
    assert out.dtype == torch.float32
    assert torch.all(torch.isfinite(out))
