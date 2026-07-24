"""Tests for the autograd-based physics residual machinery added in Phase 5
(`docs/continuous_surrogate_design.md`): `mass_conservation_penalty_autograd`
(generic, model-agnostic core), `sample_collocation_points` (PINN-style
SDF-rejection sampling), and `continuous_mass_conservation_loss` (wiring to
`ContinuousThrombusSurrogate`).

Held to the same standard as `test_physics_losses.py`'s hand-verified
finite-difference tests: exact analytic ground truth, not just "runs
without crashing" or "is nonzero." In particular, a *spatially uniform*
divergence-free field (e.g. solid-body rotation, div=0 everywhere) cannot
by itself distinguish a correct per-point autograd implementation from one
that silently mixes/averages residuals across the batch -- both give zero.
`test_..._matches_analytic_divergence_pointwise_not_averaged` below uses a
spatially *varying* field specifically to catch that failure mode.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from thrombus_bench.neural.coordinate_decoder import ContinuousThrombusSurrogate, _grid_sample_per_sample
from thrombus_bench.neural.physics_losses import (
    continuous_mass_conservation_loss,
    mass_conservation_penalty_autograd,
    sample_collocation_points,
)
from thrombus_bench.mechanistic.geometry_sdf import signed_distance_to_wall
from thrombus_bench.mechanistic.mesh import GeometryConfig


def test_mass_conservation_penalty_autograd_zero_for_divergence_free_field():
    """Solid-body rotation u=-y, v=x has div(u) = du/dx + dv/dy = 0 + 0 = 0
    EXACTLY at every point (analytic ground truth, not an approximation)."""

    torch.manual_seed(0)
    points = torch.rand(50, 2) * 4.0 - 2.0  # arbitrary domain; no SDF/geometry involved in this isolated test

    def velocity_fn(p: torch.Tensor) -> torch.Tensor:
        x, y = p[:, 0], p[:, 1]
        return torch.stack([-y, x], dim=-1)

    penalty = mass_conservation_penalty_autograd(velocity_fn, points)
    assert penalty.item() == pytest.approx(0.0, abs=1e-6)


def test_mass_conservation_penalty_autograd_matches_analytic_divergence_pointwise_not_averaged():
    """u = x^2, v = y^2 has div(u) = 2x + 2y, which VARIES per point. This
    is the case that actually stresses per-point correctness: a spatially
    *uniform* residual (like the rotation field above) is identically zero
    whether computed correctly per point or incorrectly mixed/averaged
    across the batch, so it can't catch that bug. This field's residual
    must match `mean((2*x_i + 2*y_i)**2)` computed from each point's OWN
    coordinates -- a batch-mixing bug (e.g. summing across the wrong axis,
    or using a shared/averaged gradient) would produce a different,
    generally smaller number as points with opposite-signed local
    divergence spuriously cancelled.
    """

    torch.manual_seed(1)
    points = torch.rand(37, 2) * 3.0 - 1.5
    points_snapshot = points.clone()

    def velocity_fn(p: torch.Tensor) -> torch.Tensor:
        return torch.stack([p[:, 0].pow(2), p[:, 1].pow(2)], dim=-1)

    penalty = mass_conservation_penalty_autograd(velocity_fn, points)

    expected_div = 2.0 * points_snapshot[:, 0] + 2.0 * points_snapshot[:, 1]
    expected = expected_div.pow(2).mean()
    assert torch.allclose(penalty, expected, atol=1e-5)

    # Also confirm this isn't a degenerate/trivial pass: mean-of-squares
    # (correct) must differ from square-of-mean (what a bug that collapses
    # per-point residuals into one shared/averaged value before squaring
    # would produce -- equal only if every point's local divergence were
    # identical, which isn't the case for these random points).
    wrong_aggregation = expected_div.mean().pow(2)
    assert not torch.allclose(penalty, wrong_aggregation)


def test_mass_conservation_penalty_autograd_through_grid_sample_interpolation():
    """Closes the gap between the trivial closed-form cases above and the
    full model: `grid_sample`'s bilinear interpolation is exactly the
    "autograd through interpolation" risk this phase is about. A latent
    grid whose values are themselves samples of an AFFINE function of
    normalized position is reproduced *exactly* by bilinear interpolation
    (affine functions are a special case of bilinear ones -- no
    interpolation error), giving an exact, hand-computable analytic
    divergence to check the composed (grid_sample -> autograd) path
    against, not just a full learned model's opaque output.
    """

    H, W = 9, 9
    ys = torch.linspace(-1.0, 1.0, H)
    xs = torch.linspace(-1.0, 1.0, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    a, b, c, d = 2.0, 0.3, -1.5, 0.7
    u_grid = a * gx + b  # u(x, y) = a*x + b
    v_grid = c * gy + d  # v(x, y) = c*y + d
    latent_grid = torch.stack([u_grid, v_grid], dim=0).unsqueeze(0)  # (1, 2, H, W)

    torch.manual_seed(2)
    query_points_norm = torch.rand(40, 2) * 1.6 - 0.8  # well inside [-1, 1], away from edge artifacts
    batch_index = torch.zeros(40, dtype=torch.long)

    def velocity_fn(p: torch.Tensor) -> torch.Tensor:
        return _grid_sample_per_sample(latent_grid, p, batch_index)

    penalty = mass_conservation_penalty_autograd(velocity_fn, query_points_norm)
    expected = (a + c) ** 2  # constant divergence a+c everywhere, exactly
    assert penalty.item() == pytest.approx(expected, abs=1e-4)


def test_mass_conservation_penalty_autograd_does_not_mutate_input():
    points = torch.rand(5, 2)
    assert not points.requires_grad

    def velocity_fn(p: torch.Tensor) -> torch.Tensor:
        return torch.stack([-p[:, 1], p[:, 0]], dim=-1)

    mass_conservation_penalty_autograd(velocity_fn, points)
    assert not points.requires_grad, "the caller's tensor must not be mutated in place"


def test_sample_collocation_points_are_strictly_inside_fluid_domain():
    geometry_mm = torch.tensor([[7.0, 3.2], [10.0, 4.0]])
    query_points_m, batch_index = sample_collocation_points(
        geometry_mm, vessel_length_mm=50.0, n_points_per_sample=20, rng=np.random.default_rng(0)
    )

    assert query_points_m.shape[1] == 2
    assert batch_index.shape[0] == query_points_m.shape[0]
    # With oversample_factor=4 on this domain shape, expect (not strictly
    # guarantee) the full requested count per sample.
    assert int((batch_index == 0).sum()) == 20
    assert int((batch_index == 1).sum()) == 20

    for b, (aneurysm_mm, vessel_mm) in enumerate([(7.0, 3.2), (10.0, 4.0)]):
        geom = GeometryConfig(vessel_diameter_mm=vessel_mm, aneurysm_diameter_mm=aneurysm_mm, vessel_length_mm=50.0)
        pts = query_points_m[batch_index == b].numpy()
        sdf = signed_distance_to_wall(pts[:, 0], pts[:, 1], geom)
        assert np.all(sdf > 0.0)


def _tiny_continuous_cfg() -> dict:
    return {
        "encoder": {"param_dim": 9, "latent_grid_size": (8, 8), "hidden_channels": 8, "n_layers": 1},
        "operator_core": {"type": "fno", "fno": {"modes": 2, "hidden_channels": 8, "n_layers": 1}},
        "coordinate_encoding": {"num_frequency_bands": 4},
        "coordinate_decoder": {"mlp_hidden": 16, "n_residual_blocks": 1},
        "output_channels": 11,
        "uncertainty": {"mc_dropout_rate": 0.1},
    }


def test_gradient_flows_to_model_parameters_through_autograd_mass_conservation_loss():
    """Full-training-step check: `loss.backward()` on
    `continuous_mass_conservation_loss`'s output alone (no data loss)
    must populate finite gradients on model parameters throughout the
    network -- both the decoder's output layer (closest to the loss) and
    the encoder (furthest, reached only by backpropagating through the
    whole Stage 1 backbone) -- confirming `create_graph=True` actually did
    its job. Without it, every one of these would show `p.grad is None`
    despite `loss.backward()` raising no exception."""

    torch.manual_seed(0)
    model = ContinuousThrombusSurrogate(_tiny_continuous_cfg())
    batch = 2
    params_with_time = torch.randn(batch, 9)
    geometry_mm = torch.tensor([[7.0, 3.2], [10.0, 4.0]])

    loss = continuous_mass_conservation_loss(
        model, params_with_time, geometry_mm, n_points_per_sample=8,
        vessel_length_mm=50.0, rng=np.random.default_rng(0),
    )
    assert torch.isfinite(loss)
    loss.backward()

    decoder_weight = model.decoder.output_proj.weight
    encoder_weight = model.backbone.encoder.param_mlp[0].weight

    assert decoder_weight.grad is not None
    assert torch.all(torch.isfinite(decoder_weight.grad))
    assert encoder_weight.grad is not None
    assert torch.all(torch.isfinite(encoder_weight.grad))

    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    assert n_with_grad > 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad))
