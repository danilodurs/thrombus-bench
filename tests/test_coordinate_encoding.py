"""Tests for `neural/coordinate_encoding.FourierFeatureEncoding`."""

from __future__ import annotations

import torch

from thrombus_bench.neural.coordinate_encoding import DEFAULT_N_FREQUENCIES, FourierFeatureEncoding


def test_output_dim_matches_formula():
    enc = FourierFeatureEncoding(n_coords=2, n_frequencies=6)
    coords = torch.zeros(4, 2)
    out = enc(coords)
    assert out.shape == (4, 2 * 2 * 6)
    assert enc.output_dim == out.shape[-1]


def test_deterministic_across_calls_and_instances():
    coords = torch.tensor([[0.3, -0.7], [1.0, -1.0], [0.0, 0.0]])
    enc_a = FourierFeatureEncoding()
    enc_b = FourierFeatureEncoding()
    out_a1 = enc_a(coords)
    out_a2 = enc_a(coords)
    out_b = enc_b(coords)
    assert torch.equal(out_a1, out_a2)
    assert torch.equal(out_a1, out_b)


def test_nearby_points_map_to_nearby_but_distinct_encodings():
    enc = FourierFeatureEncoding()
    p0 = torch.tensor([[0.1, 0.2]])
    p_near = torch.tensor([[0.1001, 0.2001]])
    p_far = torch.tensor([[-0.9, 0.85]])

    e0 = enc(p0)
    e_near = enc(p_near)
    e_far = enc(p_far)

    d_near = torch.linalg.norm(e0 - e_near)
    d_far = torch.linalg.norm(e0 - e_far)

    assert d_near > 0.0, "encoding should distinguish two distinct nearby points"
    assert d_near < d_far, "a nearby point should stay closer in encoding space than a distant one"


def test_distant_points_do_not_collide():
    # Strictly interior grid: the domain endpoints +-1 are a documented,
    # inherent exception (see test_boundary_endpoints_collide_by_construction)
    # since every frequency is an integer multiple of pi over a period-2
    # domain -- excluded here so this test targets genuine collisions only.
    enc = FourierFeatureEncoding()
    grid = torch.linspace(-0.99, 0.99, 25)
    xs, ys = torch.meshgrid(grid, grid, indexing="ij")
    coords = torch.stack([xs.flatten(), ys.flatten()], dim=-1)
    encoded = enc(coords)

    dists = torch.cdist(encoded, encoded)
    dists.fill_diagonal_(float("inf"))
    min_dist = dists.min().item()
    assert min_dist > 1e-6, "distinct interior grid points collided in encoding space"


def test_boundary_endpoints_collide_by_construction():
    """Documented artifact: p=-1 and p=+1 are indistinguishable under this
    encoding for any L, since every frequency is an integer multiple of pi
    over the [-1, 1] domain (see module docstring "Known boundary
    artifact"). Asserted explicitly so the behavior stays visible rather
    than silently regressing either way."""

    enc = FourierFeatureEncoding(n_coords=1)
    p_minus = torch.tensor([[-1.0]])
    p_plus = torch.tensor([[1.0]])
    # float32 pi has enough rounding error at the highest frequency band
    # (2**7 * pi) that this isn't bit-exact -- atol reflects that, not a
    # loosening of the claim (still five orders of magnitude below the
    # ~1e-1 distances seen between genuinely distinct interior points).
    assert torch.allclose(enc(p_minus), enc(p_plus), atol=1e-4)


def test_default_n_frequencies_is_eight():
    assert DEFAULT_N_FREQUENCIES == 8


def test_rejects_wrong_last_dim():
    enc = FourierFeatureEncoding(n_coords=2)
    bad_coords = torch.zeros(3, 3)
    try:
        enc(bad_coords)
        assert False, "expected ValueError for mismatched last dim"
    except ValueError:
        pass
