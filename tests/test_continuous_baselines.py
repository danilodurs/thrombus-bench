"""Tests for the point-query baselines added in Phase 6 (`docs/
continuous_surrogate_design.md`): `ContinuousMeanFieldBaseline` and
`ContinuousNearestNeighborBaseline` (`neural/baselines.py`). Held to the
same standard as the existing grid-baseline tests: exact, hand-verifiable
expected values, not just "runs and returns something.\""""

from __future__ import annotations

import torch

from thrombus_bench.neural.baselines import ContinuousMeanFieldBaseline, ContinuousNearestNeighborBaseline

GEOMETRY_MM = (7.0, 3.2)  # aneurysm_diameter_mm, vessel_diameter_mm


class _FakeContinuousDataset:
    """Minimal duck-typed stand-in for PointCloudThrombusDataset -- both
    baselines' fit() only need len()/__getitem__ returning a dict with
    node_coords/fields/geometry_mm (and, for the nearest-neighbor variant,
    params_with_time)."""

    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _item(node_coords: torch.Tensor, field_values: torch.Tensor, params_with_time: torch.Tensor | None = None) -> dict:
    return {
        "node_coords": node_coords,
        "fields": field_values,
        "geometry_mm": torch.tensor(GEOMETRY_MM, dtype=torch.float32),
        "params_with_time": params_with_time if params_with_time is not None else torch.zeros(9),
    }


def test_continuous_mean_field_baseline_raises_before_fit():
    baseline = ContinuousMeanFieldBaseline()
    try:
        baseline(torch.zeros(1, 9), torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), torch.zeros(1, 2))
        assert False, "expected RuntimeError before fit()"
    except RuntimeError:
        pass


def test_continuous_mean_field_baseline_nearest_single_neighbor_exact_match():
    """n_neighbors=1 degenerates to a plain nearest-point lookup -- querying
    exactly at a training point must return exactly that point's field."""

    node_coords = torch.tensor([[0.01, 0.001], [0.03, 0.002], [0.045, 0.0005]])
    fields = torch.stack(
        [torch.full((11,), 1.0), torch.full((11,), 2.0), torch.full((11,), 3.0)], dim=0
    )
    dataset = _FakeContinuousDataset([_item(node_coords, fields)])
    baseline = ContinuousMeanFieldBaseline(n_neighbors=1).fit(dataset)

    query = node_coords[1:2]  # exactly the second training point
    batch_index = torch.zeros(1, dtype=torch.long)
    geometry_mm = torch.tensor([GEOMETRY_MM])
    out = baseline(torch.zeros(1, 9), query, batch_index, geometry_mm)

    assert torch.allclose(out[0], fields[1])


def test_continuous_mean_field_baseline_ignores_params_with_time():
    node_coords = torch.tensor([[0.01, 0.001], [0.03, 0.002]])
    fields = torch.stack([torch.full((11,), 1.0), torch.full((11,), 5.0)], dim=0)
    dataset = _FakeContinuousDataset([_item(node_coords, fields)])
    baseline = ContinuousMeanFieldBaseline(n_neighbors=1).fit(dataset)

    query = node_coords[:1]
    batch_index = torch.zeros(1, dtype=torch.long)
    geometry_mm = torch.tensor([GEOMETRY_MM])

    out_zero_params = baseline(torch.zeros(1, 9), query, batch_index, geometry_mm)
    out_wild_params = baseline(torch.randn(1, 9) * 100.0, query, batch_index, geometry_mm)
    assert torch.allclose(out_zero_params, out_wild_params)


def test_continuous_mean_field_baseline_averages_exactly_k_nearest():
    """Two training points equidistant from the query, plus one far-away
    outlier; n_neighbors=2 must average exactly the two close points, not
    include the outlier."""

    node_coords = torch.tensor([[0.024, 0.0016], [0.026, 0.0016], [0.001, 0.0001]])
    fields = torch.stack(
        [torch.full((11,), 10.0), torch.full((11,), 20.0), torch.full((11,), 1000.0)], dim=0
    )
    dataset = _FakeContinuousDataset([_item(node_coords, fields)])
    baseline = ContinuousMeanFieldBaseline(n_neighbors=2).fit(dataset)

    query = torch.tensor([[0.025, 0.0016]])  # midpoint of the first two, far from the outlier
    batch_index = torch.zeros(1, dtype=torch.long)
    geometry_mm = torch.tensor([GEOMETRY_MM])
    out = baseline(torch.zeros(1, 9), query, batch_index, geometry_mm)

    expected = torch.full((11,), 15.0)  # mean(10, 20)
    assert torch.allclose(out[0], expected)


def test_continuous_nearest_neighbor_raises_before_fit():
    baseline = ContinuousNearestNeighborBaseline()
    try:
        baseline(torch.zeros(1, 9), torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), torch.zeros(1, 2))
        assert False, "expected RuntimeError before fit()"
    except RuntimeError:
        pass


def test_continuous_nearest_neighbor_reproduces_exact_training_match():
    node_coords = torch.tensor([[0.01, 0.001], [0.03, 0.002]])
    fields = torch.stack([torch.full((11,), 1.0), torch.full((11,), 2.0)], dim=0)
    params_with_time = torch.randn(9)
    dataset = _FakeContinuousDataset([_item(node_coords, fields, params_with_time=params_with_time)])
    baseline = ContinuousNearestNeighborBaseline().fit(dataset)

    query_points = node_coords  # exact match to both training rows
    batch_index = torch.zeros(2, dtype=torch.long)
    geometry_mm = torch.tensor([GEOMETRY_MM])
    params_batch = params_with_time.unsqueeze(0)

    out = baseline(params_batch, query_points, batch_index, geometry_mm)
    assert torch.allclose(out, fields)


def test_continuous_nearest_neighbor_distinguishes_by_params_at_the_same_location():
    """Two samples share the exact same node location but have different
    params_with_time and different field values there -- the combined-
    distance lookup must pick the one closer in *params*, not just
    location, since both are equally close spatially."""

    shared_coords = torch.tensor([[0.025, 0.0016]])
    params_a = torch.zeros(9)
    params_b = torch.full((9,), 5.0)
    item_a = _item(shared_coords, torch.full((1, 11), 1.0), params_with_time=params_a)
    item_b = _item(shared_coords, torch.full((1, 11), 2.0), params_with_time=params_b)
    dataset = _FakeContinuousDataset([item_a, item_b])
    baseline = ContinuousNearestNeighborBaseline().fit(dataset)

    query_params = torch.full((1, 9), 4.5)  # much closer to params_b than params_a
    batch_index = torch.zeros(1, dtype=torch.long)
    geometry_mm = torch.tensor([GEOMETRY_MM])
    out = baseline(query_params, shared_coords, batch_index, geometry_mm)

    assert torch.allclose(out[0], torch.full((11,), 2.0))
