"""Tests for neural/baselines.py's trivial accuracy baselines, which give
benchmark/run_benchmark.py a way to tell whether the FNO surrogate is
adding value over something trivial (MeanFieldBaseline: a constant;
NearestNeighborBaseline: a lookup table over the training set)."""

from __future__ import annotations

import torch

from thrombus_bench.neural.baselines import MeanFieldBaseline, NearestNeighborBaseline


class _FakeDataset:
    """Minimal duck-typed stand-in for ThrombusSurrogateDataset -- both
    baselines' fit() only need len()/__getitem__ returning a dict with
    "fields" (and, for NearestNeighborBaseline, "params")."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def test_mean_field_baseline_returns_training_mean_regardless_of_params():
    torch.manual_seed(0)
    fields = [torch.randn(11, 4, 4) for _ in range(5)]
    params = [torch.randn(8) for _ in range(5)]
    dataset = _FakeDataset([{"fields": f, "params": p} for f, p in zip(fields, params)])

    baseline = MeanFieldBaseline().fit(dataset)
    expected_mean = torch.stack(fields, dim=0).mean(dim=0)

    query_zeros = torch.zeros(3, 8)
    query_wild = torch.randn(3, 8) * 100.0  # wildly different params

    out_zeros = baseline(query_zeros)
    out_wild = baseline(query_wild)

    assert out_zeros.shape == (3, 11, 4, 4)
    assert torch.allclose(out_zeros, expected_mean.unsqueeze(0).expand(3, -1, -1, -1))
    assert torch.allclose(out_zeros, out_wild)  # params are ignored entirely


def test_mean_field_baseline_raises_before_fit():
    baseline = MeanFieldBaseline()
    try:
        baseline(torch.zeros(1, 8))
        assert False, "expected RuntimeError before fit()"
    except RuntimeError:
        pass


def test_nearest_neighbor_baseline_reproduces_exact_training_match():
    torch.manual_seed(1)
    fields = [torch.randn(11, 4, 4) for _ in range(6)]
    params = [torch.randn(8) for _ in range(6)]
    dataset = _FakeDataset([{"fields": f, "params": p} for f, p in zip(fields, params)])

    baseline = NearestNeighborBaseline().fit(dataset)

    query = torch.stack(params, dim=0)  # exact match to every training row
    out = baseline(query)

    expected = torch.stack(fields, dim=0)
    assert torch.allclose(out, expected)


def test_nearest_neighbor_baseline_picks_closest_non_exact_match():
    train_params = torch.tensor([[0.0] * 8, [1.0] * 8, [10.0] * 8])
    train_fields = torch.stack(
        [torch.full((11, 2, 2), 0.0), torch.full((11, 2, 2), 1.0), torch.full((11, 2, 2), 10.0)], dim=0
    )
    dataset = _FakeDataset(
        [{"params": train_params[i], "fields": train_fields[i]} for i in range(3)]
    )

    baseline = NearestNeighborBaseline().fit(dataset)

    query = torch.tensor([[0.9] * 8])  # closest to the [1.0]*8 training row
    out = baseline(query)

    assert torch.allclose(out[0], train_fields[1])


def test_nearest_neighbor_baseline_raises_before_fit():
    baseline = NearestNeighborBaseline()
    try:
        baseline(torch.zeros(1, 8))
        assert False, "expected RuntimeError before fit()"
    except RuntimeError:
        pass
