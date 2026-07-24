"""Tests for `generate_dataset._output_every_n_steps_for_snapshots` -- the
stride-derivation helper for the multi-checkpoint point-cloud save path,
see that function's docstring and `docs/continuous_surrogate_design.md`
Phase 1/3."""

from __future__ import annotations

import pytest

from thrombus_bench.data.generate_dataset import _output_every_n_steps_for_snapshots


@pytest.mark.parametrize("n_steps", [1, 3, 10, 100])
def test_n_snapshots_one_reproduces_todays_final_only_formula(n_steps):
    """Regression guard: every dataset generated before this function
    existed relied on `output_every_n_steps=max(1, n_steps)` (see
    `_run_one_sample`) for its single saved checkpoint."""

    assert _output_every_n_steps_for_snapshots(n_steps, n_snapshots=1) == max(1, n_steps)


def test_n_snapshots_default_zero_or_negative_also_final_only():
    assert _output_every_n_steps_for_snapshots(50, n_snapshots=0) == 50


def test_n_snapshots_produces_approximately_requested_checkpoint_count():
    """Emulates `run_coupled_simulation`'s actual (1-indexed) recording
    condition, `(step + 1) % output_every_n_steps == 0 or step == n_steps
    - 1` -- NOT `step % output_every_n_steps == 0`, which would spuriously
    always include `step == 0` regardless of stride (see the coupled_solver
    fix this phase made, and `test_coupled_solver.
    test_output_every_n_steps_equal_to_n_steps_records_exactly_one_checkpoint`)."""

    n_steps = 100
    for n_snapshots in (2, 5, 10):
        stride = _output_every_n_steps_for_snapshots(n_steps, n_snapshots)
        recorded = {step for step in range(n_steps) if (step + 1) % stride == 0}
        recorded.add(n_steps - 1)  # run_coupled_simulation always keeps the final step
        # Allow +-1 slack: rounding the stride can land one checkpoint off
        # from the exact request.
        assert abs(len(recorded) - n_snapshots) <= 1


def test_stride_is_at_least_one():
    assert _output_every_n_steps_for_snapshots(1, n_snapshots=6) >= 1
