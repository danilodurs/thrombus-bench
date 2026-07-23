"""Tests for the log-compressed-vs-physical-space handling in physics_losses.py.

`total_physics_loss`'s input is entirely in `data/dataset.field_to_log`'s
log-compressed space (same as the model's raw output). `mass_conservation`
must be evaluated on physical-space velocity (divergence is a linear
operator, not invariant under the nonlinear log transform); `nonnegativity`
is correct evaluated directly on log-compressed values (the transform is
sign-preserving). See `physics_losses.py`'s module docstring.
"""

from __future__ import annotations

import torch

from thrombus_bench.data.dataset import field_to_log, log_to_field
from thrombus_bench.neural.physics_losses import (
    _central_diff,
    mass_conservation_penalty,
    nonnegativity_penalty,
    total_physics_loss,
)


def test_mass_conservation_penalty_requires_physical_space_inversion():
    """A velocity ramp [0, 1, 2, 3] along x (H=1, so the dv/dy term of the
    divergence is identically zero regardless of velocity_y -- `_central_diff`
    returns all-zeros for a size-1 axis) has a known, hand-verifiable
    physical-space divergence: `_central_diff`'s one-sided-at-the-boundary,
    central-in-the-interior stencil on [0, 1, 2, 3] gives per-point values
    [1, 1, 1, 1] (a forward/backward difference of 1 at each boundary point,
    matching the interior slope), so mean(div^2) == 1.0 exactly.

    A second log-compressed field, built by adding a constant shift to the
    first field's log-compressed values, has an IDENTICAL log-space
    divergence (central differences cancel additive constants exactly) but
    a very different physical-space divergence once inverted, since
    log_to_field (expm1) is nonlinear. Before the fix, total_physics_loss
    called mass_conservation_penalty directly on log-compressed velocity, so
    these two fields would have been indistinguishable; evaluated correctly
    on log_to_field-inverted (physical-space) velocity, they are not.
    """

    u_phys = torch.tensor([0.0, 1.0, 2.0, 3.0]).reshape(1, 1, 1, 4)
    v_dummy = torch.zeros(1, 1, 1, 4)  # H=1 makes the dv/dy term identically zero

    u_log = field_to_log(u_phys)
    shifted_log = u_log + 2.5  # same central differences as u_log, different physical values after inversion

    # Sanity check on the construction itself: the shift must be invisible
    # to a divergence computed directly on log-space values.
    naive_penalty = mass_conservation_penalty(u_log, v_dummy)
    naive_penalty_shifted = mass_conservation_penalty(shifted_log, v_dummy)
    assert torch.allclose(naive_penalty, naive_penalty_shifted)

    corrected_penalty = mass_conservation_penalty(log_to_field(u_log), v_dummy)
    corrected_penalty_shifted = mass_conservation_penalty(log_to_field(shifted_log), v_dummy)

    assert torch.allclose(corrected_penalty, torch.tensor(1.0), atol=1e-5)
    assert corrected_penalty_shifted.item() > 0
    assert corrected_penalty_shifted.item() > corrected_penalty.item()
    # The two fields share identical log-space divergence but must be
    # distinguished once evaluated in physical space -- this is exactly the
    # failure mode the pre-fix total_physics_loss had.
    assert not torch.allclose(corrected_penalty, corrected_penalty_shifted)


def test_nonnegativity_penalty_equivalent_in_log_and_physical_space():
    """field_to_log is sign-preserving and zero-preserving
    (y = sign(x)*log1p(|x|) has y >= 0 <=> x >= 0), so checking
    non-negativity directly on log-compressed values gives the same verdict
    as checking physical values. nonnegativity_penalty should NOT receive a
    physical-space inversion -- this pins that down as a regression guard."""

    physical = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
    log_compressed = field_to_log(physical)

    assert torch.equal(physical < 0, log_compressed < 0)

    penalty_physical = nonnegativity_penalty(physical)
    penalty_log = nonnegativity_penalty(log_compressed)
    assert penalty_physical.item() > 0
    assert penalty_log.item() > 0


def test_total_physics_loss_inverts_velocity_before_mass_conservation():
    """End-to-end regression guard: total_physics_loss's mass_conservation
    term must match inverting the velocity channels (log_to_field) and
    calling mass_conservation_penalty directly, and must NOT match calling
    it on the raw (still log-compressed) channels."""

    torch.manual_seed(0)
    predicted_fields = torch.randn(2, 11, 6, 6)
    weights = {"data": 1.0, "mass_conservation": 0.05, "nonnegativity": 0.1}

    result = total_physics_loss(predicted_fields, weights, mode="finite_difference")

    velocity_x_physical = log_to_field(predicted_fields[:, 0:1])
    velocity_y_physical = log_to_field(predicted_fields[:, 1:2])
    expected_mass_conservation = mass_conservation_penalty(velocity_x_physical, velocity_y_physical)
    assert torch.allclose(result["mass_conservation"], expected_mass_conservation)

    buggy_mass_conservation = mass_conservation_penalty(predicted_fields[:, 0:1], predicted_fields[:, 1:2])
    assert not torch.allclose(result["mass_conservation"], buggy_mass_conservation)


def test_central_diff_matches_analytic_gradient_on_linear_ramp_including_boundary():
    """f(x, y) = x has constant analytic gradient df/dx = 1, df/dy = 0
    everywhere, including at the domain boundary -- there is nothing special
    about the edges of a linear ramp. `_central_diff` must match this
    exactly at every point, not just in the interior."""

    w, h = 5, 3
    ramp = torch.arange(w, dtype=torch.float32).reshape(1, 1, 1, w).expand(1, 1, h, w)

    df_dx = _central_diff(ramp, axis=3)
    df_dy = _central_diff(ramp, axis=2)

    assert torch.allclose(df_dx, torch.ones(1, 1, h, w))
    assert torch.allclose(df_dy, torch.zeros(1, 1, h, w))


def test_central_diff_has_no_periodic_wraparound_artifact_at_boundary():
    """Regression guard for the old `torch.roll`-based implementation: on a
    monotonic ramp [0, 1, 2, 3, 4], the old periodic version wrapped the
    high edge around to be adjacent to the low edge, producing a spurious
    sign flip and wrong magnitude (-1.5) at both boundary points instead of
    the correct constant slope (1.0). The fixed, non-periodic implementation
    must show neither: no sign flip, and it must disagree with what the old
    `torch.roll` formula would have produced at the boundary."""

    ramp = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 1, 5)

    diff = _central_diff(ramp, axis=3)
    assert torch.allclose(diff, torch.ones(1, 1, 1, 5))

    old_periodic = (torch.roll(ramp, -1, dims=3) - torch.roll(ramp, 1, dims=3)) / 2.0
    assert old_periodic[..., 0].item() < 0  # the old implementation's sign-flip artifact
    assert old_periodic[..., -1].item() < 0

    assert diff[..., 0].item() > 0  # no spurious sign flip at the left boundary
    assert diff[..., -1].item() > 0  # no spurious sign flip at the right boundary
    assert not torch.allclose(diff[..., 0], old_periodic[..., 0])
    assert not torch.allclose(diff[..., -1], old_periodic[..., -1])


def test_central_diff_zero_for_size_one_axis():
    """A size-1 axis has no spatial variation to differentiate -- must be
    exactly zero, matching the old torch.roll-based implementation's
    (incidental) behavior on a size-1 dimension, which several
    mass_conservation_penalty tests rely on (H=1 makes the dv/dy term
    identically zero regardless of velocity_y's values)."""

    field = torch.tensor([3.0]).reshape(1, 1, 1, 1)
    assert torch.equal(_central_diff(field, axis=2), torch.zeros_like(field))
    assert torch.equal(_central_diff(field, axis=3), torch.zeros_like(field))
