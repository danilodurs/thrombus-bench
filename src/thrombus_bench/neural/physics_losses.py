"""Physics-informed loss terms: PDE residuals, BC flux residuals, conservation penalties.

Responsibility
---------------
Penalize violations of the governing equations in the neural surrogate's
predicted fields, weighted per `configs/training.yaml` `physics_loss.weights`:

Space each penalty operates in
-------------------------------
`total_physics_loss`'s input (and the model's raw output, `pred`) is
entirely in the **log-compressed space** of `data/dataset.field_to_log`
(`sign(x) * log1p(|x|)`, applied per-channel including velocity -- see that
module's docstring). Which space a penalty must evaluate in depends on
whether the physical law it encodes is linear in the field or not:

* `mass_conservation`: div(u) = du/dx + dv/dy is a *linear* differential
  operator, and `log1p`/`sign` are nonlinear, so differencing the
  log-compressed velocity channels directly does NOT measure physical
  divergence -- `d/dx[sign(u)*log1p(|u|)] != f(du/dx)` for any simple `f`.
  `total_physics_loss` therefore inverts the velocity channels back to
  **physical space** via `data/dataset.log_to_field` before calling
  `mass_conservation_penalty` (steady incompressible flow requires
  div(u)=0 pointwise; `mechanistic/flow_solver.compute_boundary_flux` checks
  the integrated form exactly for the mechanistic solver -- here it is a
  *soft* differentiable training penalty on the surrogate's output).
* `nonnegativity`: non-negativity is a *sign* condition, and
  `field_to_log`/`log_to_field` are sign-preserving and zero-preserving
  (`y >= 0 <=> x >= 0` for `y = field_to_log(x)`), so checking it directly
  on the **log-compressed** predicted channels (no inversion) gives exactly
  the same verdict as checking physical concentrations. All physical
  species/surface fields are non-negative.
* `navier_stokes_residual` / `cdr_residual`: residuals of Eqs. (3)-(4) and
  Eq. (1); like `mass_conservation`, these involve differential operators
  that are not invariant under the log transform, so any caller wiring them
  in must pass **physical-space** velocity/pressure/concentration fields
  (see `total_physics_loss` below for why `navier_stokes_residual` isn't
  currently wired in at all).

Fluid-domain masking (optional `mask` argument)
-------------------------------------------------
`mass_conservation_penalty`, `nonnegativity_penalty`, and `total_physics_loss`
all accept an optional `mask` -- the per-sample fluid/exterior raster
`data/dataset.py`'s `fluid_mask` (True where a grid cell is inside the
actual FEM mesh domain; see `data/generate_dataset._fluid_mask`). The
rasterization grid's bounding box contains genuine exterior cells (the
vessel+aneurysm domain is an L/T-shaped union), and those cells hold
whatever `griddata(method="nearest")` filled in -- not meaningful values.
Without masking, a penalty averaged over the whole grid is diluted by the
(easy, mostly-constant/near-zero) exterior background, understating how
badly it's actually doing on the (hard) fluid interior. When `mask` is
given, `mass_conservation_penalty` additionally erodes it by one pixel
before use (see `_erode_mask_one_pixel`) because it's a spatial-derivative
penalty: a central difference computed one cell inside the mask boundary
still reads its exterior neighbor's meaningless value. `nonnegativity_penalty`
is a pointwise sign check with no derivative, so it uses the mask directly,
unmodified.

Residual computation mode (`configs/training.yaml` `physics_loss.residual_mode`)
-------------------------------------------------------------------------------
* `"finite_difference"`: spatial derivatives via central-difference
  convolution kernels on the regular output grid. **Implemented** -- used
  by `total_physics_loss` below.
* `"autograd"`: derivatives via `torch.autograd.grad` on a continuous
  coordinate -> field reparameterization of the network. **Not
  implemented** in this project (scope note: would require the operator
  core to support continuous query points rather than only fixed-grid
  output; `total_physics_loss` raises `NotImplementedError` for this mode).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.dataset import log_to_field


def _central_diff(field: torch.Tensor, axis: int) -> torch.Tensor:
    """d(field)/d(axis), axis=2 -> y (rows), axis=3 -> x (cols); (B,C,H,W).

    Central differences in the interior; one-sided forward/backward
    differences at the two boundary rows/columns of that axis. The grid is
    a bounded vessel+aneurysm domain (inlet/outlet/wall boundaries), not a
    periodic one, so a `torch.roll`-based implementation (wrapping the
    boundary around) is physically wrong -- it treats a value at one edge of
    the grid as adjacent to the opposite edge. (Padding with
    `mode="replicate"` and reusing the central-difference formula everywhere
    would halve the boundary derivative -- e.g. `(f[1]-f[0])/2` instead of
    `f[1]-f[0]` -- since the replicated edge value counts as one of the two
    central-difference neighbors; slicing explicitly avoids that.)
    """

    n = field.size(axis)
    if n == 1:
        return torch.zeros_like(field)

    first = field.narrow(axis, 0, 1)
    second = field.narrow(axis, 1, 1)
    last = field.narrow(axis, n - 1, 1)
    second_last = field.narrow(axis, n - 2, 1)

    left = second - first  # forward difference at the first boundary point
    right = last - second_last  # backward difference at the last boundary point

    if n == 2:
        return torch.cat([left, right], dim=axis)

    interior = (field.narrow(axis, 2, n - 2) - field.narrow(axis, 0, n - 2)) / 2.0
    return torch.cat([left, interior, right], dim=axis)


def _prepare_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-sample fluid mask, shape (B, H, W) (as saved/exposed
    by `data/dataset.py`'s `fluid_mask`), to `reference`'s (B, C, H, W)
    shape by inserting the channel axis, matching its dtype/device."""

    if mask.dim() == reference.dim() - 1:
        mask = mask.unsqueeze(1)
    return mask.to(dtype=reference.dtype, device=reference.device)


def _erode_mask_one_pixel(mask: torch.Tensor) -> torch.Tensor:
    """Erode a 0/1 fluid mask by one pixel (4-connected: up/down/left/right
    neighbors) -- see module docstring's "Fluid-domain masking" section for
    why `mass_conservation_penalty` needs this and `nonnegativity_penalty`
    does not.

    A cell survives erosion only if it and every one of its *existing*
    neighbors are also fluid. Missing neighbors (true array edges, not mask
    edges) are treated as trivially satisfied rather than as exterior: at a
    genuine grid edge, `_central_diff` already falls back to a one-sided
    stencil that never reads anything outside the array, so there is no
    fluid/exterior mixing to guard against there -- only at an internal
    mask boundary, where a central difference reads across it.
    """

    up = torch.ones_like(mask)
    up[..., 1:, :] = mask[..., :-1, :]
    down = torch.ones_like(mask)
    down[..., :-1, :] = mask[..., 1:, :]
    left = torch.ones_like(mask)
    left[..., :, 1:] = mask[..., :, :-1]
    right = torch.ones_like(mask)
    right[..., :, :-1] = mask[..., :, 1:]
    return mask * up * down * left * right


def mass_conservation_penalty(
    velocity_x: torch.Tensor, velocity_y: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Soft penalty on the discrete divergence div(u) = du/dx + dv/dy of the
    predicted velocity field (should vanish for steady incompressible flow,
    Eq. 4).

    If `mask` is given (fluid/exterior raster, see module docstring's
    "Fluid-domain masking" section), the mean is taken only over fluid
    cells, eroded by one pixel first (`_erode_mask_one_pixel`) so that
    cells whose central difference reads an exterior neighbor are excluded
    too, not just genuinely exterior cells themselves.
    """

    div = _central_diff(velocity_x, axis=3) + _central_diff(velocity_y, axis=2)
    if mask is None:
        return div.pow(2).mean()

    eroded = _erode_mask_one_pixel(_prepare_mask(mask, div))
    return (div.pow(2) * eroded).sum() / eroded.sum().clamp_min(1.0)


def nonnegativity_penalty(*fields: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """sum of relu(-field)^2 over all physical (non-negative) predicted fields.

    If `mask` is given (fluid/exterior raster, see module docstring's
    "Fluid-domain masking" section), each field's mean is taken only over
    fluid cells -- unlike `mass_conservation_penalty`, no erosion: this is a
    pointwise sign check with no spatial derivative, so a fluid cell's own
    value is trustworthy regardless of its neighbors.
    """

    if mask is None:
        return sum(F.relu(-f).pow(2).mean() for f in fields)

    def _masked_term(f: torch.Tensor) -> torch.Tensor:
        m = _prepare_mask(mask, f)
        return (F.relu(-f).pow(2) * m).sum() / m.sum().clamp_min(1.0)

    return sum(_masked_term(f) for f in fields)


def navier_stokes_residual(velocity_x: torch.Tensor, velocity_y: torch.Tensor, pressure: torch.Tensor, viscosity: float) -> torch.Tensor:
    """Simplified steady-Stokes momentum residual (Eqs. 3-4, inertia
    dropped, matching `mechanistic/flow_solver.py`'s own simplification):
    `-grad(p) + mu*laplacian(u) ~= 0`, penalized as a soft residual norm."""

    lap_u = _central_diff(_central_diff(velocity_x, 3), 3) + _central_diff(_central_diff(velocity_x, 2), 2)
    lap_v = _central_diff(_central_diff(velocity_y, 3), 3) + _central_diff(_central_diff(velocity_y, 2), 2)
    dp_dx = _central_diff(pressure, 3)
    dp_dy = _central_diff(pressure, 2)
    res_x = -dp_dx + viscosity * lap_u
    res_y = -dp_dy + viscosity * lap_v
    return res_x.pow(2).mean() + res_y.pow(2).mean()


def cdr_residual(*args, **kwargs) -> torch.Tensor:
    raise NotImplementedError(
        "physics_losses.cdr_residual: not implemented in this project (scope note in module docstring)"
    )


def surface_flux_bc_residual(*args, **kwargs) -> torch.Tensor:
    raise NotImplementedError(
        "physics_losses.surface_flux_bc_residual: not implemented in this project (scope note in module docstring)"
    )


def total_physics_loss(
    predicted_fields: torch.Tensor,
    weights: dict,
    mode: str,
    viscosity: float = 0.0035,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted physics-informed loss terms computable without ground-truth
    reference data, evaluated on `predicted_fields` (B, C, H, W) ordered per
    `data/dataset.FIELD_NAMES`: [velocity_x, velocity_y, conc_RP, conc_AP,
    conc_APR, conc_APS, conc_T, conc_AT, conc_PT, conc_FG, conc_FI].

    Returns a dict of named (unweighted) loss terms; `train.py` applies
    `weights` and sums. Only `mass_conservation` and `nonnegativity` are
    implemented here (see module docstring for `navier_stokes_residual`,
    which is implemented but not wired into this default set: besides a
    representative viscosity scalar, it needs a predicted pressure field,
    and pressure is not one of the surrogate's output channels -- there is
    no pressure entry in `data/dataset.FIELD_NAMES`. Wiring it in would mean
    adding a pressure output channel through the dataset/model pipeline, not
    just this function).

    `predicted_fields` is in `data/dataset.field_to_log`'s log-compressed
    space (same as the model's raw output and training target). See the
    module docstring's "Space each penalty operates in" section:
    `mass_conservation` needs the velocity channels inverted back to
    physical space first (`data/dataset.log_to_field`); `nonnegativity` is
    evaluated directly on the log-compressed species channels, which is
    equivalent to physical-space non-negativity since the transform is
    sign-preserving.

    `mask` (optional, `data/dataset.py`'s `fluid_mask`, shape (B, H, W)) is
    forwarded to both penalties -- see the module docstring's "Fluid-domain
    masking" section.
    """

    if mode != "finite_difference":
        raise NotImplementedError(
            f"physics_losses.total_physics_loss: residual_mode={mode!r} not implemented (only 'finite_difference' is)"
        )

    velocity_x_log, velocity_y_log = predicted_fields[:, 0:1], predicted_fields[:, 1:2]
    species_fields_log = predicted_fields[:, 2:]

    velocity_x_physical = log_to_field(velocity_x_log)
    velocity_y_physical = log_to_field(velocity_y_log)

    return {
        "mass_conservation": mass_conservation_penalty(velocity_x_physical, velocity_y_physical, mask=mask),
        "nonnegativity": nonnegativity_penalty(species_fields_log, mask=mask),
    }
