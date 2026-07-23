"""Physics-informed loss terms: PDE residuals, BC flux residuals, conservation penalties.

Responsibility
---------------
Penalize violations of the governing equations in the neural surrogate's
predicted fields, weighted per `configs/training.yaml` `physics_loss.weights`:

* `navier_stokes_residual` / `cdr_residual`: residuals of Eqs. (3)-(4) and
  Eq. (1) evaluated on the predicted fields.
* `mass_conservation`: penalizes a nonzero discrete divergence of the
  predicted velocity field (steady incompressible flow requires div(u)=0
  pointwise; `mechanistic/flow_solver.compute_boundary_flux` checks the
  integrated form exactly for the mechanistic solver -- here it is a *soft*
  differentiable training penalty on the surrogate's output).
* `nonnegativity`: penalizes negative predicted concentrations (all
  physical species/surface fields are non-negative).

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


def _central_diff(field: torch.Tensor, axis: int) -> torch.Tensor:
    """d(field)/d(axis), axis=2 -> y (rows), axis=3 -> x (cols); (B,C,H,W)."""

    if axis == 3:
        return (torch.roll(field, -1, dims=3) - torch.roll(field, 1, dims=3)) / 2.0
    return (torch.roll(field, -1, dims=2) - torch.roll(field, 1, dims=2)) / 2.0


def mass_conservation_penalty(velocity_x: torch.Tensor, velocity_y: torch.Tensor) -> torch.Tensor:
    """Soft penalty on the discrete divergence div(u) = du/dx + dv/dy of the
    predicted velocity field (should vanish for steady incompressible flow,
    Eq. 4)."""

    div = _central_diff(velocity_x, axis=3) + _central_diff(velocity_y, axis=2)
    return div.pow(2).mean()


def nonnegativity_penalty(*fields: torch.Tensor) -> torch.Tensor:
    """sum of relu(-field)^2 over all physical (non-negative) predicted fields."""

    return sum(F.relu(-f).pow(2).mean() for f in fields)


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


def total_physics_loss(predicted_fields: torch.Tensor, weights: dict, mode: str, viscosity: float = 0.0035) -> dict[str, torch.Tensor]:
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
    """

    if mode != "finite_difference":
        raise NotImplementedError(
            f"physics_losses.total_physics_loss: residual_mode={mode!r} not implemented (only 'finite_difference' is)"
        )

    velocity_x, velocity_y = predicted_fields[:, 0:1], predicted_fields[:, 1:2]
    species_fields = predicted_fields[:, 2:]

    return {
        "mass_conservation": mass_conservation_penalty(velocity_x, velocity_y),
        "nonnegativity": nonnegativity_penalty(species_fields),
    }
