"""Physics-informed loss terms: PDE residuals, BC flux residuals, conservation penalties.

Responsibility
---------------
Penalize violations of the governing equations in the neural surrogate's
predicted fields, weighted per `configs/training.yaml` `physics_loss.weights`:

* `navier_stokes_residual`: residual of Eqs. (3)-(4) (momentum +
  incompressibility) evaluated on the predicted velocity/pressure/viscosity
  fields.
* `cdr_residual`: residual of Eq. (1) (species convection-diffusion-reaction)
  evaluated on the predicted concentration fields, using the predicted
  velocity for advection.
* `surface_flux_bc_residual`: residual of the Eqs. (6)-(13) wall flux
  boundary conditions, evaluated at predicted wall-adjacent nodes.
* `mass_conservation`: global inlet-flux == outlet-flux penalty (the same
  quantity `mechanistic/flow_solver.compute_boundary_flux` checks exactly
  for the mechanistic solver; here it is a soft training penalty since the
  surrogate's output is not guaranteed divergence-free).
* `nonnegativity`: penalizes negative predicted concentrations/surface
  coverage (all physical species/surface fields are non-negative).

Residual computation mode (`configs/training.yaml` `physics_loss.residual_mode`)
-------------------------------------------------------------------------------
* `"finite_difference"`: compute spatial derivatives via finite differences
  on the regular latent/output grid (cheap, matches the encoder's fixed-grid
  representation, but grid-resolution-limited accuracy).
* `"autograd"`: reparameterize the network as a continuous coordinate ->
  field function and use `torch.autograd.grad` for exact derivatives
  (more accurate, more expensive, requires the operator core to support
  continuous query points rather than only fixed-grid output).

Both are exposed behind this config flag per project scope; see
`configs/training.yaml` for the default.

Not yet implemented -- this is a scaffolding stub.
"""

from __future__ import annotations

import torch


def navier_stokes_residual(velocity: torch.Tensor, pressure: torch.Tensor, viscosity: torch.Tensor, mode: str) -> torch.Tensor:
    """Residual of Eqs. (3)-(4) on predicted fields."""

    raise NotImplementedError("physics_losses.navier_stokes_residual: not yet implemented")


def cdr_residual(concentration: torch.Tensor, velocity: torch.Tensor, diffusivity: float, source: torch.Tensor, mode: str) -> torch.Tensor:
    """Residual of Eq. (1) on a predicted species field."""

    raise NotImplementedError("physics_losses.cdr_residual: not yet implemented")


def surface_flux_bc_residual(predicted_fields: dict, wall_mask: torch.Tensor) -> torch.Tensor:
    """Residual of the Eqs. (6)-(13) wall flux boundary conditions."""

    raise NotImplementedError("physics_losses.surface_flux_bc_residual: not yet implemented")


def mass_conservation_penalty(velocity: torch.Tensor, inlet_mask: torch.Tensor, outlet_mask: torch.Tensor) -> torch.Tensor:
    """Soft penalty on |flux_in + flux_out| for the predicted velocity field."""

    raise NotImplementedError("physics_losses.mass_conservation_penalty: not yet implemented")


def nonnegativity_penalty(*fields: torch.Tensor) -> torch.Tensor:
    """sum of relu(-field)^2 over all physical (non-negative) predicted fields."""

    return sum(torch.relu(-f).pow(2).mean() for f in fields)


def total_physics_loss(predicted_fields: dict, weights: dict, mode: str) -> torch.Tensor:
    """Weighted sum of all physics-informed loss terms, per
    `configs/training.yaml` `physics_loss.weights`."""

    raise NotImplementedError("physics_losses.total_physics_loss: not yet implemented")
