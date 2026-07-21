"""Monolithic time-stepping coupling loop for the full mechanistic model.

Responsibility
---------------
Couples `flow_solver.py` (Navier-Stokes/Stokes + Carreau viscosity),
`species_transport.py` (9-species CDR), `activation.py` (chemical +
mechanical activation source terms), `surface_ode.py` (wall surface
coverage ODEs), and `fibrin.py` (fibrin generation) into the full transient
"Navier Stokes-CDR-ODE" system described in Sec. 2.6-2.7 of the paper.

Coupling scheme
------------------
Each time step:

1. Compute wall shear rate gamma_w and its axial gradient d(gamma_w)/dx
   from the current `FlowSolution` (needed by the Eq. 6-7 mechanical flux
   BCs).
2. Update the surface ODE state (`surface_ode.py`) using the current
   species concentrations and shear field.
3. Update species concentrations one CDR time step (`species_transport.py`),
   using source terms from `activation.py` and `fibrin.py` and the
   just-updated surface state for the wall flux BCs.
4. Recompute the Carreau viscosity field including the Eq. (18) thrombus
   multiplier (`flow_solver.viscosity_multiplier`), evaluated from the
   updated M_at and FI fields.
5. Re-solve the flow field (`flow_solver.solve_steady_flow`, or a
   quasi-steady re-solve for pulsatile inflow) with the updated viscosity.
6. Repeat until the requested end time, checkpointing state at the
   requested output cadence.

This is a fixed-point (Picard-style) operator-splitting scheme across
physics, distinct from the Picard iteration *within* `flow_solver.py` that
handles the Carreau nonlinearity at fixed viscosity-affecting fields.

Not yet implemented -- this is a scaffolding stub. Depends on
`species_transport.py`, `activation.py`, and `surface_ode.py` being
implemented first.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoupledSimulationState:
    """Full model state at a single time point: flow, species, surface, fibrin."""

    time_s: float
    flow: object  # flow_solver.FlowSolution
    species: object  # species_transport.SpeciesTransportState
    surface: object  # surface_ode.SurfaceState


def run_coupled_simulation(*args, **kwargs):
    """Advance the full coupled model from t=0 to `end_time_s`, per the
    time-stepping scheme in this module's docstring."""

    raise NotImplementedError("coupled_solver.run_coupled_simulation: not yet implemented")
