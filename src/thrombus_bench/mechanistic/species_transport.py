"""Convection-diffusion-reaction (CDR) transport of the nine chemical species.

Responsibility
---------------
Assemble and solve Eq. (1) for each of the nine species tracked by the
model, advected by the velocity field from `flow_solver.py`:

    d[C_i]/dt + div(u * [C_i]) = div(D_i * grad([C_i])) + S_i

Species (paper Sec. 2.1): resting platelets (RP), activated platelets (AP),
agonists released from platelet granules (APR, e.g. ADP), agonists
synthesized by activated platelets (APS, e.g. TxA2), thrombin (T),
antithrombin (AT), prothrombin (PT), fibrinogen (FG), fibrin (FI).

Source terms S_i are defined in Appendix A (assembled by `activation.py` for
the platelet/agonist/thrombin terms and `fibrin.py` for the FG/FI terms) and
depend on the surface state (M, M_r, M_as, M_at) tracked by `surface_ode.py`.

Diffusivity closure (explicit modeling assumption)
------------------------------------------------------
Table 1 gives Brownian diffusivities `D_b,i` (configs/physio_params.yaml
`species.diffusion_cm2_s`). The paper states these are augmented by a
"shear-dependent" enhancement due to red blood cells (Sec. 2.1) but does not
give the closed-form expression used. This module should implement a
configurable enhancement `D_i(gamma_dot) = D_b,i * (1 + k_rbc * gamma_dot)`
(or similar), with `k_rbc` exposed as a config parameter and defaulting to 0
(pure Brownian diffusion) when not specified -- document the chosen closure
explicitly in README.md "Assumptions & Deviations" once implemented.

Boundary conditions (Sec. 2.1, Sec. 2.3-2.4)
-----------------------------------------------
* Inlet: Dirichlet, constant physiological concentration per species
  (`configs/physio_params.yaml` `species.*_inlet_*`), thrombin = 0.
* Outlet: zero axial concentration gradient (natural/do-nothing in the weak
  form).
* Walls: heterogeneous flux (Robin-type) boundary conditions from Appendix C
  (`surface_ode.py`) for platelet species; no-flux for species not involved
  in surface reactions (per Appendix C note on antithrombin).

Not yet implemented -- this is a scaffolding stub. See mesh.py/flow_solver.py
for the first fully-implemented modules and README.md for project status.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Species(str, Enum):
    """The nine transported species, Sec. 2.1."""

    RESTING_PLATELETS = "RP"
    ACTIVATED_PLATELETS = "AP"
    AGONIST_RELEASED = "APR"  # e.g. ADP
    AGONIST_SYNTHESIZED = "APS"  # e.g. TxA2
    THROMBIN = "T"
    ANTITHROMBIN = "AT"
    PROTHROMBIN = "PT"
    FIBRINOGEN = "FG"
    FIBRIN = "FI"


@dataclass
class SpeciesTransportState:
    """Nodal concentration fields for all nine species at a single time step."""

    concentrations: dict  # Species -> np.ndarray (nodal DOF vector)
    time_s: float


def assemble_cdr_system(*args, **kwargs):
    """Assemble the CDR system (Eq. 1) mass/advection/diffusion matrices.

    Will build, per species, a `BilinearForm` combining a mass matrix, an
    SUPG-stabilized advection term using the frozen velocity field from
    `flow_solver.FlowSolution`, and a diffusion term using the (optionally
    shear-enhanced) diffusivity. Source terms are added as a `LinearForm`
    evaluated from the current activation/fibrin/surface-ODE state.
    """

    raise NotImplementedError("species_transport.assemble_cdr_system: not yet implemented")


def step_species_transport(*args, **kwargs):
    """Advance all nine species concentration fields by one implicit time step."""

    raise NotImplementedError("species_transport.step_species_transport: not yet implemented")
