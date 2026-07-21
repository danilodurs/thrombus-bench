"""Platelet surface deposition/aggregation ODE system and flux boundary conditions.

Responsibility
---------------
Implements the wall-surface reaction system of Appendix C: the flux
boundary conditions governing chemical and mechanical platelet adhesion and
aggregation, and the resulting ODEs for surface coverage state variables.

* Eqs. (6)-(7): mechanical deposition/aggregation flux boundary conditions
  for resting platelets `j_r,mech` and activated platelets `j_a,mech`,
  proportional to the *negative* axial wall-shear-rate gradient
  `d(gamma_w)/dx` (only active where the gradient is negative -- this is the
  model's key shear-gradient-induced aggregation mechanism, Nesbitt et al.).
* Eqs. (C.1)-(C.6): chemical adhesion/aggregation flux boundary conditions
  (resting-platelet adhesion, activated-platelet adhesion + aggregation,
  agonist release/synthesis, thrombin generation on resting/activated
  platelets).
* Eqs. (8)-(11): combined chemical+mechanical flux conditions.
* Eq. (C.7): saturation term S(x,t) = 1 - M(x,t)/M_inf.
* Eqs. (12)-(13), (C.8)-(C.11): definitions of M (total surface coverage),
  M_r (resting-platelet contribution), M_as (activated-surface contribution),
  M_at (total deposited activated platelets).
* Eqs. (14)-(15), (C.12)-(C.20): the ODE system for dM/dt, dM_r/dt,
  dM_as/dt, dM_at/dt, including the simplified form (Eqs. C.17-C.20)
  obtained by assuming theta=1 (all adhering resting platelets activate on
  surface contact) -- this simplified form is what this module implements
  by default, matching the paper's own simplification.

Scale terms (Table 1 / configs/physio_params.yaml `scale_terms`):
* L = 10 * vessel entrance diameter (Eqs. 6-7 flux scale factor).
* D_a: thrombus growth-rate scale factor (Eqs. 14-15); the paper does not
  report its calibrated numeric value, so it is treated here as a tunable
  calibration constant (see README.md "Assumptions & Deviations").

Not yet implemented -- this is a scaffolding stub.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SurfaceState:
    """Nodal (wall-boundary) surface coverage state, Appendix C.

    All fields are 1D arrays over wall boundary DOFs, units PLT/cm^2 (M,
    M_r, M_as, M_at).
    """

    M: np.ndarray
    M_r: np.ndarray
    M_as: np.ndarray
    M_at: np.ndarray

    @classmethod
    def zeros_like(cls, wall_dofs: np.ndarray) -> "SurfaceState":
        """Zero initial surface coverage, per paper assumption M(x,0)=M_r(x,0)
        =M_as(x,0)=M_at(x,0)=0 (see README.md "Assumptions & Deviations")."""

        z = np.zeros_like(wall_dofs, dtype=float)
        return cls(M=z.copy(), M_r=z.copy(), M_as=z.copy(), M_at=z.copy())


def saturation_term(M: np.ndarray, M_inf_plt_cm2: float) -> np.ndarray:
    """S(x,t) = 1 - M(x,t)/M_inf, Eq. (C.7)."""

    return 1.0 - M / M_inf_plt_cm2


def mechanical_flux_resting(
    wall_shear_rate: np.ndarray,
    d_shear_rate_dx: np.ndarray,
    saturation: np.ndarray,
    resting_platelet_conc: np.ndarray,
    k_rs_cm_s: float,
    L_cm: float,
):
    """j_r,mech, Eq. (6): mechanical deposition flux for resting platelets.

    Nonzero only where d(gamma_w)/dx < 0 (negative shear gradient), per the
    paper's shear-gradient-induced aggregation mechanism.
    """

    raise NotImplementedError("surface_ode.mechanical_flux_resting: not yet implemented")


def mechanical_flux_activated(
    wall_shear_rate: np.ndarray,
    d_shear_rate_dx: np.ndarray,
    saturation: np.ndarray,
    M: np.ndarray,
    M_inf_plt_cm2: float,
    activated_platelet_conc: np.ndarray,
    k_as_cm_s: float,
    k_aa_cm_s: float,
    L_cm: float,
):
    """j_a,mech, Eq. (7): mechanical deposition/aggregation flux for activated platelets."""

    raise NotImplementedError("surface_ode.mechanical_flux_activated: not yet implemented")


def surface_ode_rhs(state: SurfaceState, fluxes: dict, rates: dict, D_a: float) -> SurfaceState:
    """Right-hand side of the simplified ODE system, Eqs. (C.17)-(C.20)
    (theta=1 case): returns d(state)/dt as a `SurfaceState`."""

    raise NotImplementedError("surface_ode.surface_ode_rhs: not yet implemented")
