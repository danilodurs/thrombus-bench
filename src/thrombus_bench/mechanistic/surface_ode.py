"""Platelet surface deposition/aggregation ODE system and flux boundary conditions.

Responsibility
---------------
Implements the wall-surface reaction system: Appendix C's chemical flux
boundary conditions and simplified (theta=1) ODE system, combined with the
main-text Eqs. (6)-(7) mechanical (shear-gradient) flux boundary conditions
and Eq. (15)'s D_a-scaled combination pattern.

* Eqs. (6)-(7): mechanical deposition/aggregation flux boundary conditions
  for resting platelets `j_r,mech` and activated platelets `j_a,mech`,
  proportional to the *negative* axial wall-shear-rate gradient
  `d(gamma_w)/dx` (only active where the gradient is negative -- the
  model's key shear-gradient-induced aggregation mechanism, Nesbitt et al.).
* Eqs. (C.1)-(C.2): chemical adhesion/aggregation flux boundary conditions
  for resting (`j_r,chem`) and activated (`j_a,chem`) platelets.
* Eq. (C.7): saturation term S(x,t) = 1 - M(x,t)/M_inf.
* Eqs. (C.17)-(C.20): the theta=1-simplified chemical-only ODE system
  (M_r == 0 identically; M, M_as, M_at driven purely by k_rs[RP]/k_as[AP]
  adhesion + k_aa aggregation).
* Eq. (15): dM_at/dt = D_a*(j_r + j_a), where j_r = j_r,mech + j_r,chem
  (Eq. 8) and j_a = j_a,mech + j_a,chem (Eq. 9) -- i.e. M_at accumulates the
  *full* mechanical+chemical flux, including platelet-platelet aggregation.

Reconstruction note (explicit modeling assumption)
-------------------------------------------------------
The main-text block defining Eqs. (8)-(13) (how the general, theta-explicit
chemical fluxes C.1-C.6 combine with the mechanical fluxes Eqs. 6-7) render
as dense inline mathematics in the source PDF that could not be extracted
with full confidence for every subscript. This module instead builds the
combined system from the two blocks extracted with high confidence: the
mechanical fluxes (Eqs. 6-7, clearly legible) and the theta=1-simplified
chemical system (Eqs. C.17-C.20, clearly legible), following the pattern
Eq. (15) states unambiguously (`dM_at/dt = D_a*(j_r+j_a)`, `j_r`/`j_a` =
mechanical + chemical per platelet type):

    dM/dt     = D_a * (j_r,mech + j_r,chem + j_a,mech + S*k_as*[AP])
    dM_as/dt  = D_a * (j_r,mech + j_r,chem + j_a,mech + S*k_as*[AP])   (mirrors M; no aggregation)
    dM_at/dt  = D_a * (j_r,mech + j_r,chem + j_a,mech + j_a,chem)      (full j_a, includes k_aa aggregation)
    dM_r/dt   = 0                                                      (Eq. C.18, theta=1)

where `j_r,chem = S*k_rs*[RP]` (Eq. C.1) and `j_a,chem = S*k_as*[AP] +
(M_at/M_inf)*k_aa*[AP]` (Eq. C.2). M and M_as get only the *adhesion*
portion of the activated-platelet flux (excluding the k_aa aggregation
term), consistent with M_as's Eq. (C.19) chemical-only definition (which
has no k_aa term) and with M_at being the quantity that "represents...
platelet-platelet adhesion" (Appendix C, definition of M_at) on top of
surface adhesion. See README.md "Assumptions & Deviations from Source
Paper" for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


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

        z = np.zeros(len(wall_dofs), dtype=float)
        return cls(M=z.copy(), M_r=z.copy(), M_as=z.copy(), M_at=z.copy())


def saturation_term(M: np.ndarray, M_inf_plt_cm2: float) -> np.ndarray:
    """S(x,t) = 1 - M(x,t)/M_inf, Eq. (C.7)."""

    return 1.0 - M / M_inf_plt_cm2


def mechanical_flux_resting(
    gamma_w_ref: float,
    d_shear_rate_dx: np.ndarray,
    saturation: np.ndarray,
    resting_platelet_conc: np.ndarray,
    k_rs_cm_s: float,
    L_m: float,
) -> np.ndarray:
    """j_r,mech, Eq. (6): mechanical deposition flux for resting platelets.

    Nonzero only where d(gamma_w)/dx < 0 (negative shear gradient), per the
    paper's shear-gradient-induced aggregation mechanism.

    `gamma_w_ref` is a single scalar: "the wall shear rate in the straight
    vessel segment upstream of the aneurysm" (paper, Sec. 2.4) -- a fixed
    reference value, *not* the local per-node shear rate. Using the local
    value here would let the L/gamma_w factor blow up wherever local shear
    rate is small (e.g. near the inlet before the plug-flow BC has
    developed into a parabolic profile), which is not what the paper
    describes and is numerically catastrophic in practice.
    """

    gate = d_shear_rate_dx < 0.0
    factor = (L_m / max(gamma_w_ref, _EPS)) * np.abs(d_shear_rate_dx)
    return np.where(gate, factor * saturation * k_rs_cm_s * resting_platelet_conc, 0.0)


def mechanical_flux_activated(
    gamma_w_ref: float,
    d_shear_rate_dx: np.ndarray,
    saturation: np.ndarray,
    M: np.ndarray,
    M_inf_plt_cm2: float,
    activated_platelet_conc: np.ndarray,
    k_as_cm_s: float,
    k_aa_cm_s: float,
    L_m: float,
) -> np.ndarray:
    """j_a,mech, Eq. (7): mechanical deposition/aggregation flux for activated
    platelets. See `mechanical_flux_resting` docstring re: `gamma_w_ref`."""

    gate = d_shear_rate_dx < 0.0
    factor = (L_m / max(gamma_w_ref, _EPS)) * np.abs(d_shear_rate_dx)
    rate = saturation * k_as_cm_s + (M / M_inf_plt_cm2) * k_aa_cm_s
    return np.where(gate, factor * rate * activated_platelet_conc, 0.0)


def chemical_flux_resting(saturation: np.ndarray, resting_platelet_conc: np.ndarray, k_rs_cm_s: float) -> np.ndarray:
    """j_r,chem, Eq. (C.1)."""

    return saturation * k_rs_cm_s * resting_platelet_conc


def chemical_flux_activated(
    saturation: np.ndarray,
    M_at: np.ndarray,
    M_inf_plt_cm2: float,
    activated_platelet_conc: np.ndarray,
    k_as_cm_s: float,
    k_aa_cm_s: float,
) -> np.ndarray:
    """j_a,chem, Eq. (C.2)."""

    return (saturation * k_as_cm_s + (M_at / M_inf_plt_cm2) * k_aa_cm_s) * activated_platelet_conc


def surface_ode_rhs(
    state: SurfaceState,
    gamma_w_ref: float,
    d_shear_rate_dx: np.ndarray,
    resting_platelet_conc: np.ndarray,
    activated_platelet_conc: np.ndarray,
    k_rs_cm_s: float,
    k_as_cm_s: float,
    k_aa_cm_s: float,
    M_inf_plt_cm2: float,
    L_m: float,
    D_a: float,
) -> SurfaceState:
    """Right-hand side d(state)/dt, per this module's "Reconstruction note".

    `gamma_w_ref` is the fixed upstream-straight-vessel-segment shear rate
    scalar (see `mechanical_flux_resting` docstring), not a per-node array.
    """

    S = saturation_term(state.M, M_inf_plt_cm2)

    j_r_mech = mechanical_flux_resting(gamma_w_ref, d_shear_rate_dx, S, resting_platelet_conc, k_rs_cm_s, L_m)
    j_a_mech = mechanical_flux_activated(
        gamma_w_ref, d_shear_rate_dx, S, state.M, M_inf_plt_cm2, activated_platelet_conc, k_as_cm_s, k_aa_cm_s, L_m
    )
    j_r_chem = chemical_flux_resting(S, resting_platelet_conc, k_rs_cm_s)
    j_a_chem_adhesion_only = S * k_as_cm_s * activated_platelet_conc
    j_a_chem_full = chemical_flux_activated(S, state.M_at, M_inf_plt_cm2, activated_platelet_conc, k_as_cm_s, k_aa_cm_s)

    dM_dt = D_a * (j_r_mech + j_r_chem + j_a_mech + j_a_chem_adhesion_only)
    dM_as_dt = dM_dt.copy()
    dM_at_dt = D_a * (j_r_mech + j_r_chem + j_a_mech + j_a_chem_full)
    dM_r_dt = np.zeros_like(state.M)  # Eq. (C.18), theta=1

    return SurfaceState(M=dM_dt, M_r=dM_r_dt, M_as=dM_as_dt, M_at=dM_at_dt)


def step_surface_state(state: SurfaceState, rhs: SurfaceState, dt_s: float) -> SurfaceState:
    """Explicit Euler step, clipped to keep coverage fields non-negative
    (all are physically non-negative accumulated quantities)."""

    return SurfaceState(
        M=np.maximum(state.M + dt_s * rhs.M, 0.0),
        M_r=np.maximum(state.M_r + dt_s * rhs.M_r, 0.0),
        M_as=np.maximum(state.M_as + dt_s * rhs.M_as, 0.0),
        M_at=np.maximum(state.M_at + dt_s * rhs.M_at, 0.0),
    )
