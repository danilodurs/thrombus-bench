"""Chemical and mechanical platelet activation source terms.

Responsibility
---------------
Implements the source terms S_i in Eq. (1) related to platelet activation
and agonist kinetics (Appendix A), and the mechanical (shear-induced)
activation rate constant (Appendix B):

* Eq. (A.1)-(A.7): chemical source terms for RP, AP, APR (e.g. ADP), APS
  (e.g. TxA2), PT, T, AT, following Sorensen et al.'s formulation.
* Eq. (A.8)-(A.9): the chemical activation rate constant `k_pa,chem` as a
  function of the weighted-agonist activation function `Omega`, gated by a
  smooth step function localized where `Omega == 1`.
* Eq. (A.10): heparin-catalyzed thrombin inhibition rate `Gamma`.
* Eq. (B.1): the mechanical activation rate constant `k_pa,mech`, a
  piecewise-linear function of local shear rate above `gamma_crit`.
* Eq. (5): total activation rate `k_pa = k_pa,chem + k_pa,mech`.

Smooth step functions (explicit modeling assumption)
---------------------------------------------------------
The paper defines `step(Omega)` (Eq. A.8-A.9) as a function "localized where
Omega is equal to one and whose minimum and maximum values equal zero and
one" without giving its functional form, and `k_pa,mech` (Eq. B.1) as a hard
piecewise-linear function of `gamma_dot/gamma_crit` that is itself
discontinuous in slope at gamma_dot = gamma_crit. Both are implemented here
as `(x / t_act) * logistic_step(x, center=1)`, i.e. the exact hard-threshold
value multiplied by a logistic gate centered at the threshold -- this
reproduces the paper's stated limits (0 below threshold, the exact linear
value well above it) while smoothing the transition, per
`configs/physio_params.yaml` `activation.smoothing.steepness_omega`. This is
a documented deviation from the paper, which does not specify a smoothing
form -- see README.md "Assumptions & Deviations from Source Paper".
"""

from __future__ import annotations

import numpy as np


def logistic_step(x, center: float, steepness: float, lo: float = 0.0, hi: float = 1.0):
    """Sigmoid approximation of a hard step at x=center (relative deviation).

    Shared functional form with `flow_solver._logistic_step` (Eq. 18's
    Theta_plts/Theta_FI); duplicated here (rather than imported) since it is
    a three-line pure function used by conceptually distinct equations
    (Appendix A/B activation vs. Eq. 18 viscosity feedback).
    """

    return lo + (hi - lo) / (1.0 + np.exp(-steepness * (np.asarray(x, dtype=float) / center - 1.0)))


def activation_function_omega(agonist_concentrations: dict, agonist_thresholds: dict, weights: dict) -> np.ndarray:
    """Weighted-agonist activation function, Eq. (A.9):

        Omega = sum_j w_j * [a_j] / [a_j,crit]

    `agonist_concentrations`, `agonist_thresholds`, `weights` are dicts
    keyed by agonist name (this model's two agonists: "ADP", "TxA2"), each
    mapping to an array (nodal concentration field) or scalar.
    """

    omega = 0.0
    for name, conc in agonist_concentrations.items():
        omega = omega + weights[name] * conc / agonist_thresholds[name]
    return np.asarray(omega, dtype=float)


def chemical_activation_rate(omega: np.ndarray, t_act_s: float, steepness_omega: float) -> np.ndarray:
    """k_pa,chem, Eq. (A.8)-(A.9) (smoothed, see module docstring):

        k_pa,chem = (Omega / t_act) * logistic_step(Omega, center=1)
    """

    omega = np.asarray(omega, dtype=float)
    gate = logistic_step(omega, center=1.0, steepness=steepness_omega, lo=0.0, hi=1.0)
    return (omega / t_act_s) * gate


def mechanical_activation_rate(
    shear_rate: np.ndarray, gamma_crit_per_s: float, t_act_s: float, steepness_omega: float
) -> np.ndarray:
    """k_pa,mech, Eq. (B.1) (smoothed, see module docstring):

        k_pa,mech = (gamma_dot / (gamma_crit * t_act)) * logistic_step(gamma_dot/gamma_crit, center=1)
    """

    ratio = np.asarray(shear_rate, dtype=float) / gamma_crit_per_s
    gate = logistic_step(ratio, center=1.0, steepness=steepness_omega, lo=0.0, hi=1.0)
    return (ratio / t_act_s) * gate


def total_activation_rate(k_pa_chem: np.ndarray, k_pa_mech: np.ndarray) -> np.ndarray:
    """k_pa = k_pa,chem + k_pa,mech, Eq. (5)."""

    return k_pa_chem + k_pa_mech


def thrombin_inhibition_rate(
    heparin_conc_uM: float,
    thrombin_conc_uM: np.ndarray,
    antithrombin_conc_uM: np.ndarray,
    k1_T_per_s: float,
    K_AT_uM: float,
    K_T_uM: float,
    alpha: float,
    beta_nmol_per_U: float,
) -> np.ndarray:
    """Gamma, Eq. (A.10): heparin-catalyzed thrombin inhibition by antithrombin.

        Gamma = k1_T * [H] * [AT] / (alpha*K_AT*K_T + alpha*K_AT*[T] + [AT]*beta*[T])
    """

    T = np.asarray(thrombin_conc_uM, dtype=float)
    AT = np.asarray(antithrombin_conc_uM, dtype=float)
    numerator = k1_T_per_s * heparin_conc_uM * AT
    denominator = alpha * K_AT_uM * K_T_uM + alpha * K_AT_uM * T + AT * beta_nmol_per_U * T
    return numerator / denominator


def chemical_source_terms(
    RP: np.ndarray,
    AP: np.ndarray,
    ADP: np.ndarray,
    TxA2: np.ndarray,
    PT: np.ndarray,
    T: np.ndarray,
    AT: np.ndarray,
    k_pa_chem: np.ndarray,
    gamma_inhibition: np.ndarray,
    lambda_ADP_nmol_plt: float,
    k1_ADP_per_s: float,
    s_p_TxA2_nmol_plt_s: float,
    k1_TxA2_per_s: float,
    phi_at_U_plt_s_uMPT: float,
    phi_rt_U_plt_s_uMPT: float,
    beta_nmol_per_U: float,
) -> dict:
    """Bulk chemical source terms S_i in Eq. (1), Eqs. (A.1)-(A.7).

    `k_pa_chem` is the resting-platelet activation rate (this function's
    `chemical_activation_rate` output); `gamma_inhibition` is Gamma (this
    function's `thrombin_inhibition_rate` output). Returns a dict of arrays
    keyed by species name, matching `species_transport.Species`.

    Note: this bulk formulation uses [AP] and [RP] (not the wall-restricted
    M_at/M_r used by the Appendix C flux boundary conditions in
    `surface_ode.py`) -- Eqs. (A.5)-(A.7) describe thrombin generation from
    *bulk* platelets, distinct from (and additive with) the *surface*
    thrombin generation flux BCs (Eqs. C.5-C.6).
    """

    S_RP = -k_pa_chem * RP
    S_AP = k_pa_chem * RP
    S_ADP = lambda_ADP_nmol_plt * k_pa_chem * RP - k1_ADP_per_s * ADP
    S_TxA2 = s_p_TxA2_nmol_plt_s * k_pa_chem * RP - k1_TxA2_per_s * TxA2
    thrombin_generation = PT * (phi_at_U_plt_s_uMPT * AP + phi_rt_U_plt_s_uMPT * RP)
    S_PT = -beta_nmol_per_U * thrombin_generation
    S_T = -gamma_inhibition * T + thrombin_generation
    S_AT = -gamma_inhibition * beta_nmol_per_U * T

    return {
        "RP": S_RP,
        "AP": S_AP,
        "APR": S_ADP,
        "APS": S_TxA2,
        "PT": S_PT,
        "T": S_T,
        "AT": S_AT,
    }
