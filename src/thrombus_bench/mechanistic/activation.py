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
discontinuous in slope at gamma_dot = gamma_crit. Per configs/physio_params.yaml
`activation.smoothing.steepness_omega`, this module approximates `step(Omega)`
with a logistic sigmoid centered at Omega=1, and (optionally) smooths the
Eq. (B.1) kink with the same family of function. This is a documented
deviation from the paper, which does not specify a smoothing form -- see
README.md "Assumptions & Deviations from Source Paper".

Not yet implemented -- this is a scaffolding stub.
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

    return lo + (hi - lo) / (1.0 + np.exp(-steepness * (np.asarray(x) / center - 1.0)))


def activation_function_omega(agonist_concentrations: dict, agonist_thresholds: dict, weights: dict) -> np.ndarray:
    """Weighted-agonist activation function Omega, Eq. (A.9).

    Omega = sum_j w_j * [a_j] / [a_j,crit]
    """

    raise NotImplementedError("activation.activation_function_omega: not yet implemented")


def chemical_activation_rate(omega: np.ndarray, steepness_omega: float) -> np.ndarray:
    """k_pa,chem = Omega * step(Omega), Eq. (A.8)-(A.9) (smoothed, see module docstring)."""

    raise NotImplementedError("activation.chemical_activation_rate: not yet implemented")


def mechanical_activation_rate(shear_rate: np.ndarray, gamma_crit: float, k_pa_mech_coeff: float) -> np.ndarray:
    """k_pa,mech, Eq. (B.1): piecewise-linear ramp above the critical shear rate."""

    raise NotImplementedError("activation.mechanical_activation_rate: not yet implemented")


def total_activation_rate(k_pa_chem: np.ndarray, k_pa_mech: np.ndarray) -> np.ndarray:
    """k_pa = k_pa,chem + k_pa,mech, Eq. (5)."""

    return k_pa_chem + k_pa_mech


def thrombin_inhibition_rate(
    heparin_conc_uM: float,
    thrombin_conc_uM: np.ndarray,
    antithrombin_conc_uM: np.ndarray,
    sorensen_params: dict,
) -> np.ndarray:
    """Gamma, Eq. (A.10): heparin-catalyzed thrombin inhibition by antithrombin."""

    raise NotImplementedError("activation.thrombin_inhibition_rate: not yet implemented")
