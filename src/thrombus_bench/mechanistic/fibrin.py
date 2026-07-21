"""Thrombin-mediated fibrin generation (Michaelis-Menten kinetics).

Responsibility
---------------
Implements the fibrinogen-consumption / fibrin-generation source terms
added to Eq. (1) for species FG and FI, following Anand et al. (2003):

    S_FG = - k_fi^th * [T] * [FG] / (k_m,fi^th + [FG])      Eq. (16)
    S_FI = + k_fi^th * [T] * [FG] / (k_m,fi^th + [FG])      Eq. (17)

with k_fi^th = 59 s^-1 and k_m,fi^th = 3.16 uM (Table 1 / configs/physio_params.yaml
`fibrin`). Inlet concentrations: fibrinogen 7 uM, fibrin 0 uM.

Fibrin is one of the two thrombus-viscosity trigger fields (the other being
deposited activated platelets M_at), consumed by `flow_solver.viscosity_multiplier`
(Eq. 18) once FI locally exceeds `fibrin.fibrin_critical_uM` (0.6 uM, Table 1).

Not yet implemented -- this is a scaffolding stub.
"""

from __future__ import annotations

import numpy as np


def fibrin_source_terms(
    thrombin_conc_uM: np.ndarray,
    fibrinogen_conc_uM: np.ndarray,
    k_fi_th_per_s: float,
    k_mfi_th_uM: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (S_FG, S_FI), Eqs. (16)-(17)."""

    rate = k_fi_th_per_s * thrombin_conc_uM * fibrinogen_conc_uM / (k_mfi_th_uM + fibrinogen_conc_uM)
    return -rate, rate
