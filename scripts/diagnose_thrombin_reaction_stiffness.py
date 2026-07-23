"""Isolated diagnostic for the [T]/[FI] runaway-growth "Known limitation"
documented in `mechanistic/coupled_solver.py` and README.md "Known
limitations".

This does NOT change any production code or equation. It is read-only
diagnosis: take ONE representative local (single-node, no spatial
transport) state and integrate the *local reaction ODE system only* two
ways, to separate three competing explanations for the runaway:

1. The local reaction ODE system itself is unstable/stiff-but-correct.
2. The coupling/substepping scheme (`species_transport.reaction_step`'s
   fixed-substep, fixed-iteration-count implicit backward-Euler + Newton
   solve) is introducing/amplifying an instability that isn't really there.
3. There is a genuine units/scaling issue in the C.5-C.6 surface-flux
   pathway (`phi_at`/`phi_rt` acting on M_at/M_r).

Method
------
`source_fn` here is assembled from the exact same building blocks as
`coupled_solver.run_coupled_simulation`'s inner `source_fn` closure:
`activation.chemical_source_terms` (Eqs. A.1-A.7, includes the *bulk*
platelet thrombin-generation term via [AP]/[RP]) + `activation.thrombin_
inhibition_rate` (Eq. A.10's Gamma) + `fibrin.fibrin_source_terms` (Eqs.
16-17). Scenarios:

* `bulk_only`: exactly `coupled_solver.py`'s production `source_fn` --
  no C.5-C.6 term at all (in production, that term is applied separately,
  downstream of `reaction_step`, as a mesh-dependent Neumann flux BC in
  `species_transport.solve_transport_step`, not as part of the local
  reaction ODE).
* `bulk_plus_surface_flux_*`: the same, plus an added term with the same
  functional form as Eqs. C.5-C.6's thrombin generation,
  (M_at*phi_at + M_r*phi_rt) * PT, subtracted from PT and added to T
  (matching `coupled_solver.py`'s sign convention: `S_PT -= beta*gen`,
  `S_T += gen`). M_r = 0 throughout, matching `surface_ode.py`'s permanent
  M_r=0 under the theta=1 simplification. Multiple M_at magnitudes/unit
  conventions are tested (see `SURFACE_FLUX_SCENARIOS` below), because a
  first pass at this diagnostic (native-unit M_at=M_inf, no areal
  conversion) showed almost no effect versus bulk_only -- tracing through
  why revealed a closed-form bifurcation: Gamma(T)*T (Eq. A.10's
  consumption) saturates to a *finite ceiling* as T -> infinity
  (numerator/(alpha*K_AT + AT*beta), using `thrombin_inhibition_rate`'s own
  formula), so whenever total thrombin production (bulk + surface-flux)
  exceeds that ceiling, [T] must diverge in ANY correct integrator -- this
  is a property of the equations, not a numerical artifact. Where that
  ceiling sits relative to production depends heavily on M_at's magnitude
  and on whether `coupled_solver.py`'s CM2_TO_M2 (1e-4) areal SI conversion
  is applied (production's FEM Neumann-flux pathway effectively multiplies
  the native M_at, in PLT/cm^2, by 1/CM2_TO_M2 = 1e4 before combining it
  with phi_at/phi_rt/PT) -- so this script tests representative M_at values
  at BOTH native and SI-converted magnitude to show where the threshold
  actually falls, since a 0-D isolation has no mesh to derive it from
  first-principles geometry.

Each scenario is integrated two ways over the identical RHS function:

(a) `species_transport.reaction_step`, called exactly as `coupled_solver.py`
    calls it -- once per macro step of `dt_s`, with its default
    `n_substeps=10`, `n_newton_iters=6`.
(b) `scipy.integrate.solve_ivp` (Radau, tight rtol/atol) as an independent,
    high-accuracy reference integrator, on the exact same RHS function,
    evaluated at the same output times.

Run: `python scripts/diagnose_thrombin_reaction_stiffness.py`
(from `thrombus-benchmark/`, with the `thrombus-bench` conda env active).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.integrate import solve_ivp

from thrombus_bench.mechanistic import activation, fibrin
from thrombus_bench.mechanistic.species_transport import reaction_step

_SPECIES_ORDER = ("RP", "AP", "APR", "APS", "T", "AT", "PT", "FG", "FI")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_configs() -> tuple[dict, dict]:
    with open(os.path.join(_REPO_ROOT, "configs", "physio_params.yaml")) as f:
        physio = yaml.safe_load(f)
    with open(os.path.join(_REPO_ROOT, "configs", "geometry.yaml")) as f:
        geometry = yaml.safe_load(f)
    return physio, geometry


def _representative_shear_rate_s(physio: dict, geometry: dict) -> float:
    """gamma_w_ref = 6*v_mean/D for the aneurysm_7mm preset (planar
    Poiseuille wall shear rate, matching `coupled_solver.py`'s own
    `gamma_w_ref` formula) -- a representative, physiologically-plausible
    local wall shear rate, well below `gamma_crit` (10000 /s) so mechanical
    activation is inactive here and only chemical (Omega-driven) activation
    is exercised, isolating the thrombin/fibrin pathway under test."""

    preset = geometry["presets"][geometry["default_preset"]]
    v_mean_m_s = preset["inlet_velocity_cm_s"] * 1.0e-2
    d_m = preset["vessel_diameter_mm"] * 1.0e-3
    return 6.0 * v_mean_m_s / d_m


def _initial_concentrations(physio: dict) -> dict[str, float]:
    """Matches `coupled_solver.run_coupled_simulation`'s initial state:
    uniform inlet values everywhere except T=0."""

    sp = physio["species"]
    return {
        "RP": sp["resting_platelets_inlet_plt_ml"],
        "AP": sp["resting_platelets_inlet_plt_ml"] * sp["activated_platelets_inlet_fraction"],
        "APR": 0.0,
        "APS": 0.0,
        "T": 0.0,
        "AT": sp["antithrombin_inlet_uM"],
        "PT": sp["prothrombin_inlet_uM"],
        "FG": sp["fibrinogen_inlet_uM"],
        "FI": sp["fibrin_inlet_uM"],
    }


# Surface-flux scenarios to test, beyond `bulk_only`. Each entry is
# (label, M_at_plt_cm2 (native units, or None to disable the flux term
# entirely), apply_si_areal_conversion). `apply_si_areal_conversion=True`
# multiplies M_at by 1/CM2_TO_M2 (1e4) before combining with phi_at/phi_rt/PT
# -- matching what `coupled_solver.py` actually feeds its FEM Neumann-flux
# assembly (`M_at_si = surface_state.M_at / CM2_TO_M2`), as opposed to using
# the native (PLT/cm^2) magnitude directly, which is what a naive first pass
# at this diagnostic assumed. See module docstring.
CM2_TO_M2 = 1.0e-4


def _surface_flux_scenarios(physio: dict) -> list[tuple[str, float, bool]]:
    M_inf = physio["adhesion_aggregation"]["M_inf_plt_cm2"]
    M_at_critical = physio["adhesion_aggregation"]["M_at_critical_plt_cm2"]
    return [
        ("flux_M_inf_native_units", M_inf, False),
        ("flux_M_inf_SI_converted", M_inf, True),
        ("flux_M_at_critical_SI_converted", M_at_critical, True),
    ]


def make_source_fn(physio: dict, gamma_nodal: float, M_at_plt_cm2: float | None, apply_si_conversion: bool):
    """Reproduces `coupled_solver.run_coupled_simulation`'s inner
    `source_fn` closure exactly (same calls, same parameter order), operating
    on 1-element arrays (single node, no spatial transport). If
    `M_at_plt_cm2` is not None, adds the C.5-C.6-shaped term described in
    this module's docstring, using that representative M_at magnitude
    (optionally SI-areal-converted, per `apply_si_conversion`); M_r is fixed
    at 0 (matches `surface_ode.py`'s permanent M_r=0 under theta=1)."""

    t_act_s = physio["sorensen_chemical"]["t_act_s"]
    steepness_omega = physio["activation"]["smoothing"]["steepness_omega"]
    gamma_crit = physio["activation"]["shear_rate_critical_s"]
    heparin_uM = physio["heparin"]["concentration_uM"]

    phi_at = physio["sorensen_chemical"]["phi_at_U_plt_s_uMPT"]
    phi_rt = physio["sorensen_chemical"]["phi_rt_U_plt_s_uMPT"]
    beta = physio["sorensen_chemical"]["beta_nmol_per_U"]
    M_r_representative = 0.0
    if M_at_plt_cm2 is not None:
        M_at_representative = M_at_plt_cm2 / CM2_TO_M2 if apply_si_conversion else M_at_plt_cm2
    else:
        M_at_representative = None

    def source_fn(conc: dict) -> dict:
        omega = activation.activation_function_omega(
            {"ADP": conc["APR"], "TxA2": conc["APS"]},
            {"ADP": physio["activation"]["adp_critical_uM"], "TxA2": physio["activation"]["txa2_critical_uM"]},
            {"ADP": physio["sorensen_chemical"]["agonist_weight_wj"], "TxA2": physio["sorensen_chemical"]["agonist_weight_wj"]},
        )
        k_chem = activation.chemical_activation_rate(omega, t_act_s, steepness_omega)
        k_mech = activation.mechanical_activation_rate(gamma_nodal, gamma_crit, t_act_s, steepness_omega)
        k_pa = activation.total_activation_rate(k_chem, k_mech)
        gamma_inh = activation.thrombin_inhibition_rate(
            heparin_uM, conc["T"], conc["AT"],
            physio["sorensen_chemical"]["k1_T_per_s"], physio["sorensen_chemical"]["K_AT_uM"],
            physio["sorensen_chemical"]["K_T_uM"], physio["sorensen_chemical"]["alpha"],
            physio["sorensen_chemical"]["beta_nmol_per_U"],
        )
        S = activation.chemical_source_terms(
            conc["RP"], conc["AP"], conc["APR"], conc["APS"], conc["PT"], conc["T"], conc["AT"],
            k_pa, gamma_inh,
            physio["sorensen_chemical"]["lambda_ADP_nmol_plt"], physio["sorensen_chemical"]["k1_ADP_per_s"],
            physio["sorensen_chemical"]["s_p_TxA2_nmol_plt_s"], physio["sorensen_chemical"]["k1_TxA2_per_s"],
            physio["sorensen_chemical"]["phi_at_U_plt_s_uMPT"], physio["sorensen_chemical"]["phi_rt_U_plt_s_uMPT"],
            physio["sorensen_chemical"]["beta_nmol_per_U"],
        )
        S_FG, S_FI = fibrin.fibrin_source_terms(
            conc["T"], conc["FG"], physio["fibrin"]["k_fi_th_per_s"], physio["fibrin"]["k_mfi_th_uM"]
        )
        S["FG"], S["FI"] = S_FG, S_FI

        if M_at_representative is not None:
            thrombin_gen = (M_at_representative * phi_at + M_r_representative * phi_rt) * conc["PT"]
            S["PT"] = S["PT"] - beta * thrombin_gen
            S["T"] = S["T"] + thrombin_gen

        return S

    return source_fn


def _run_reaction_step(source_fn, initial: dict[str, float], dt_s: float, n_steps: int) -> np.ndarray:
    """Advances `reaction_step` (production's implicit backward-Euler +
    Newton substepping, default n_substeps=10/n_newton_iters=6) once per
    macro step of `dt_s`, exactly as `coupled_solver.py` calls it. Returns
    an (n_steps+1, 9) trajectory array ordered per `_SPECIES_ORDER`."""

    conc = {name: np.array([initial[name]]) for name in _SPECIES_ORDER}
    trajectory = [np.array([conc[name][0] for name in _SPECIES_ORDER])]
    diverged = False
    for _ in range(n_steps):
        if diverged:
            trajectory.append(np.full(len(_SPECIES_ORDER), np.nan))
            continue
        try:
            conc = reaction_step(conc, dt_s, source_fn)
            row = np.array([conc[name][0] for name in _SPECIES_ORDER])
        except (FloatingPointError, np.linalg.LinAlgError):
            diverged = True
            row = np.full(len(_SPECIES_ORDER), np.nan)
        if not np.all(np.isfinite(row)):
            diverged = True
        trajectory.append(row)
    return np.array(trajectory)


def _run_solve_ivp(source_fn, initial: dict[str, float], t_eval: np.ndarray) -> np.ndarray:
    """Independent high-accuracy stiff reference integration (Radau) of the
    identical RHS function, evaluated at the same output times as the
    reaction_step trajectory."""

    y0 = np.array([initial[name] for name in _SPECIES_ORDER])

    def rhs(t, y):
        conc = {name: np.array([y[i]]) for i, name in enumerate(_SPECIES_ORDER)}
        S = source_fn(conc)
        return np.array([S[name][0] for name in _SPECIES_ORDER])

    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), y0, method="Radau", t_eval=t_eval,
        rtol=1e-10, atol=1e-12, max_step=t_eval[-1] / 200.0,
    )
    out = np.full((len(t_eval), len(_SPECIES_ORDER)), np.nan)
    out[: sol.y.shape[1]] = sol.y.T
    if not sol.success:
        print(f"    ! solve_ivp stopped early ({sol.message}); remaining time points shown as NaN "
              f"(reached t={sol.t[-1]:.3g}s of {t_eval[-1]:.3g}s -- itself informative: the reference "
              f"integrator's own step size collapsed, consistent with a genuine finite-time-blowup-like growth).")
    return out  # (len(t_eval), 9)


def _print_table(label: str, t: np.ndarray, traj_newton: np.ndarray, traj_ivp: np.ndarray) -> None:
    i_T, i_FI = _SPECIES_ORDER.index("T"), _SPECIES_ORDER.index("FI")
    print(f"\n--- {label}: [T] and [FI] (uM) ---")
    print(f"{'t (s)':>8} | {'T (reaction_step)':>18} | {'T (solve_ivp)':>15} | {'FI (reaction_step)':>19} | {'FI (solve_ivp)':>15}")
    for k in range(len(t)):
        print(
            f"{t[k]:8.2f} | {traj_newton[k, i_T]:18.4g} | {traj_ivp[k, i_T]:15.4g} | "
            f"{traj_newton[k, i_FI]:19.4g} | {traj_ivp[k, i_FI]:15.4g}"
        )


def _relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def main() -> None:
    physio, geometry = _load_configs()
    gamma_nodal = _representative_shear_rate_s(physio, geometry)
    initial = _initial_concentrations(physio)

    dt_s = 0.1  # matches CLAUDE.md's example --dt-s 0.1 macro step
    end_time_s = 3.0
    n_steps = int(round(end_time_s / dt_s))
    t_eval = np.linspace(0.0, end_time_s, n_steps + 1)

    print("=" * 88)
    print("Isolated local-reaction-ODE diagnostic (no spatial transport)")
    print("=" * 88)
    print(f"Representative wall shear rate gamma_w_ref = {gamma_nodal:.2f} /s "
          f"(gamma_crit = {physio['activation']['shear_rate_critical_s']:.0f} /s -> mechanical activation inactive)")
    print(f"Initial concentrations: {initial}")
    print(f"dt_s = {dt_s}, end_time_s = {end_time_s}, macro steps = {n_steps}")

    scenario_names = ["bulk_only"] + [s[0] for s in _surface_flux_scenarios(physio)]
    scenario_args = {"bulk_only": (None, False)}
    for name, M_at, si in _surface_flux_scenarios(physio):
        scenario_args[name] = (M_at, si)

    results = {}
    for case_name in scenario_names:
        M_at_plt_cm2, apply_si = scenario_args[case_name]
        source_fn = make_source_fn(physio, gamma_nodal, M_at_plt_cm2, apply_si)
        traj_newton = _run_reaction_step(source_fn, initial, dt_s, n_steps)
        traj_ivp = _run_solve_ivp(source_fn, initial, t_eval)
        results[case_name] = (traj_newton, traj_ivp)
        _print_table(case_name, t_eval, traj_newton, traj_ivp)

    # --- Plain-language diagnostic report ---
    i_T, i_FI = _SPECIES_ORDER.index("T"), _SPECIES_ORDER.index("FI")
    print("\n" + "=" * 88)
    print("DIAGNOSTIC REPORT")
    print("=" * 88)

    for case_name in scenario_names:
        traj_newton, traj_ivp = results[case_name]
        final_T_newton, final_T_ivp = traj_newton[-1, i_T], traj_ivp[-1, i_T]
        final_FI_newton, final_FI_ivp = traj_newton[-1, i_FI], traj_ivp[-1, i_FI]
        newton_diverged = not np.isfinite(final_T_newton)
        ivp_diverged = not np.isfinite(final_T_ivp)
        print(f"\n[{case_name}] final t={end_time_s}s:")
        if newton_diverged or ivp_diverged:
            print(f"  reaction_step diverged/failed: {newton_diverged}  |  solve_ivp diverged/failed: {ivp_diverged}")
            # Report the last finite values reached, and when.
            for label, traj in (("reaction_step", traj_newton), ("solve_ivp", traj_ivp)):
                finite_mask = np.isfinite(traj[:, i_T])
                if finite_mask.any():
                    last_idx = np.where(finite_mask)[0][-1]
                    print(f"    {label}: last finite [T]={traj[last_idx, i_T]:.4g} uM, [FI]={traj[last_idx, i_FI]:.4g} uM at t={t_eval[last_idx]:.2f}s")
        else:
            rel_T = _relative_diff(final_T_newton, final_T_ivp)
            rel_FI = _relative_diff(final_FI_newton, final_FI_ivp)
            print(f"  T:  reaction_step={final_T_newton:.4g} uM  vs  solve_ivp={final_T_ivp:.4g} uM  (rel. diff {rel_T:.3f})")
            print(f"  FI: reaction_step={final_FI_newton:.4g} uM  vs  solve_ivp={final_FI_ivp:.4g} uM  (rel. diff {rel_FI:.3f})")

    plot_path = os.path.join(_REPO_ROOT, "scripts", "thrombin_reaction_stiffness_diagnostic.png")
    _make_plot(t_eval, results, scenario_names, plot_path)
    print(f"\nSaved comparison plot to {plot_path}")


def _make_plot(t_eval: np.ndarray, results: dict, scenario_names: list[str], path: str) -> None:
    i_T, i_FI = _SPECIES_ORDER.index("T"), _SPECIES_ORDER.index("FI")
    n = len(scenario_names)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 7), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)
    for col, case_name in enumerate(scenario_names):
        traj_newton, traj_ivp = results[case_name]
        axes[0, col].plot(t_eval, traj_newton[:, i_T], "o-", label="reaction_step (production)", color="tab:blue", markersize=3)
        axes[0, col].plot(t_eval, traj_ivp[:, i_T], "--", label="solve_ivp (Radau, reference)", color="tab:red")
        axes[0, col].set_title(f"{case_name}\n[T]", fontsize=9)
        axes[0, col].set_ylabel("[T] (uM)")
        axes[0, col].set_yscale("symlog")
        axes[0, col].legend(fontsize=7)

        axes[1, col].plot(t_eval, traj_newton[:, i_FI], "o-", label="reaction_step (production)", color="tab:blue", markersize=3)
        axes[1, col].plot(t_eval, traj_ivp[:, i_FI], "--", label="solve_ivp (Radau, reference)", color="tab:red")
        axes[1, col].set_title("[FI]", fontsize=9)
        axes[1, col].set_ylabel("[FI] (uM)")
        axes[1, col].set_yscale("symlog")
        axes[1, col].set_xlabel("time (s)")
        axes[1, col].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
