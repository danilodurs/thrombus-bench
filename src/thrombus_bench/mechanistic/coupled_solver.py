"""Monolithic time-stepping coupling loop for the full mechanistic model.

Responsibility
---------------
Couples `flow_solver.py` (Stokes + Carreau viscosity), `species_transport.py`
(9-species CDR via SUPG transport + implicit local reaction substeps),
`activation.py` (chemical + mechanical activation source terms),
`surface_ode.py` (wall surface coverage ODEs), and `fibrin.py` (fibrin
generation) into the transient "Navier Stokes-CDR-ODE" system described in
Sec. 2.6-2.7 of the paper.

Coupling scheme, per macro time step dt
-------------------------------------------
1. Project the current flow field's shear rate onto a nodal (P1) field and
   read off wall-node values; split the wall into "top" (vessel top wall +
   aneurysm sac, y > vessel_diameter/2) and "bottom" branches, sorted by x,
   and finite-difference the axial gradient d(gamma_w)/dx within each
   branch (the mechanically-relevant negative-shear-gradient regions are
   concentrated on the top/aneurysm wall, per the paper's Fig. 3/5/9-11;
   this branch split avoids interleaving two different wall curves that
   happen to share x-coordinates -- see this module's `_wall_branches`
   docstring).
2. Advance the surface ODE state (`surface_ode.py`) explicitly (substepped)
   using the current wall species concentrations and shear field.
3. Advance species concentrations one macro step via Strang-split
   reaction (`species_transport.reaction_step`, using `activation.py` +
   `fibrin.py` source terms) then transport (`species_transport.solve_transport_step`,
   with Appendix C wall flux Neumann BCs for RP/AP/PT/T built from the
   just-updated surface state).
4. Recompute the Carreau viscosity field including the Eq. (18) thrombus
   multiplier (`flow_solver.viscosity_multiplier`) from the updated M_at
   and FI fields, and re-solve the flow field
   (`flow_solver.solve_steady_flow`).
5. Repeat until `end_time_s`, recording state at the requested output
   cadence.

This is a first-order (Lie/Strang-lite) fixed-point operator-splitting
scheme across physics (flow / surface / species), distinct from the Picard
iteration *within* `flow_solver.py` (Carreau nonlinearity) and the Newton
iteration *within* `species_transport.reaction_step` (stiff local
kinetics). See README.md "Assumptions & Deviations from Source Paper" for
the numerical-method caveats this entails.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skfem import BilinearForm, Basis, ElementTriP1, ElementVector, ElementTriP2, FacetBasis, LinearForm, solve

from . import activation, fibrin, surface_ode
from .flow_solver import CarreauParams, FlowSolution, ThrombusViscosityFields, shear_rate, solve_steady_flow
from .mesh import TaggedMesh
from .species_transport import WALL_REACTIVE_SPECIES, reaction_step, solve_transport_step

_ALL_SPECIES = ("RP", "AP", "APR", "APS", "T", "AT", "PT", "FG", "FI")


@dataclass
class CoupledSimulationState:
    """Full model state at a single time point."""

    time_s: float
    flow: FlowSolution
    concentrations: dict  # species name -> nodal (P1) array
    surface: surface_ode.SurfaceState
    wall_dofs: np.ndarray


@dataclass
class CoupledSimulationHistory:
    """Checkpointed states over the run, plus the (fixed) mesh/basis used."""

    tagged_mesh: TaggedMesh
    basis_c: Basis
    states: list = field(default_factory=list)  # list[CoupledSimulationState]


def _wall_branches(basis_c: Basis, vessel_diameter_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Split wall DOFs into top (vessel top wall + aneurysm sac) and bottom
    branches, each sorted by x-coordinate, so an axial (streamwise) gradient
    can be computed unambiguously within each (see module docstring)."""

    wall_names = [n for n in ("wall_vessel", "wall_sac") if len(basis_c.mesh.boundaries.get(n, [])) > 0]
    wall_dofs = np.unique(np.concatenate([basis_c.get_dofs(n).all() for n in wall_names]))
    coords = basis_c.doflocs[:, wall_dofs]
    top_mask = coords[1] > 0.5 * vessel_diameter_m
    top = wall_dofs[top_mask][np.argsort(coords[0, top_mask])]
    bottom = wall_dofs[~top_mask][np.argsort(coords[0, ~top_mask])]
    return top, bottom


def _axial_shear_gradient(
    basis_c: Basis, wall_dofs: np.ndarray, shear_nodal: np.ndarray, top_branch: np.ndarray, bottom_branch: np.ndarray
) -> np.ndarray:
    """d(gamma_w)/dx at every wall DOF, computed independently within the
    top and bottom branches via `numpy.gradient` (handles the non-uniform
    node spacing from `mesh.py`'s neck-refined mesh)."""

    grad_full = np.zeros(basis_c.N)
    for branch in (top_branch, bottom_branch):
        if len(branch) < 2:
            continue
        x = basis_c.doflocs[0, branch]
        g = shear_nodal[branch]
        grad_full[branch] = np.gradient(g, x)
    return grad_full[wall_dofs]


def _wall_shear_rate_nodal(basis_c: Basis, flow: FlowSolution) -> np.ndarray:
    """L2-project the quadrature-point shear rate field (Eq. 2) onto the
    nodal P1 basis used for species/surface state.

    `skfem.project` expects `fun` to be a DOF vector in a *source* function
    space (it internally does `mass(basis_from, basis_to) @ fun`), not a raw
    quadrature-point array -- so the projection is assembled directly here:
    interpolate velocity at `basis_c`'s own quadrature points (via a
    matching-quadrature vector basis), then solve `M @ gamma_nodal = f` for
    the standard L2 projection mass matrix `M` and load vector
    `f = integral(gamma_quad * v) dx`.
    """

    vel_basis_matching = basis_c.with_element(ElementVector(ElementTriP2()))
    gamma_quad = shear_rate(vel_basis_matching.interpolate(flow.u))

    @BilinearForm
    def mass(u, v, w):
        return u * v

    @LinearForm
    def load(v, w):
        return w["gamma"] * v

    M = mass.assemble(basis_c)
    f = load.assemble(basis_c, gamma=gamma_quad)
    return solve(M, f)


def run_coupled_simulation(
    tagged_mesh: TaggedMesh,
    inlet_velocity_m_s: float,
    physio: dict,
    end_time_s: float,
    dt_s: float,
    output_every_n_steps: int = 1,
    flow_resolve_every_n_steps: int = 1,
    surface_substeps: int = 20,
) -> CoupledSimulationHistory:
    """Run the coupled model from t=0 to `end_time_s`. `physio` is
    `configs/physio_params.yaml`'s top-level dict (fluid/species/activation/
    adhesion_aggregation/fibrin/thrombus_growth/scale_terms/heparin/
    sorensen_chemical sections)."""

    mesh = tagged_mesh.mesh
    basis_c = Basis(mesh, ElementTriP1())

    carreau = CarreauParams.from_config(physio["fluid"]["carreau"])
    vessel_diameter_m = tagged_mesh.geometry.vessel_diameter_mm * 1.0e-3
    # L (Eqs. 6-7 scale factor) enters only the dimensionless
    # (L/gamma_w) * d(gamma_w)/dx ratio, and d(gamma_w)/dx is computed from
    # this mesh's meter-based node coordinates (`_axial_shear_gradient`) --
    # so L must be in meters here too, *not* the paper's native centimeters
    # (Table 1 defines L = 10 * entrance diameter with no unit conversion
    # implied; using centimeters here against a meter-based gradient would
    # silently inflate the mechanical flux ~100x). k_rs/k_as/k_aa/M_inf/
    # M_at_critical below are left in the paper's native cm-based units,
    # self-consistent within `surface_ode.py`'s own dimensionless M/M_inf
    # ratios; only the wall Neumann flux handed to the (meter-based)
    # `species_transport.solve_transport_step` needs an explicit cm^2 -> m^2
    # areal-flux conversion, applied where that flux is assembled below.
    L_m = physio["scale_terms"]["L_entrance_diameter_multiplier"] * vessel_diameter_m
    # Unit reconciliation for the wall Neumann flux fed into
    # species_transport.solve_transport_step (whose weak form is assembled
    # against this meter-based mesh, with bulk species concentrations left
    # in their native, unconverted units -- PLT/mL for platelets, uM for
    # everything else -- since the reaction kinetics in activation.py are
    # calibrated for exactly those units, Table D.2).
    #
    # Two structurally different quantities appear in the Appendix C flux
    # formulas, and they convert differently:
    #  - Rate constants with velocity units (k_rs/k_as/k_aa, cm/s) multiply
    #    a *volumetric* bulk concentration (PLT/mL = PLT/cm^3, left native).
    #    Since the bulk transport weak form's mass/advection/diffusion
    #    matrices are assembled with this same native-unit concentration
    #    against a meter-based mesh, only the rate constant's length scale
    #    needs converting: cm/s -> m/s (x CM_TO_M).
    #  - Surface-density quantities (M, M_r, M_as, M_at, PLT/cm^2) that
    #    appear as a *bare* (non-ratio) factor -- i.e. not divided by
    #    another same-unit quantity like M_inf -- need their own areal unit
    #    converted: PLT/cm^2 -> PLT/m^2 (x 1/CM2_TO_M2). Ratios such as
    #    M/M_inf or M_at/M_inf are scale-invariant (both PLT/cm^2) and need
    #    no conversion at all.
    CM_TO_M = 1.0e-2
    CM2_TO_M2 = 1.0e-4
    D_a = physio["scale_terms"]["D_a"]
    M_inf = physio["adhesion_aggregation"]["M_inf_plt_cm2"]
    steepness_omega = physio["activation"]["smoothing"]["steepness_omega"]
    steepness_theta = physio["activation"]["smoothing"]["steepness_theta"]
    gamma_crit = physio["activation"]["shear_rate_critical_s"]
    t_act_s = physio["sorensen_chemical"]["t_act_s"]
    heparin_uM = physio["heparin"]["concentration_uM"]

    diffusivity_m2_s = {
        "RP": physio["species"]["diffusion_cm2_s"]["platelets"] * 1e-4,
        "AP": physio["species"]["diffusion_cm2_s"]["platelets"] * 1e-4,
        "APR": physio["species"]["diffusion_cm2_s"]["ADP"] * 1e-4,
        "APS": physio["species"]["diffusion_cm2_s"]["TxA2"] * 1e-4,
        "T": physio["species"]["diffusion_cm2_s"]["thrombin"] * 1e-4,
        "AT": physio["species"]["diffusion_cm2_s"]["antithrombin"] * 1e-4,
        "PT": physio["species"]["diffusion_cm2_s"]["prothrombin"] * 1e-4,
        "FG": physio["species"]["diffusion_cm2_s"]["platelets"] * 1e-4,  # not tabulated; platelet-order fallback, see README
        "FI": physio["species"]["diffusion_cm2_s"]["platelets"] * 1e-4,
    }
    inlet_value = {
        "RP": physio["species"]["resting_platelets_inlet_plt_ml"],
        "AP": physio["species"]["resting_platelets_inlet_plt_ml"] * physio["species"]["activated_platelets_inlet_fraction"],
        "APR": 0.0,
        "APS": 0.0,
        "T": physio["species"]["thrombin_initial_uM"],
        "AT": physio["species"]["antithrombin_inlet_uM"],
        "PT": physio["species"]["prothrombin_inlet_uM"],
        "FG": physio["species"]["fibrinogen_inlet_uM"],
        "FI": physio["species"]["fibrin_inlet_uM"],
    }
    # Generous numerical safety caps -- see the "Known limitation" comment
    # at their point of use below.
    concentration_cap = {
        "RP": inlet_value["RP"] * 2.0,
        "AP": inlet_value["RP"] * 2.0,
        "APR": physio["activation"]["adp_critical_uM"] * 1000.0,
        "APS": physio["activation"]["txa2_critical_uM"] * 1000.0,
        "T": 1000.0,
        "AT": inlet_value["AT"] * 2.0,
        "PT": inlet_value["PT"] * 2.0,
        "FG": inlet_value["FG"] * 2.0,
        "FI": inlet_value["FG"] * 2.0,
    }

    concentrations = {name: np.full(basis_c.N, inlet_value[name]) for name in _ALL_SPECIES}
    concentrations["T"] = np.zeros(basis_c.N)  # thrombin initial = 0 throughout, not just at inlet

    top_branch, bottom_branch = _wall_branches(basis_c, vessel_diameter_m)
    wall_dofs = np.union1d(top_branch, bottom_branch)
    surface_state = surface_ode.SurfaceState.zeros_like(wall_dofs)

    flow = solve_steady_flow(tagged_mesh, inlet_velocity_m_s, carreau)
    history = CoupledSimulationHistory(tagged_mesh=tagged_mesh, basis_c=basis_c, states=[])

    # Fixed reference wall shear rate "in the straight vessel segment
    # upstream of the aneurysm" (Sec. 2.4), used as the L/gamma_w scale
    # factor's denominator (Eqs. 6-7) -- a single scalar, not a per-node
    # field; see surface_ode.mechanical_flux_resting docstring. Computed
    # analytically from fully-developed planar Poiseuille flow,
    # gamma_w = 6 * v_mean / D, since it only needs to be representative of
    # the undisturbed upstream vessel, not the locally-resolved flow field.
    gamma_w_ref = 6.0 * inlet_velocity_m_s / vessel_diameter_m

    n_steps = int(round(end_time_s / dt_s))
    for step in range(n_steps):
        t = step * dt_s

        gamma_nodal = _wall_shear_rate_nodal(basis_c, flow)
        d_gamma_dx_wall = _axial_shear_gradient(basis_c, wall_dofs, gamma_nodal, top_branch, bottom_branch)

        # Substep the explicit-Euler surface-state update: a single step at
        # the full macro dt can overshoot M_inf badly before the saturation
        # term S=1-M/M_inf has a chance to act (the unsaturated initial rate
        # can be large relative to M_inf/dt) -- mirrors why
        # species_transport.reaction_step substeps its own implicit update.
        for _ in range(surface_substeps):
            rhs = surface_ode.surface_ode_rhs(
                surface_state,
                gamma_w_ref=gamma_w_ref,
                d_shear_rate_dx=d_gamma_dx_wall,
                resting_platelet_conc=concentrations["RP"][wall_dofs],
                activated_platelet_conc=concentrations["AP"][wall_dofs],
                k_rs_cm_s=physio["adhesion_aggregation"]["k_rs_cm_s"],
                k_as_cm_s=physio["adhesion_aggregation"]["k_as_cm_s"],
                k_aa_cm_s=physio["adhesion_aggregation"]["k_aa_cm_s"],
                M_inf_plt_cm2=M_inf,
                L_m=L_m,
                D_a=D_a,
            )
            surface_state = surface_ode.step_surface_state(surface_state, rhs, dt_s / surface_substeps)

        def source_fn(conc: dict, _gamma=gamma_nodal) -> dict:
            omega = activation.activation_function_omega(
                {"ADP": conc["APR"], "TxA2": conc["APS"]},
                {"ADP": physio["activation"]["adp_critical_uM"], "TxA2": physio["activation"]["txa2_critical_uM"]},
                {"ADP": physio["sorensen_chemical"]["agonist_weight_wj"], "TxA2": physio["sorensen_chemical"]["agonist_weight_wj"]},
            )
            k_chem = activation.chemical_activation_rate(omega, t_act_s, steepness_omega)
            k_mech = activation.mechanical_activation_rate(_gamma, gamma_crit, t_act_s, steepness_omega)
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
            return S

        concentrations = reaction_step(concentrations, dt_s, source_fn)

        # Wall flux for species_transport (mesh/SI-consistent, see the
        # CM_TO_M / CM2_TO_M2 note above) -- built directly here rather than
        # via surface_ode.chemical_flux_resting/activated, which use the
        # paper's native cm-based rate constants (correct for surface_ode's
        # own, mesh-independent M/M_r/M_as/M_at update, computed separately
        # via surface_ode.surface_ode_rhs above).
        k_rs_si = physio["adhesion_aggregation"]["k_rs_cm_s"] * CM_TO_M
        k_as_si = physio["adhesion_aggregation"]["k_as_cm_s"] * CM_TO_M
        k_aa_si = physio["adhesion_aggregation"]["k_aa_cm_s"] * CM_TO_M
        S = surface_ode.saturation_term(surface_state.M, M_inf)  # dimensionless, unit-independent

        j_r_chem_si = S * k_rs_si * concentrations["RP"][wall_dofs]
        j_a_chem_si = (S * k_as_si + (surface_state.M_at / M_inf) * k_aa_si) * concentrations["AP"][wall_dofs]

        phi_at = physio["sorensen_chemical"]["phi_at_U_plt_s_uMPT"]
        phi_rt = physio["sorensen_chemical"]["phi_rt_U_plt_s_uMPT"]
        beta = physio["sorensen_chemical"]["beta_nmol_per_U"]
        PT_wall = concentrations["PT"][wall_dofs]
        M_at_si = surface_state.M_at / CM2_TO_M2  # PLT/cm^2 -> PLT/m^2 (bare, non-ratio use)
        M_r_si = surface_state.M_r / CM2_TO_M2
        thrombin_gen_si = (M_at_si * phi_at + M_r_si * phi_rt) * PT_wall
        j_pt_chem_si = beta * thrombin_gen_si  # Eq. (C.5): consumption of prothrombin
        j_t_chem_si = -thrombin_gen_si  # Eq. (C.6): generation of thrombin

        wall_flux_nodal = {"RP": j_r_chem_si, "AP": j_a_chem_si, "PT": j_pt_chem_si, "T": j_t_chem_si}

        bc_vec = basis_c.with_element(ElementVector(ElementTriP2()))
        velocity_field = bc_vec.interpolate(flow.u)

        wall_names = [n for n in ("wall_vessel", "wall_sac") if len(mesh.boundaries.get(n, [])) > 0]
        fb = FacetBasis(mesh, basis_c.elem, facets=wall_names)

        new_concentrations = {}
        for name in _ALL_SPECIES:
            wf = None
            if name in WALL_REACTIVE_SPECIES:
                # Scatter the wall-only flux onto a full nodal field (zero
                # off the wall) and let the FacetBasis's own interpolation
                # read it off at the wall quadrature points -- FacetBasis
                # shares the parent CellBasis's element/dof numbering, so no
                # cross-basis projection is needed.
                nodal_flux = np.zeros(basis_c.N)
                nodal_flux[wall_dofs] = wall_flux_nodal[name]
                wf = fb.interpolate(nodal_flux)
            new_concentrations[name] = solve_transport_step(
                basis_c, tagged_mesh, velocity_field, concentrations[name],
                diffusivity_m2_s[name], dt_s, inlet_value[name], wall_flux=wf,
            )
        # Known limitation: the surface thrombin-generation flux (Eqs.
        # C.5-C.6, via phi_at/phi_rt on M_at/M_r) combined with Eq. (A.10)'s
        # Gamma -- which *saturates* rather than accelerates as [T] grows,
        # since T appears in Gamma's denominator -- produces unbounded
        # positive-feedback growth in [T] (and, downstream, [FI]) over
        # timescales of seconds in this coupling scheme, well beyond
        # physiological ranges (~0.1-10 uM). This was not resolved by the
        # unit reconciliation applied elsewhere in this module (verified
        # correct via an isolated wall-flux mass-balance check) and likely
        # reflects either a further scaling issue specific to the C.5-C.6
        # pathway or a genuine sensitivity of the model as reconstructed
        # (see surface_ode.py's "Reconstruction note") that would need
        # comparison against the paper's own (unavailable) reference
        # implementation to fully resolve. A generous concentration cap is
        # applied as a numerical safety net so simulations remain bounded
        # and finite rather than diverging; see README.md "Assumptions &
        # Deviations from Source Paper" / "Known limitations".
        concentrations = {
            k: np.clip(v, 0.0, concentration_cap[k]) for k, v in new_concentrations.items()
        }

        if (step + 1) % flow_resolve_every_n_steps == 0 or step == n_steps - 1:
            # M_at lives only on wall DOFs (surface_ode.py); scatter onto a
            # full nodal field (zero in the bulk) for Eq. (18)'s viscosity
            # multiplier. FI is already a bulk species field.
            M_at_nodal_full = np.zeros(basis_c.N)
            M_at_nodal_full[wall_dofs] = surface_state.M_at
            thrombus_fields = ThrombusViscosityFields(
                M_at_nodal=M_at_nodal_full,
                FI_nodal=concentrations["FI"],
                M_at_critical_plt_cm2=physio["adhesion_aggregation"]["M_at_critical_plt_cm2"],
                fibrin_critical_uM=physio["fibrin"]["fibrin_critical_uM"],
                steepness_theta=steepness_theta,
                multiplier_max=physio["thrombus_growth"]["viscosity_multiplier_single_threshold"],
            )
            flow = solve_steady_flow(tagged_mesh, inlet_velocity_m_s, carreau, thrombus_fields=thrombus_fields)

        if step % output_every_n_steps == 0 or step == n_steps - 1:
            history.states.append(
                CoupledSimulationState(
                    time_s=t + dt_s,
                    flow=flow,
                    concentrations={k: v.copy() for k, v in concentrations.items()},
                    surface=surface_ode.SurfaceState(
                        M=surface_state.M.copy(), M_r=surface_state.M_r.copy(),
                        M_as=surface_state.M_as.copy(), M_at=surface_state.M_at.copy(),
                    ),
                    wall_dofs=wall_dofs,
                )
            )

    return history
