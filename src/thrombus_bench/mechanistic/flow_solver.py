"""Blood flow solver: non-Newtonian (Carreau) steady/quasi-steady Stokes flow.

Responsibility
---------------
Solve the blood flow field used by every other mechanistic module (species
transport advects with this velocity; adhesion/aggregation flux BCs use the
resulting wall shear rate and its axial gradient). Implements:

* Eq. (2): the Carreau generalized-Newtonian viscosity closure
  ``mu(gamma_dot) = mu_inf + (mu0 - mu_inf) * (1 + (lambda*gamma_dot)^2)^((n-1)/2)``.
* Eq. (18): the thrombus-feedback viscosity multiplier
  ``Theta_plts(M_at) + Theta_FI(FI)``, applied multiplicatively on top of the
  Carreau closure once a thrombus is present (used by ``coupled_solver.py``;
  defaults to a multiplier of 1 -- i.e. no thrombus -- for flow-only runs).
* Eqs. (3)-(4): momentum + incompressibility, ``rho*(du/dt + u.grad(u)) =
  -grad(p) + div(2*mu*sym_grad(u))``, ``div(u) = 0``.

Governing-equation simplification (explicit modeling assumption)
-------------------------------------------------------------------
This module solves the **steady, inertialess (Stokes) limit** of Eqs. (3)-(4)
with the nonlinear Carreau viscosity handled by Picard fixed-point iteration,
rather than the full unsteady Navier-Stokes system the paper solves in
COMSOL. This is a deliberate simplification to keep the solver simple enough
to run on CPU with a coarse mesh (per project scope), at the cost of
dropping convective/inertial effects (the paper reports Reynolds numbers of
455-908, i.e. not creeping flow -- see README "Assumptions & Deviations").
Pulsatile inlet conditions (Sec. 3.3.4) are supported as a **quasi-steady**
sequence of Stokes solves at each sampled point of the inlet waveform (valid
when transients equilibrate faster than the waveform period; the paper
reports a Womersley number of 2.75, so this is an approximation, not exact).

Discretization: Taylor-Hood P2(velocity)/P1(pressure) elements via
scikit-fem, no-slip walls, a uniform ("plug") prescribed inlet velocity per
Sec. 2.2 ("a prescribed constant velocity is imposed" at the inlet), and a
do-nothing (zero total traction, i.e. zero gauge pressure) natural outlet
condition approximating the paper's stated outlet BC (atmospheric/zero gauge
pressure, Sec. 2.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, ElementTriP2, bmat, condense, solve
from skfem.helpers import ddot, div, sym_grad

from .mesh import TaggedMesh


@dataclass
class CarreauParams:
    """Carreau non-Newtonian viscosity parameters, Eq. (2) / Table 1."""

    mu_inf_pa_s: float
    mu_0_pa_s: float
    lambda_s: float
    n: float

    def viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        return self.mu_inf_pa_s + (self.mu_0_pa_s - self.mu_inf_pa_s) * (
            1.0 + (self.lambda_s * gamma_dot) ** 2
        ) ** ((self.n - 1.0) / 2.0)

    @classmethod
    def from_config(cls, cfg: dict) -> "CarreauParams":
        return cls(
            mu_inf_pa_s=float(cfg["mu_inf_pa_s"]),
            mu_0_pa_s=float(cfg["mu_0_pa_s"]),
            lambda_s=float(cfg["lambda_s"]),
            n=float(cfg["n"]),
        )


@dataclass
class FlowSolution:
    """Converged velocity/pressure field on a `TaggedMesh`."""

    tagged_mesh: TaggedMesh
    basis_u: Basis
    basis_p: Basis
    u: np.ndarray  # velocity DOF vector (ElementVector(ElementTriP2))
    p: np.ndarray  # pressure DOF vector (ElementTriP1)
    inlet_velocity_m_s: float
    n_iterations: int
    converged: bool
    residual: float

    def shear_rate_at_quadrature(self) -> np.ndarray:
        """Generalized shear rate gamma_dot = sqrt(2 D:D) at quadrature points, Eq. (2)."""
        return shear_rate(self.basis_u.interpolate(self.u))

    def viscosity_at_quadrature(self, carreau: CarreauParams) -> np.ndarray:
        return carreau.viscosity(self.shear_rate_at_quadrature())


def shear_rate(velocity_field) -> np.ndarray:
    """Generalized shear rate gamma_dot = sqrt(2 D:D), D = sym_grad(u), Eq. (2).

    `velocity_field` is a scikit-fem `DiscreteField` (e.g. from
    `basis.interpolate(u_dofs)`); returns an array shaped like the field's
    quadrature-point grid (n_elements, n_quadrature_points).
    """

    d = sym_grad(velocity_field)
    return np.sqrt(np.maximum(2.0 * ddot(d, d), 0.0))


def _logistic_step(x, center: float, steepness: float, lo: float = 0.0, hi: float = 1.0):
    """Sigmoid approximation of a hard step at x=center.

    Uses the *relative* deviation (x/center - 1) so that `steepness` is a
    dimensionless knob usable regardless of x's absolute scale (M_at ~ 1e7,
    FI ~ 1e-1-1e0). Not specified by the source paper -- see README
    "Assumptions & Deviations from Source Paper".
    """

    return lo + (hi - lo) / (1.0 + np.exp(-steepness * (np.asarray(x) / center - 1.0)))


def viscosity_multiplier(
    M_at: np.ndarray,
    FI: np.ndarray,
    M_at_critical_plt_cm2: float,
    fibrin_critical_uM: float,
    steepness_theta: float,
    multiplier_max: float = 80.0,
) -> np.ndarray:
    """Eq. (18) thrombus viscosity multiplier: Theta_plts(M_at) + Theta_FI(FI).

    Theta_plts in [1, multiplier_max], Theta_FI in [0, multiplier_max]; sum
    ranges from 1 (no thrombus) to multiplier_max (one threshold exceeded) to
    2*multiplier_max (both exceeded), matching the paper's 1/80/160 values
    for the default `multiplier_max=80`.
    """

    theta_plts = _logistic_step(M_at, M_at_critical_plt_cm2, steepness_theta, lo=1.0, hi=multiplier_max)
    theta_fi = _logistic_step(FI, fibrin_critical_uM, steepness_theta, lo=0.0, hi=multiplier_max)
    return theta_plts + theta_fi


def _inlet_dofs(basis_u: Basis) -> tuple[np.ndarray, np.ndarray]:
    """(ux_dofs, uy_dofs) at the `inlet` boundary, both nodal and P2 edge dofs."""

    dofs = basis_u.get_dofs("inlet")
    ux = np.concatenate([dofs.nodal["u^1"], dofs.facet["u^1"]])
    uy = np.concatenate([dofs.nodal["u^2"], dofs.facet["u^2"]])
    return ux, uy


def _wall_dofs(basis_u: Basis) -> np.ndarray:
    names = [n for n in ("wall_vessel", "wall_sac") if len(basis_u.mesh.boundaries.get(n, [])) > 0]
    if not names:
        return np.array([], dtype=int)
    return np.unique(np.concatenate([basis_u.get_dofs(n).all() for n in names]))


@dataclass
class ThrombusViscosityFields:
    """Nodal (P1) M_at/FI fields over the *flow* mesh, feeding the Eq. (18)
    viscosity multiplier into a `solve_steady_flow` call. Nodal values away
    from the wall (where `surface_ode.py`'s M_at/FI live) should be filled
    with 0 -- the multiplier reduces to 1 (no thrombus) there automatically.
    """

    M_at_nodal: np.ndarray
    FI_nodal: np.ndarray
    M_at_critical_plt_cm2: float
    fibrin_critical_uM: float
    steepness_theta: float
    multiplier_max: float = 80.0


def solve_steady_flow(
    tagged_mesh: TaggedMesh,
    inlet_velocity_m_s: float,
    carreau: CarreauParams,
    picard_max_iter: int = 50,
    picard_tol: float = 1e-7,
    thrombus_fields: ThrombusViscosityFields | None = None,
) -> FlowSolution:
    """Solve steady, inertialess, Carreau-viscosity Stokes flow via Picard iteration.

    Parameters
    ----------
    tagged_mesh:
        Mesh with `inlet` / `outlet` / `wall_vessel` / `wall_sac` boundary
        tags, as produced by `mesh.build_channel_mesh` / `build_aneurysm_mesh`.
    inlet_velocity_m_s:
        Uniform (plug) inlet velocity in the +x direction, Sec. 2.2.
    carreau:
        Carreau viscosity parameters, Eq. (2).
    thrombus_fields:
        Optional `ThrombusViscosityFields` (nodal M_at/FI over this same
        mesh) folding the Eq. (18) thrombus viscosity multiplier into the
        Carreau closure. Used by `coupled_solver.py` once a thrombus is
        present; omitted (multiplier == 1 everywhere) for flow-only
        validation runs.
    """

    mesh = tagged_mesh.mesh
    basis_u = Basis(mesh, ElementVector(ElementTriP2()))
    basis_p = basis_u.with_element(ElementTriP1())

    ux_in, uy_in = _inlet_dofs(basis_u)
    wall_dofs = _wall_dofs(basis_u)
    dirichlet_dofs = np.unique(np.concatenate([wall_dofs, ux_in, uy_in]))

    x0 = np.zeros(basis_u.N + basis_p.N)
    x0[ux_in] = inlet_velocity_m_s

    if thrombus_fields is not None:
        M_at_quad = basis_p.interpolate(thrombus_fields.M_at_nodal)
        FI_quad = basis_p.interpolate(thrombus_fields.FI_nodal)
        multiplier_quad = viscosity_multiplier(
            M_at_quad, FI_quad,
            thrombus_fields.M_at_critical_plt_cm2, thrombus_fields.fibrin_critical_uM,
            thrombus_fields.steepness_theta, thrombus_fields.multiplier_max,
        )
    else:
        multiplier_quad = None

    @BilinearForm
    def a_visc(u, v, w):
        sg_prev = sym_grad(w["u_prev"])
        gamma_dot = np.sqrt(np.maximum(2.0 * ddot(sg_prev, sg_prev), 0.0))
        mu = carreau.viscosity(gamma_dot)
        if multiplier_quad is not None:
            mu = mu * multiplier_quad
        return 2.0 * mu * ddot(sym_grad(u), sym_grad(v))

    @BilinearForm
    def b_coupling(u, p, w):
        return -p * div(u)

    u_prev = basis_u.zeros()
    n_iterations = 0
    residual = np.inf
    converged = False
    xsol = np.concatenate([u_prev, basis_p.zeros()])

    for it in range(1, picard_max_iter + 1):
        A = a_visc.assemble(basis_u, u_prev=basis_u.interpolate(u_prev))
        B = b_coupling.assemble(basis_u, basis_p)
        K = bmat([[A, B.T], [B, None]], "csr")
        f = np.zeros(K.shape[0])

        xsol = solve(*condense(K, f, x=x0, D=dirichlet_dofs))
        u_new = xsol[: basis_u.N]

        denom = np.linalg.norm(u_new) + 1e-30
        residual = float(np.linalg.norm(u_new - u_prev) / denom)
        u_prev = u_new
        n_iterations = it
        if residual < picard_tol:
            converged = True
            break

    return FlowSolution(
        tagged_mesh=tagged_mesh,
        basis_u=basis_u,
        basis_p=basis_p,
        u=xsol[: basis_u.N],
        p=xsol[basis_u.N :],
        inlet_velocity_m_s=inlet_velocity_m_s,
        n_iterations=n_iterations,
        converged=converged,
        residual=residual,
    )


def solve_pulsatile_flow(
    tagged_mesh: TaggedMesh,
    times_s: np.ndarray,
    mean_velocity_m_s: float,
    amplitude_m_s: float,
    frequency_hz: float,
    carreau: CarreauParams,
    picard_max_iter: int = 50,
    picard_tol: float = 1e-7,
) -> list[FlowSolution]:
    """Quasi-steady pulsatile flow: a sequence of Stokes-Carreau solves.

    Inlet waveform v(t) = mean_velocity + amplitude * cos(2*pi*frequency*t),
    matching the non-reversing sinusoid of Sec. 3.3.4 / Fig. 10-11. Each time
    sample is solved as an independent steady Stokes problem (see module
    docstring "Governing-equation simplification" for the quasi-steady
    approximation this entails).
    """

    solutions = []
    for t in times_s:
        v_in = mean_velocity_m_s + amplitude_m_s * np.cos(2.0 * np.pi * frequency_hz * t)
        solutions.append(
            solve_steady_flow(
                tagged_mesh,
                inlet_velocity_m_s=float(v_in),
                carreau=carreau,
                picard_max_iter=picard_max_iter,
                picard_tol=picard_tol,
            )
        )
    return solutions


def compute_boundary_flux(flow: FlowSolution, boundary_name: str) -> float:
    """Volumetric flux (per unit out-of-plane depth) through a tagged boundary.

    Used for mass-conservation sanity checks (inlet flux should equal outlet
    flux for a steady, source-free flow field).
    """

    from skfem import FacetBasis, Functional
    from skfem.helpers import dot

    fb = FacetBasis(flow.tagged_mesh.mesh, ElementVector(ElementTriP2()), facets=boundary_name)

    @Functional
    def flux(w):
        return dot(w["u"], w.n)

    return float(flux.assemble(fb, u=fb.interpolate(flow.u)))
