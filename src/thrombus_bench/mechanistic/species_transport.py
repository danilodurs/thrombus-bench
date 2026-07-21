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

Numerical method (explicit modeling choice)
-----------------------------------------------
Table 1's Brownian diffusivities are tiny relative to the advective and
domain length scales (e.g. D_thrombin ~ 4e-11 m^2/s vs. a ~cm domain and
~0.1-1 m/s flow -- Peclet numbers of order 1e6-1e8). This is a genuinely
advection-dominated transport regime, and unstabilized P1 Galerkin FEM is
unconditionally unstable there. This module uses:

1. **SUPG** (streamline-upwind Petrov-Galerkin) stabilization for the
   transport operator, with a stabilization parameter combining transient,
   advective, and diffusive scales: `tau = (1/dt + 2|u|/h + 4D/h^2)^-1`.
2. **Strang/Lie operator splitting** between (fast, stiff) reaction kinetics
   and (linear) transport: each macro time step first solves the *local*
   (spatially-decoupled) reaction ODEs at every mesh node with an implicit
   stiff integrator (`scipy.integrate.solve_ivp`, `activation.py` +
   `fibrin.py` source terms -- rate constants up to ~O(1e4) s^-1 make
   explicit reaction integration impractical at any physically meaningful
   macro time step), then a single implicit-Euler + SUPG linear transport
   step per species using the post-reaction concentrations as initial data.

This is a deliberate simplification relative to a monolithic implicit
solve (as COMSOL would do); it trades some splitting error for a much
simpler, still second-order-consistent-in-the-relevant-limits scheme
appropriate for a coarse research-prototype mesh. See README.md
"Assumptions & Deviations from Source Paper".

Boundary conditions (Sec. 2.1, Sec. 2.3-2.4)
-----------------------------------------------
* Inlet: Dirichlet, constant physiological concentration per species
  (`configs/physio_params.yaml` `species.*_inlet_*`), thrombin = 0.
* Outlet: zero axial concentration gradient (natural/do-nothing in the weak
  form -- no boundary term added).
* Walls: heterogeneous Neumann flux boundary conditions from Appendix C
  (`surface_ode.py`) for RP, AP (adhesion/aggregation), PT, T (surface
  thrombin generation, Eqs. C.5-C.6); no-flux (natural) for ADP, TxA2, AT,
  FG, FI, matching the paper's statement that antithrombin (and, by
  omission, the agonists/fibrinogen/fibrin) are "not involved in
  surface-based reactions."
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from skfem import Basis, BilinearForm, FacetBasis, LinearForm, condense, solve
from skfem.helpers import dot, grad

from .mesh import TaggedMesh

# Species with a nonzero wall (Neumann) flux boundary condition, per
# Appendix C: RP/AP via adhesion+aggregation (Eqs. C.1-C.2), PT/T via
# surface thrombin generation (Eqs. C.5-C.6). All others are natural
# (zero-flux) at the wall.
WALL_REACTIVE_SPECIES = ("RP", "AP", "PT", "T")


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

    concentrations: dict  # name -> np.ndarray (nodal DOF vector, P1 basis)
    time_s: float


def _supg_forms(basis_c: Basis, velocity_field, diffusivity_m2_s: float, dt_s: float):
    """Bilinear (transport operator) and mass-matrix forms shared by every
    species' transport step (only the RHS/source differs per species)."""

    @BilinearForm
    def bilinear(u, v, w):
        vel = w["vel"]
        h = w.h
        speed = np.sqrt(np.maximum(dot(vel, vel), 1e-30))
        tau = 1.0 / (1.0 / dt_s + 2.0 * speed / h + 4.0 * diffusivity_m2_s / h**2)

        galerkin = (u / dt_s) * v + dot(vel, grad(u)) * v + diffusivity_m2_s * dot(grad(u), grad(v))
        supg = tau * (u / dt_s + dot(vel, grad(u))) * dot(vel, grad(v))
        return galerkin + supg

    @LinearForm
    def rhs_history(v, w):
        vel = w["vel"]
        h = w.h
        speed = np.sqrt(np.maximum(dot(vel, vel), 1e-30))
        tau = 1.0 / (1.0 / dt_s + 2.0 * speed / h + 4.0 * diffusivity_m2_s / h**2)
        C_prev = w["C_prev"]
        return (C_prev / dt_s) * v + tau * (C_prev / dt_s) * dot(vel, grad(v))

    return bilinear, rhs_history


def solve_transport_step(
    basis_c: Basis,
    tagged_mesh: TaggedMesh,
    velocity_field,
    C_prev: np.ndarray,
    diffusivity_m2_s: float,
    dt_s: float,
    inlet_value: float,
    wall_flux: np.ndarray | None = None,
) -> np.ndarray:
    """One implicit-Euler + SUPG transport step (no source; apply reaction
    separately via `reaction_step`), Eq. (1)'s transport operator.

    `wall_flux`, if given, is a per-wall-quadrature-point array (matching a
    `FacetBasis(mesh, ..., facets=["wall_vessel", "wall_sac"])` quadrature
    layout) of the Appendix C Neumann flux (positive = efflux/consumption),
    contributing `-integral(wall_flux * v) ds` to the right-hand side.
    """

    bilinear, rhs_history = _supg_forms(basis_c, velocity_field, diffusivity_m2_s, dt_s)

    A = bilinear.assemble(basis_c, vel=velocity_field)
    b = rhs_history.assemble(basis_c, vel=velocity_field, C_prev=basis_c.interpolate(C_prev))

    if wall_flux is not None:
        wall_names = [n for n in ("wall_vessel", "wall_sac") if len(tagged_mesh.mesh.boundaries.get(n, [])) > 0]
        fb = FacetBasis(tagged_mesh.mesh, basis_c.elem, facets=wall_names)

        @LinearForm
        def wall_term(v, w):
            return -w["flux"] * v

        b = b + wall_term.assemble(fb, flux=wall_flux)

    inlet_dofs = basis_c.get_dofs("inlet").all()
    x0 = np.zeros(basis_c.N)
    x0[inlet_dofs] = inlet_value

    return solve(*condense(A, b, x=x0, D=inlet_dofs))


def reaction_step(
    concentrations: dict[str, np.ndarray],
    dt_s: float,
    source_fn,
    n_substeps: int = 10,
    n_newton_iters: int = 6,
) -> dict[str, np.ndarray]:
    """Advance all species' *local* (spatially-decoupled) reaction kinetics
    by `dt_s` using vectorized implicit backward-Euler + Newton iteration.

    The reaction system is spatially block-diagonal: each mesh node's nine
    species evolve independently of every other node's (no spatial
    derivatives in the reaction terms). A single `scipy.integrate.solve_ivp`
    call over the full flattened (9 * n_nodes)-dimensional state would treat
    this as a dense system and attempt a dense finite-difference Jacobian --
    catastrophically slow/memory-heavy for meshes with more than a few dozen
    nodes. Instead, this function solves the small 9x9 implicit system at
    every node *simultaneously* via `numpy.linalg.solve`'s native batched
    (n_nodes, 9, 9) solve, using a numerical (finite-difference) Jacobian of
    `source_fn` evaluated once per Newton iteration. `dt_s` is subdivided
    into `n_substeps` backward-Euler steps (backward Euler is unconditionally
    stable, but accuracy over one large step is poor given rate constants up
    to O(1e4) s^-1 in this system, per Eq. A.10's Gamma) -- see module
    docstring "Numerical method".

    `source_fn(conc_dict) -> dict[str, np.ndarray]` returns the pointwise
    reaction source terms S_i (e.g. from `activation.chemical_source_terms`
    plus `fibrin.fibrin_source_terms`) given the current nodal concentration
    arrays.
    """

    names = list(concentrations.keys())
    n_species = len(names)
    n_nodes = concentrations[names[0]].shape[0]
    C = np.stack([concentrations[name] for name in names], axis=1)  # (n_nodes, n_species)
    sub_dt = dt_s / n_substeps

    def source_stack(C_stacked: np.ndarray) -> np.ndarray:
        conc = {name: C_stacked[:, i] for i, name in enumerate(names)}
        S = source_fn(conc)
        return np.stack([S[name] for name in names], axis=1)

    for _ in range(n_substeps):
        C_prev = C.copy()
        C_guess = C_prev.copy()
        for _ in range(n_newton_iters):
            base_S = source_stack(C_guess)
            residual = C_guess - C_prev - sub_dt * base_S  # (n_nodes, n_species)

            jac = np.zeros((n_nodes, n_species, n_species))
            for j in range(n_species):
                perturbation = 1e-6 * np.maximum(np.abs(C_guess[:, j]), 1.0)
                C_pert = C_guess.copy()
                C_pert[:, j] += perturbation
                dS_dCj = (source_stack(C_pert) - base_S) / perturbation[:, None]
                jac[:, :, j] = -sub_dt * dS_dCj
            jac[:, np.arange(n_species), np.arange(n_species)] += 1.0

            # np.linalg.solve batches over leading dims when b has a trailing
            # size-1 "n" axis (gufunc signature (m,m),(m,n)->(m,n)); without
            # it, b's shape (n_nodes, n_species) is misread as a single
            # (m,n) matrix rather than n_nodes independent m-vectors.
            delta = np.linalg.solve(jac, -residual[..., None])[..., 0]
            C_guess = C_guess + delta
        C = np.maximum(C_guess, 0.0)

    return {name: C[:, i] for i, name in enumerate(names)}
