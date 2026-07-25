# Optional Navier-Stokes flow: design assessment (Phase 5a)

**Status: assessment only. No implementation code was written for this
phase.** Numbers below are computed directly from this repo's actual
configured values (`configs/geometry.yaml`, `configs/physio_params.yaml`),
not assumed from the paper, and cross-checked against a real mesh built
by `mesh.py` at the resolutions this project actually uses.

## 1. Reynolds number, using this repo's own numbers

`Re = rho * V * D / mu`. Viscosity is shear-dependent (Carreau, Eq. 2), so
three variants, computed for both paper presets at their own configured
inlet velocity:

| Preset | D | V | wall shear rate `gamma_w = 6V/D` | `mu` at that shear rate | Re(`mu_inf`) | Re(apparent `mu`) | Re(`mu_0`) | Paper's reported Re |
|---|---|---|---|---|---|---|---|---|
| `aneurysm_7mm` | 3.2 mm | 0.47 m/s | 881 /s | 0.00381 Pa·s | **477** | 438 | 30 | 455 |
| `aneurysm_10mm` | 4.0 mm | 0.75 m/s | 1125 /s | 0.00376 Pa·s | **951** | 885 | 59 | 908 |

**I used `mu_inf` (or, equivalently, the apparent viscosity at the actual
wall shear rate, which comes out nearly identical) as the representative
value, not `mu_0`, and not a `mu_inf`-to-`mu_0` "range."** Reasoning: the
Carreau time constant `lambda_s = 3.313 s` gives a shear-thinning
transition around `1/lambda ~ 0.3 /s` -- orders of magnitude below the
actual wall shear rates here (881-1125 /s). At these flow rates the fluid
is essentially fully shear-thinned almost everywhere in the domain (away
from stagnation points/recirculation cores), so `mu_0` describing the
*zero-shear* limit is not representative of the bulk flow at all -- it's
a lower bound on Re, not a physically likely operating point. Computing
the Carreau viscosity at the actual wall shear rate confirms this
directly: the apparent viscosity (0.00381, 0.00376 Pa·s) is within 9% of
`mu_inf` (0.0035 Pa·s), not partway toward `mu_0` (0.056 Pa·s). This also
explains why `Re(mu_inf)` lands so close to the paper's own reported
values (477 vs. 455, 951 vs. 908) -- the paper almost certainly computed
its Re the same way, and this repo's density/viscosity/diameter/velocity
values are copied directly from the same source (Table 1, Sec. 2.8), so
they should and do reproduce it.

**Sampled parameter range** (`data/sampler.py`'s `DEFAULT_RANGES`:
`vessel_diameter_mm` 3.2-4.0, `inlet_velocity_cm_s` 30-100): `Re(mu_inf)`
spans **~304 to ~1269** across the full space `thrombus-generate-dataset`
actually samples from -- comfortably non-creeping (`Re >> 1`) everywhere
in that range, not just at the two paper presets.

**Bottom line, stated plainly since the plan explicitly asked me not to
default to a fixed conclusion: this repo's own configured operating
conditions are genuinely not creeping flow.** Re ~ 300-1270 is moderate
laminar flow with non-negligible inertia -- Stokes is a real, non-trivial
physical simplification here, not an incidentally-fine approximation of
this project's own reduced setup. I want to be direct about this because
it would have been convenient (and dishonest) to find otherwise.

## 2. How ambitious does the convective term need to be?

This is where the honest assessment gets more interesting than a Reynolds
number alone suggests, and where I'd push back on the plan's framing that
a "straightforward Picard-iterated convective term" is likely sufficient.

**I built the actual mesh this project generates and measured its element
size directly**, then computed the *element-scale* Reynolds number
(`Re_h = rho * V * h / mu`, using the bulk inlet velocity and `mu_inf`) --
the quantity that actually governs whether a plain Galerkin discretization
of the convective term is numerically stable, independent of the
domain-scale Reynolds number above:

| Preset | `target_num_elements` | mean element size `h` | `Re_h` |
|---|---|---|---|
| `aneurysm_7mm` | 800 (current CLI default) | 0.69 mm | **103** |
| `aneurysm_7mm` | 2000 (`configs/geometry.yaml` default) | 0.48 mm | **72** |
| `aneurysm_10mm` | 800 | 0.80 mm | **191** |
| `aneurysm_10mm` | 2000 | 0.56 mm | **133** |

Plain (non-stabilized) Galerkin finite elements for an advection-dominated
term are well known to become oscillatory/unstable once the element-scale
Reynolds/Péclet number exceeds roughly **2** -- this project's own
`species_transport.py` module already builds in SUPG stabilization for
exactly this reason, citing Péclet numbers of `~1e6-1e8` for species
transport. The momentum equation's convective term at `Re_h` of **72 to
191** is nowhere near that extreme, but it is 35-95x past the plain-
Galerkin stability threshold at every mesh resolution this project
actually uses.

**Concretely, this means "just add `u . grad(u)` to `a_visc`'s
`BilinearForm` and Picard-iterate" (extending the existing pattern
literally) would very likely produce an oscillatory, physically
unrealistic velocity field at the mesh resolutions this project has
standardized on** -- not a hypothetical concern, a direct consequence of
the measured `Re_h` above. A robust implementation needs the momentum
equivalent of what `species_transport.py` already does for species: SUPG
(streamline-upwind Petrov-Galerkin) stabilization of the convective term,
and for the mixed velocity-pressure formulation this solver uses, likely
PSPG (pressure-stabilized Petrov-Galerkin) as well to maintain the
inf-sup-stable behavior once the Stokes-only structure is perturbed by a
stabilized convective term. This is a well-established, but genuinely
more involved, FEM technique than the existing Stokes solve -- it
requires a stabilization parameter (`tau`) calibrated to local element
size/velocity/viscosity, additional terms in both the momentum and
continuity weak forms, and careful handling to not silently degrade the
existing Stokes-mode accuracy when the new mode is disabled (this
project's own convention: opt-in, default-off, `flow_solver.py`'s
`solve_steady_flow` today).

Two more concrete complexity items past what the plan's framing
suggested:
- **Picard iteration on the convective term is not guaranteed to converge
  without under-relaxation** at `Re` in the hundreds -- this is standard
  CFD practice (e.g., SIMPLE-family algorithms typically under-relax
  velocity by a factor of 0.3-0.7), not an edge case; I would expect the
  existing `picard_max_iter=50`/`picard_tol=1e-7` pattern to need an
  under-relaxation factor added, not just more iterations.
- **Increasing mesh resolution to push `Re_h` back under ~2 is not a free
  alternative to stabilization.** Getting `Re_h` from ~100 down to ~2
  needs roughly a 50x reduction in `h`, i.e. ~2500x more elements in 2D
  -- directly at odds with this project's explicit, repeatedly-stated CPU-
  runtime-budget design constraint (`configs/pilot.yaml`'s own header
  comment: mesh/dataset generation cost dominates total runtime and scales
  with element count). Stabilization, not refinement, is the realistic
  path if this is pursued.

## 3. Is it worth the complexity? My honest opinion

**Leaning no, given this project's own stated goals -- but it's a real
tradeoff, not a clear-cut call, and I want to lay out both sides rather
than pick for you.**

**The case for not doing it:**
- The project's explicitly stated goal (`README.md`'s own opening
  paragraph) is qualitative behavior, not quantitative agreement with the
  paper's figures. The aneurysm sac's recirculation zone -- the qualitative
  feature that actually matters for the shear-gradient-driven mechanical
  aggregation mechanism this whole model exists to demonstrate -- is
  primarily a consequence of the sudden geometric expansion at the neck,
  and is already visibly present in the current Stokes solutions (see the
  streamline figures from Stage 1/4b's visual verification, this
  conversation). Missing inertial effects would refine the recirculation's
  *strength and exact extent*, not create a qualitatively different flow
  topology from scratch.
- The real engineering cost is SUPG/PSPG stabilization, under-relaxation
  tuning, and new convergence-behavior tests across both presets -- a
  meaningfully larger scope than "extend the existing Picard loop,"
  closer in size to the SUPG work `species_transport.py` already required
  once, than to a small incremental change.
- This project already carries one open, unresolved numerical-fidelity
  issue (`coupled_solver.py`'s thrombin/fibrin concentration-cap
  limitation, documented in `README.md`) that arguably has more direct
  bearing on the model's headline "does a thrombus form" result than the
  Stokes-vs-Navier-Stokes flow question does.

**The case for doing it anyway:**
- `Re` genuinely is in the hundreds at every configuration this project
  samples from, including both paper presets -- "Stokes is fine here" is
  not a defensible blanket statement, and I won't make it. If quantitative
  fidelity to the paper's actual flow fields (not just qualitative
  thrombus-formation behavior) becomes a project goal later, this gap
  would need closing eventually.
- The existing Stokes solver is well-tested and stable; an opt-in,
  default-off convective mode (per the plan's own scoping) risks nothing
  for the existing pipeline if built carefully, and having it available
  would let you directly show a reader the Stokes-vs-Navier-Stokes
  difference on the same geometry -- a genuinely interesting comparison
  figure in its own right.

**My recommendation:** don't build this now. The qualitative-behavior goal
this project has already committed to doesn't need it, and the honest
engineering cost (stabilization, not just a convective term) is
substantial enough that I'd want a specific reason to want it -- e.g., a
concrete plan to eventually compare quantitative wall-shear/velocity
values against the paper's figures -- before taking it on. If you do want
it, I'd scope Phase 5b explicitly to include SUPG/PSPG stabilization and
under-relaxation from the start, not attempt the naive convective-term
version first and add stabilization only after observing instability.

Waiting for your decision before writing any implementation code.
