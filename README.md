# thrombus-bench

A benchmarking pipeline comparing **(A)** a mechanistic 2D thrombus formation
model against **(B)** a biophysics-aware neural surrogate, following:

> Cardillo, G., Pouponneau, P., & Barakat, A.I. (2026). *A computational model
> of chemically- and mechanically-induced thrombus formation in cerebral
> aneurysms.* Computers in Biology and Medicine, 213, 111829.
> https://doi.org/10.1016/j.compbiomed.2026.111829

**This is an independent, simplified reimplementation for benchmarking
purposes, not an official reproduction.** It is not affiliated with the
paper's authors, does not use their COMSOL implementation or code, and makes
several explicit simplifying assumptions documented below. Numeric agreement
with the paper's figures is not a project goal; qualitative behavior
(shear-gradient-driven platelet aggregation, thrombus localization near the
distal neck, viscosity feedback) is.

## Project status

| Component | Status |
|---|---|
| `mechanistic/mesh.py` | **Implemented & tested.** Idealized 2D vessel+aneurysm geometry, self-contained Delaunay mesher. |
| `mechanistic/flow_solver.py` | **Implemented & tested.** Steady Carreau-viscosity Stokes flow, Picard iteration, quasi-steady pulsatile wrapper. |
| `mechanistic/species_transport.py`, `activation.py`, `surface_ode.py`, `coupled_solver.py` | Scaffolding stubs (docstrings + signatures only). |
| `mechanistic/fibrin.py` | Implemented (closed-form Michaelis-Menten kinetics). |
| `mechanistic/run_simulation.py` | Flow-only path implemented; full transient path pending `coupled_solver.py`. |
| `data/`, `neural/`, `benchmark/`, `viz/` | Scaffolding stubs. |

Run `pytest` for the current state of test coverage: the mechanistic flow
solver's trivial steady-state validation suite
(`tests/test_mechanistic_conservation.py`) passes in full; other test files
document intended coverage for not-yet-implemented modules via `xfail`.

## Equations summary

The mechanistic model (Sec. 2 of the paper) is a coupled Navier-Stokes /
convection-diffusion-reaction (CDR) / surface-ODE system:

- **Flow** (Eq. 3-4): incompressible Navier-Stokes with a Carreau
  generalized-Newtonian viscosity closure (Eq. 2),
  `mu(gamma_dot) = mu_inf + (mu0 - mu_inf) * (1 + (lambda*gamma_dot)^2)^((n-1)/2)`.
- **Species transport** (Eq. 1): nine species (resting/activated platelets,
  released/synthesized agonists, thrombin, antithrombin, prothrombin,
  fibrinogen, fibrin) transported by CDR equations with source terms from
  Appendix A (chemical activation/inhibition kinetics) and Appendix B
  (mechanical, shear-induced activation).
- **Platelet adhesion & aggregation** (Eqs. 6-13, Appendix C): flux boundary
  conditions at the vessel wall combining chemical adhesion (first-order
  surface reactions) and **mechanical, shear-gradient-driven aggregation** —
  the paper's key mechanism, active only where the axial wall shear rate
  gradient is *negative* (Nesbitt et al.'s experimental finding).
- **Surface coverage ODEs** (Eqs. 14-15, C.17-C.20): total platelet surface
  coverage M, and its resting/activated-surface/aggregated decomposition
  (M_r, M_as, M_at).
- **Fibrin generation** (Eqs. 16-17): Michaelis-Menten thrombin-mediated
  conversion of fibrinogen to fibrin (Anand et al. 2003).
- **Viscosity feedback** (Eq. 18): once local deposited-activated-platelet
  coverage M_at or fibrin concentration FI crosses a threshold, local blood
  viscosity is multiplied by up to 80x (one threshold) or 160x (both),
  mimicking the thrombus's effect on the flow field.

See each module's docstring in `src/thrombus_bench/mechanistic/` for the
specific equation numbers it implements.

## Solver backend

The paper's reference implementation uses COMSOL Multiphysics with FEniCSx
suggested as a candidate open alternative. **This project uses
[scikit-fem](https://github.com/kinnala/scikit-fem)** instead of FEniCSx:
`fenics-dolfinx` (the FEniCSx Python package) is not reliably pip-installable
across platforms — it requires a compiled PETSc/MPI stack, typically via
conda or building from source, which conflicts with the project's
pip/conda-only, no-external-network-calls-at-runtime constraint.
`scikit-fem` is pure pip-installable, has no MPI/PETSc dependency, and is
sufficient for the coarse, idealized 2D meshes this project targets.

Because scikit-fem does not ship a CAD/CSG geometry engine, `mesh.py`
generates the idealized vessel+aneurysm domain directly: boundary points are
placed along the analytic outline (rectangle + circular-arc sac), interior
points are filled on a density-graded background grid (refined near the
proximal/distal neck), and a `scipy.spatial.Delaunay` triangulation is
filtered to the domain polygon. This avoids adding a heavier meshing
dependency (e.g. `gmsh`) while remaining fully offline after `pip install`.
It is adequate for coarse, idealized geometries; it is not a substitute for
a robust CSG mesher on complex or patient-specific geometries.

## Assumptions & Deviations from Source Paper

The paper describes the model at the level of governing equations and
parameter tables but leaves several implementation details unspecified. This
project makes the following explicit choices, each also noted in the
relevant module's docstring:

1. **Governing-equation simplification (flow).** The paper solves the full
   unsteady Navier-Stokes equations in COMSOL (Reynolds numbers 455-908,
   i.e. not creeping flow). This project solves the **steady, inertialess
   (Stokes) limit** instead, handling the Carreau nonlinearity via Picard
   fixed-point iteration, to keep the solver simple enough to run on CPU
   with a coarse mesh (explicit project scope). Pulsatile inflow (Sec.
   3.3.4) is supported as a **quasi-steady** sequence of independent Stokes
   solves at sampled points of the inlet waveform, rather than a true
   unsteady solve — an approximation, given the paper reports a Womersley
   number of 2.75 (not asymptotically small).

2. **Idealized geometry parametrization.** The paper shows the idealized
   vessel+aneurysm geometries only as images (Fig. 1) and gives only the
   vessel/aneurysm diameters and vessel length — no parametric definition of
   the sac shape. This project models the sac as the union of the vessel
   rectangle with the upper half-disk of a circle of the specified aneurysm
   diameter, centered on the vessel's top wall at its midpoint (neck width =
   full aneurysm diameter). This is a reasonable reading of Fig. 1 but is
   not the paper's literal geometry definition (which does not exist in
   published form).

3. **Smooth step functions.** The paper defines several functions as hard
   thresholds/steps with only qualitative descriptions of their smoothing
   ("to avoid functional discontinuities and consequent numerical
   instabilities, we introduced two smooth step functions...", Sec. 2.6) but
   never gives the smoothing functional form. This project approximates all
   such steps — `Theta_plts`/`Theta_FI` (Eq. 18, in `flow_solver.py`),
   `step(Omega)` (Eq. A.8-A.9, in `activation.py`), and the Eq. (B.1)
   mechanical-activation kink — with logistic sigmoids of configurable
   steepness (`configs/physio_params.yaml` `activation.smoothing`), using
   the *relative* deviation from the threshold so steepness is a
   dimensionless, scale-independent knob. This is an explicit modeling
   choice, not specified by the source paper.

4. **Initial conditions.** Per the paper: zero initial surface coverage
   (`M(x,0) = M_r(x,0) = M_as(x,0) = M_at(x,0) = 0`) and uniform initial
   species concentrations equal to the inlet values, except thrombin
   (initial and inlet concentration = 0). Implemented as stated —
   see `surface_ode.SurfaceState.zeros_like`.

5. **Missing Appendix D parameter.** Table D.2 (chemical activation
   constants, adopted from Sorensen et al.) tabulates `k_{1,TxA2}` (the
   TxA2 inhibition rate constant, Eq. A.4) but not a corresponding
   `k_{1,ADP}` for Eq. (A.3)'s analogous ADP inhibition term. This project
   reuses the `k_{1,TxA2}` value for ADP (`configs/physio_params.yaml`
   `sorensen_chemical.k1_ADP_per_s`) as a documented placeholder, pending a
   better-sourced value.

6. **`D_a` calibration constant.** Eqs. (14)-(15) include a thrombus
   growth-rate scale factor `D_a` that the paper states was tuned ("different
   values of D_a were tested, and the one specified in Table 1 was
   determined to be the one that ensures smooth convergence and best
   reproduces the experimental data") without reporting the resulting
   numeric value in the table as extracted for this project. Defaulted to
   `1.0` (`configs/physio_params.yaml` `scale_terms.D_a`); should be
   recalibrated against Fig. 4/7/8/9 if closer quantitative agreement is
   desired.

7. **Thrombin critical-concentration units.** Table 1 lists `[T]_crit = 0.1
   U l^-1`, while thrombin concentration elsewhere in the model (`[T]`, `T`
   inlet/initial conditions) is in `uM`. The paper's own `beta` conversion
   factor (Appendix D, U/mL <-> uM) suggests this is meant to be converted
   before use rather than compared directly; this project treats
   `[T]_crit` as requiring that conversion in `activation.py` (not yet
   implemented) rather than assuming a typo.

8. **Shear-enhanced species diffusivity.** Sec. 2.1 states red blood cells
   have a "shear-dependent augmenting effect" on platelet/large-protein
   diffusivity, added on top of the Table 1 Brownian coefficients, without
   giving the closed-form expression. `species_transport.py` (not yet
   implemented) documents a configurable `D_i(gamma_dot) = D_b,i * (1 +
   k_rbc * gamma_dot)` closure, defaulting to `k_rbc = 0` (pure Brownian
   diffusion) until a literature-sourced closure is selected.

## Repository layout

```
configs/                  Hydra/OmegaConf YAML configs (geometry, physio params, training)
src/thrombus_bench/
  mechanistic/             2D FEM thrombus formation solver (scikit-fem)
  data/                    Dataset generation (LHS sampling + batch mechanistic runs) for the surrogate
  neural/                  Neural surrogate (encoder + FNO/GNN core + physics-informed losses + UQ)
  benchmark/               Accuracy/runtime/OOD/calibration metrics + report generation
  viz/                     Plotting utilities
tests/                     pytest suite
notebooks/                 Exploratory notebooks
results/                   Benchmark report output (gitignored except .gitkeep)
```

## How to run

### Environment

Target Python 3.11. Example setup via conda (no system Python 3.11 assumed):

```bash
conda create -n thrombus-bench python=3.11 -y
conda activate thrombus-bench
pip install -e .
# torch CPU wheel (if the default index doesn't resolve a CPU-only build):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Run the mechanistic flow solver (currently implemented path)

```bash
python -m thrombus_bench.mechanistic.run_simulation \
    --vessel-diameter-mm 3.2 --aneurysm-diameter-mm 7.0 \
    --inlet-velocity-cm-s 47 --output flow_solution.npz
```

Or explore interactively: `jupyter notebook notebooks/01_explore_mechanistic_baseline.ipynb`.

### Run tests

```bash
pytest
```

### Not yet runnable (pending implementation)

`thrombus-generate-dataset`, `thrombus-train`, `thrombus-benchmark` are
wired up as CLI entrypoints (`pyproject.toml` `[project.scripts]`) but their
underlying modules (`coupled_solver.py`, `dataset.py`, `model.py`, etc.) are
scaffolding stubs — see "Project status" above.

## Configuration

- `configs/geometry.yaml`: the two idealized aneurysm presets
  (`aneurysm_7mm`, `aneurysm_10mm`) matching the paper's experimental
  geometries, plus mesh resolution controls.
- `configs/physio_params.yaml`: Table 1 and Table D.2 physiological/kinetic
  parameters, with inline citations to the specific equation/table each
  value comes from, and inline notes on assumptions (see "Assumptions &
  Deviations" above).
- `configs/training.yaml`: neural surrogate architecture, physics-loss
  weights, optimizer, and dataset split sizes (including the OOD split).
