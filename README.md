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

The full pipeline (mesh → mechanistic solver → dataset generation → neural
surrogate training → benchmark report) is implemented and runs end to end.
Scope was deliberately kept small given this is a single-session research
prototype -- see the size caveats below and "Known limitations".

| Component | Status |
|---|---|
| `mechanistic/mesh.py`, `flow_solver.py` | **Implemented & tested.** Idealized 2D vessel+aneurysm geometry (self-contained Delaunay mesher), steady Carreau-viscosity Stokes flow (Picard iteration), quasi-steady pulsatile wrapper, Eq. (18) thrombus viscosity feedback. Mass conservation verified to machine precision. |
| `mechanistic/activation.py`, `surface_ode.py`, `fibrin.py` | **Implemented & tested.** Chemical/mechanical activation (Appendix A/B), platelet adhesion/aggregation flux BCs + surface ODEs (Eqs. 6-7, C.1-C.20), Michaelis-Menten fibrin kinetics (Eqs. 16-17). |
| `mechanistic/species_transport.py` | **Implemented & tested.** SUPG-stabilized implicit transport (huge Péclet numbers given Table 1's tiny diffusivities) + vectorized implicit reaction substepping (stiff kinetics, e.g. Eq. A.10's Γ reaching ~1e4 s⁻¹). |
| `mechanistic/coupled_solver.py`, `run_simulation.py` | **Implemented & tested.** Full transient Stokes+CDR+surface-ODE coupling loop; CLI supports both flow-only and full-transient runs. See "Known limitations" re: thrombin/fibrin calibration. |
| `data/sampler.py`, `generate_dataset.py`, `dataset.py` | **Implemented & tested.** LHS sampling + OOD-tail split, batch mechanistic runs with grid rasterization, PyTorch `Dataset`. |
| `neural/encoder.py`, `operator_core.py`, `model.py`, `train.py` | **Implemented & tested.** FiLM-conditioned encoder + a real (if small) Fourier Neural Operator; `operator_core.type: gnn` is a documented, deliberately-unimplemented extension point. |
| `neural/physics_losses.py` | **Partially implemented.** `mass_conservation` and `nonnegativity` penalties (finite-difference mode) are implemented; `cdr_residual`, `surface_flux_bc_residual`, and `autograd` residual mode are not (see module docstring). |
| `neural/uncertainty.py` | **Implemented & tested.** MC-dropout and deep-ensemble wrappers. |
| `benchmark/metrics.py`, `ood_eval.py`, `calibration.py`, `run_benchmark.py` | **Implemented & tested.** Field RMSE, OOD degradation, UQ calibration (reliability diagram + ECE), runtime comparison, Markdown+PNG report. `thrombus_height_error`/`time_to_onset_error`/`physics_residual_audit` are not implemented (need full spatiotemporal fields this project's dataset doesn't save -- see `metrics.py` docstring). |
| `viz/plots.py` | **Implemented.** |

Run `pytest` — the full suite passes (mechanistic solver validation,
surface ODE unit tests, species transport, coupled-solver integration
tests, neural forward/gradient tests, benchmark metric tests).

**Scale caveat:** `configs/training.yaml`'s defaults (24/6/6/6 train/val/
test/ood samples, 32×32 grid, 40 epochs) are a *reduced-scale demo*, not
the full O(100-1000)-sample dataset originally scoped — chosen so the
entire pipeline (`thrombus-generate-dataset` → `thrombus-train` →
`thrombus-benchmark`) runs in a few minutes on CPU in one session. Scale up
`data.n_train`/`n_val`/`n_test`/`n_ood` and `optim.epochs` for a more
statistically meaningful benchmark; nothing in the code depends on these
particular values.

## Architecture

### (A) Mechanistic model — coupled solve loop

Per macro time step `dt`, `mechanistic/coupled_solver.py` cycles flow,
surface, and species physics on a mesh built once by `mesh.py`:

```
mesh.py (Delaunay vessel+aneurysm) ──▶ built once, reused every step
                                              │
                     ┌────────────────────────┘
                     ▼
        ┌──────────────────────────┐
        │ (1) flow_solver.py       │
        │ Stokes + Carreau         │
        │ viscosity (Picard)       │
        └─────────────┬────────────┘
                       │ wall shear rate γ_w
                       ▼
        ┌──────────────────────────┐
        │ (2) wall shear gradient  │
        │ d(γ_w)/dx, split into    │
        │ top / bottom branches    │
        └─────────────┬────────────┘
                       │ negative-gradient-gated mechanical flux
                       ▼
        ┌──────────────────────────┐
        │ (3) surface_ode.py       │
        │ coverage ODEs            │
        │ M, M_r, M_as, M_at       │
        └─────────────┬────────────┘
                       │ updated wall-flux BCs
                       ▼
        ┌──────────────────────────┐
        │ (4) species_transport.py │
        │ + activation.py/fibrin.py│
        │ SUPG transport, Strang-  │
        │ split Newton reaction    │
        └─────────────┬────────────┘
                       │ new M_at, [FI] fields
                       ▼
        ┌──────────────────────────┐
        │ (5) flow_solver.py       │
        │ viscosity_multiplier     │
        │ (Eq. 18 thrombus fdbk)   │
        └─────────────┬────────────┘
                       │
                       └───────────▶ back to (1), next macro step
                                     (repeat until end_time_s)
```

### (B) Neural surrogate — hybrid (biophysics-aware) architecture

"Hybrid" here means the model itself, not the comparison: a data-driven
Fourier Neural Operator conditioned on the same geometry/physiological
parameters as the mechanistic model, trained with physics-informed loss
terms alongside the usual data loss (`neural/model.py` wires
`encoder.py` + `operator_core.py`; `train.py` combines the losses):

```
              8-scalar parameter vector θ
        (geometry preset, physio params, inlet v)
                          │
                          ▼
        ┌───────────────────────────────┐
        │ encoder.py                    │
        │ GeometryParamEncoder           │
        │ FiLM-style MLP modulation of   │
        │ a learned base grid + fixed    │
        │ sinusoidal coordinate embedding│
        └───────────────┬───────────────┘
                        │ latent spatial grid
                        ▼
        ┌───────────────────────────────┐
        │ Dropout2d                     │
        │ (MC-dropout UQ tap point)      │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ operator_core.py              │
        │ Fourier Neural Operator        │
        │ (truncated spectral conv +     │
        │  pointwise residual, per layer)│
        │ [gnn backbone: unimplemented]  │
        └───────────────┬───────────────┘
                        │ predicted field grid
                        │ (velocity x/y + 9 species)
                        ▼
        ┌───────────────────────────────┐
        │ train.py loss                 │
        │ MSE data loss (vs. (A)'s       │
        │ rasterized fields)  +          │
        │ physics_losses.py:             │
        │ mass-conservation +             │
        │ non-negativity penalties       │
        └───────────────┬───────────────┘
                        │
                        ▼
        uncertainty.py: MC-dropout / deep-ensemble
        wrapper repeats the forward pass for UQ
```

### Benchmark pipeline — dataset → train → report

The three CLI entry points (`thrombus-generate-dataset` →
`thrombus-train` → `thrombus-benchmark`) wire (A) and (B) above into an
end-to-end pipeline:

```
data/sampler.py                data/generate_dataset.py
(LHS sampling +      ──▶       batch-runs (A) mechanistic model per
 OOD-tail split)                sample, rasterizes fields to a fixed
                                 grid, writes data/processed/{split}/
                                 sample_NNN.npz
                                       │
                                       ▼
                        thrombus-generate-dataset  (CLI)
                                       │
                                       ▼
                        data/dataset.py
                        ThrombusSurrogateDataset
                                       │
                                       ▼
                        thrombus-train  (CLI)
                        neural/train.py: optimizes (B) ThrombusSurrogate
                        against MSE + physics losses (neural/physics_losses.py)
                                       │
                                       ▼
                        checkpoints/model.pt
                                       │
                                       ▼
                        thrombus-benchmark  (CLI)
                        benchmark/run_benchmark.py: evaluate on test+ood,
                        compute metrics/ood_eval/calibration, time a fresh
                        mechanistic re-solve, render viz/plots.py
                                       │
                                       ▼
                        results/report.md + PNGs
```

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
   before use rather than compared directly. In practice, `[T]_crit` is not
   consumed by any implemented code path: the platelet-activation function
   Omega (Eq. A.9) sums only over the agonists ADP/TxA2, not thrombin
   itself, so no threshold comparison against `[T]_crit` was needed for the
   equations this project implements. The parameter remains in
   `configs/physio_params.yaml` (`activation.thrombin_critical`) as a
   documented placeholder should a future extension need it.

8. **Shear-enhanced species diffusivity.** Sec. 2.1 states red blood cells
   have a "shear-dependent augmenting effect" on platelet/large-protein
   diffusivity, added on top of the Table 1 Brownian coefficients, without
   giving the closed-form expression. This is not implemented -- `flow`
   advects/diffuses species with plain Brownian diffusivity (Table 1);
   fibrinogen/fibrin (not tabulated in Table 1 at all) fall back to the
   platelet Brownian diffusivity as an order-of-magnitude placeholder
   (`mechanistic/coupled_solver.py`, `diffusivity_m2_s` dict).

9. **Eqs. (8)-(13) reconstruction.** The main-text block combining the
   Appendix C chemical flux BCs with the Eqs. (6)-(7) mechanical flux BCs
   renders as dense inline mathematics that could not be extracted with
   full confidence for every subscript from the source PDF. `surface_ode.py`
   reconstructs the combined M/M_as/M_at ODE system from the two
   high-confidence blocks (Eqs. 6-7 and the θ=1-simplified Eqs. C.17-C.20)
   following the pattern Eq. (15) states unambiguously; see that module's
   "Reconstruction note" docstring for the exact equations used.

10. **Numerical method for species transport & coupling.** Table 1's
    diffusivities are tiny relative to advective/domain scales (Péclet
    numbers ~1e6-1e8) and several reaction rate constants reach ~1e4 s⁻¹
    (Eq. A.10's Γ) -- both far outside what a naive explicit or
    unstabilized-Galerkin scheme can handle at a physically meaningful
    macro time step. This project uses SUPG-stabilized implicit transport,
    Strang-split local reaction substeps (vectorized implicit
    backward-Euler + Newton, exploiting the reaction system's per-node
    block-diagonal structure), and substepped surface-ODE updates. COMSOL's
    internal solver details for the reference implementation are unknown;
    these are this project's own numerical choices, documented in
    `mechanistic/species_transport.py` and `coupled_solver.py`.

11. **2D "unit-depth" convention for surface-density units.** The paper's
    M/M_r/M_as/M_at (PLT/cm²) are genuinely areal (3D wall-surface)
    quantities, while this project's mesh/FEM machinery is 2D and built in
    SI (meters). `coupled_solver.py` reconciles this by treating rate
    constants that multiply a *volumetric* bulk concentration (k_rs/k_as/
    k_aa, cm/s) as needing only a length-unit conversion (cm/s → m/s),
    while surface-density quantities that appear bare (not as a
    dimensionless ratio like M/M_inf) need an explicit areal conversion
    (PLT/cm² → PLT/m², ×1e4) -- see the `CM_TO_M`/`CM2_TO_M2` derivation
    comment there. This was verified against an isolated wall-flux
    mass-balance test but is a nontrivial, self-derived unit reconciliation
    the paper's COMSOL implementation would not need to make explicit.

## Known limitations

- **Thrombin/fibrin generation is not well-calibrated.** The surface
  thrombin-generation flux (Eqs. C.5-C.6, via `phi_at`/`phi_rt` on M_at/M_r)
  combined with Eq. (A.10)'s Γ -- which *saturates* rather than
  accelerates as `[T]` grows, since T appears in Γ's denominator --
  produces runaway growth in `[T]` (and downstream `[FI]`) over timescales
  of seconds in the current coupling scheme, well beyond physiological
  ranges (~0.1-10 µM). This was not resolved despite the unit
  reconciliation above (independently verified correct via an isolated
  mass-balance check). A generous concentration cap
  (`coupled_solver.py`'s `concentration_cap`) keeps simulations numerically
  bounded rather than diverging, but `[T]`/`[FI]` values from
  `run_coupled_simulation`/`generate_dataset.py` should not be treated as
  physically meaningful without further recalibration against a reference
  implementation this project does not have access to.
- **No quantitative validation against the paper's reported results.**
  Fig. 4's ~120-minute thrombus height comparison is far outside what this
  project's demo-scale runs simulate (seconds, per the scale caveat above);
  qualitative behaviors this project *does* reproduce (mass-conserving
  flow, shear-thinning viscosity, saturating platelet surface coverage,
  negative-shear-gradient-gated mechanical flux) are checked in `tests/`,
  but there is no COMSOL reference run to compare full thrombus dynamics
  against.
- **Neural surrogate is a small demo model on a small dataset**, not a
  tuned architecture -- see the "Scale caveat" above. Its benchmark numbers
  demonstrate the *pipeline*, not a validated speed/accuracy trade-off.

## Repository layout

```
configs/                  Hydra/OmegaConf YAML configs (geometry, physio params, training)
src/thrombus_bench/
  mechanistic/             2D FEM thrombus formation solver (scikit-fem)
  data/                    Dataset generation (LHS sampling + batch mechanistic runs) for the surrogate
  neural/                  Neural surrogate (encoder + FNO core [GNN unimplemented] + physics-informed losses + UQ)
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

### Run a single mechanistic simulation

```bash
# Flow field only (fast, seconds):
python -m thrombus_bench.mechanistic.run_simulation --flow-only \
    --geometry-preset aneurysm_7mm --output flow_solution.npz

# Full transient coupled simulation (flow + species + surface ODEs):
python -m thrombus_bench.mechanistic.run_simulation \
    --geometry-preset aneurysm_7mm --end-time-s 2.0 --dt-s 0.1 \
    --output simulation.npz
```

Or explore interactively: `jupyter notebook notebooks/01_explore_mechanistic_baseline.ipynb`.

### Run the full benchmark pipeline (dataset → training → report)

```bash
# 1. Generate the (demo-scale, see "Scale caveat" above) dataset:
thrombus-generate-dataset --output-dir data/processed

# 2. Train the neural surrogate:
thrombus-train --dataset-dir data/processed --checkpoint checkpoints/model.pt

# 3. Run the benchmark, producing results/report.md + PNGs:
thrombus-benchmark --checkpoint checkpoints/model.pt --dataset-dir data/processed
```

Each step also accepts `--training-config`/`--physio-config`/
`--geometry-config` overrides (default to the `configs/*.yaml` files in this
repo) and other flags -- see `--help` on each command, or the corresponding
module's `main()` in `src/thrombus_bench/{data,neural,benchmark}/`.

### Run tests

```bash
pytest
```

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

## Contributors

- **Danilo Dursoniah** ([ddursoniah@gmail.com](mailto:ddursoniah@gmail.com))
  — Data Scientist and Computational Biologist. Implementation,
  calibration, and maintenance of the mechanistic solver, neural
  surrogate, and benchmark pipeline.

_Last updated: 2026-07-21_
