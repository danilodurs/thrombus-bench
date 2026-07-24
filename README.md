# thrombus-bench

A benchmarking pipeline comparing **(A)** a mechanistic 2D thrombus formation
model against **(B)** a physiology-conditioned, physics-informed neural surrogate, following:

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
| `mechanistic/geometry_sdf.py` | **Implemented & tested.** Closed-form signed-distance-to-wall function for the idealized vessel+aneurysm domain (exact union of a rectangle + circular-arc sac, not a naive per-shape min/max), reused by both `CoordinateDecoder`'s SDF input feature and the autograd physics residual's collocation-point rejection sampling. See "Known limitations" re: scope (idealized geometries only). |
| `data/sampler.py`, `generate_dataset.py`, `dataset.py` | **Implemented & tested.** LHS sampling + edge-of-domain tail split + opt-in genuine-extrapolation split (`sample_with_extrapolation_holdout`); `generate_dataset.py` now saves point-cloud `.npz` samples by default (mesh node coordinates + values, every checkpoint per `data.n_snapshots` -- no rasterization), with the legacy fixed-grid rasterization kept as an opt-in (`--also-save-raster`) for the comparison baseline. `dataset.py` exposes both `PointCloudThrombusDataset` (primary, ragged batching) and `ThrombusSurrogateDataset` (legacy/comparison, fixed raster). |
| `neural/coordinate_encoding.py` | **Implemented & tested.** Fourier/SIREN-style sinusoidal positional encoding for the `CoordinateDecoder`'s continuous `(x, y)` input. |
| `neural/encoder.py`, `operator_core.py`, `model.py`, `coordinate_decoder.py`, `train.py` | **Implemented & tested.** FiLM-conditioned encoder + a real (if small) Fourier Neural Operator, now factored into a shared trunk (`SurrogateBackbone`) feeding either the original `Conv2d` grid-projection head (`ThrombusSurrogate`, kept as a comparison baseline) or the new `CoordinateDecoder` head (`ContinuousThrombusSurrogate`, primary path) -- see "The coordinate-decoder design" below. `operator_core.type: gnn` is still a documented, deliberately-unimplemented extension point. `train.py` has both `train` (grid) and `train_continuous` (point-cloud, including the optional autograd physics-loss term and per-checkpoint species channel-exclusion) loops. |
| `neural/physics_losses.py` | **Partially implemented, more so than before.** `mass_conservation` (both `finite_difference`, for the grid path, and now a real `autograd` mode -- `mass_conservation_penalty_autograd` + `sample_collocation_points` + `continuous_mass_conservation_loss`, PINN-style, for the continuous path) and `nonnegativity` (finite-difference only) are implemented. `cdr_residual` and `surface_flux_bc_residual` remain unimplemented -- Phase 5 assessed `cdr_residual` for the concentration-cap-unaffected species (RP/AP/APR/APS) as more tractable than previously assumed (their reaction terms turn out to be decoupled from the unreliable T/PT/FI pathway) but still real additional scope (autograd through the whole encoder for `d/dt`, a differentiable bulk shear-rate port, per-species Laplacians), so it was deliberately deferred rather than built; see that module's docstring. |
| `neural/uncertainty.py` | **Implemented & tested.** MC-dropout and deep-ensemble wrappers; confirmed (not just assumed) to work unmodified with `ContinuousThrombusSurrogate`'s multi-argument forward signature, since both wrap `forward()` generically. |
| `benchmark/metrics.py`, `edge_holdout_eval.py`, `extrapolation_eval.py`, `calibration.py`, `run_benchmark.py` | **Implemented & tested.** Field RMSE (`field_rmse`, grid) and its point-query counterparts (`field_rmse_pointwise`, `field_rmse_by_checkpoint`, `field_rmse_by_distance_to_wall` -- the last one new: RMSE binned by distance to the nearest wall, directly answering whether near-wall accuracy is worse than bulk accuracy), edge-of-domain degradation, genuine-extrapolation degradation (opt-in, `scripts/evaluate_extrapolation.py`/`evaluate_extrapolation_continuous.py`) -- each with a grid and a continuous counterpart function -- UQ calibration (reliability diagram + ECE, works unchanged for both model families), runtime comparison, a `bootstrap_metric_by_sample` utility (resamples whole `sample_id`s, not individual points/checkpoints -- no bootstrap CI code existed before this, so this is a pre-emptive correctness guardrail, not a fix), and Markdown+PNG report generation for both the grid path (`run_benchmark`, `results/report.md`) and the continuous path (`run_benchmark_continuous`, `results/report_continuous.md`, with an optional 4th grid-FNO comparison row). `thrombus_height_error`/`time_to_onset_error`/`physics_residual_audit` remain unimplemented (the point-cloud path's full per-checkpoint spatial fields could plausibly support the first two now, but that's unassessed, separate work -- see `metrics.py` docstring). |
| `viz/plots.py`, `viz/rasterize_continuous.py` | **Implemented.** `rasterize_continuous_model` (new) queries a trained `ContinuousThrombusSurrogate` on a regular grid over the analytic bounding box (masking exterior cells via `geometry_sdf.py`) purely for display, so grid-style plots remain possible without the model itself being grid-based. |

Run `pytest` — the full suite passes (mechanistic solver verification
against analytical/sanity cases, surface ODE unit tests, species transport,
coupled-solver integration tests, neural forward/gradient tests, benchmark
metric tests). `verification/convergence_study.md` (generated by
`scripts/convergence_study.py`) is a complementary mesh/time-step
self-convergence check on the two paper geometries -- self-convergence
only, not validation against a reference solution (none exists here; see
"Known limitations" below). `verification/sensitivity_study.md` (generated
by `scripts/sensitivity_study.py`) is a one-at-a-time sensitivity sweep of
the two documented placeholder parameters (items 5/6 below) -- it found
`k1_ADP_per_s` doesn't materially affect headline outputs across a 0.5x-2x
range, but `D_a` does: `thrombosed_fraction` swings from ~0 to a large
fraction of the wall across a plausible 10x-each-way bracket, a
qualitative, not just quantitative, difference.

**Scale caveat:** the CLI default, `configs/demo_cpu.yaml` (24/6/6/6
train/val/test/edge_holdout samples, 32×32 grid, 40 epochs), is a
*reduced-scale pipeline smoke test*, not the full O(100-1000)-sample
dataset originally scoped — chosen so the entire pipeline
(`thrombus-generate-dataset` → `thrombus-train` → `thrombus-benchmark`)
runs in a few minutes on CPU in one session. Its numbers should never be
read as a real benchmark result. `configs/pilot.yaml` (240/40/40/40
samples, 150 epochs, ~15 minutes on CPU — see that file's header comment
for the wall-clock estimate and its basis) is sized to actually produce a
meaningful, if still CPU-modest, result; scale `data.n_train`/`n_val`/
`n_test`/`n_edge_holdout` and `optim.epochs` further still for something
closer to the original full scope. Nothing in the code depends on any
particular values in these config files. See "How to run" below for the
demo vs. pilot commands, and "Configuration" for what each `configs/*.yaml`
file is for.

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

### (B) Neural surrogate — hybrid (physiology-conditioned, physics-informed) architecture

"Hybrid" here means the model itself, not the comparison: a data-driven
Fourier Neural Operator conditioned on the same geometry/physiological
parameters as the mechanistic model, trained with physics-informed loss
terms alongside the usual data loss. As of the continuous-surrogate work
(`docs/continuous_surrogate_design.md`), the model is a **shared trunk with
two interchangeable heads**: `encoder.py` + `operator_core.py`'s FNO blocks
now stop one layer earlier (`neural/model.py`'s `SurrogateBackbone`),
producing a latent *feature grid* rather than projecting straight to
physical fields, and either **(a)** a `Conv2d` projection head
(`ThrombusSurrogate`, today's original path, kept as a comparison baseline)
or **(b)** a `CoordinateDecoder` head (`neural/coordinate_decoder.py`,
`ContinuousThrombusSurrogate`, the new primary path) can consume that same
latent grid. See "The coordinate-decoder design" below for why (b) exists
and what it changes; `train.py` combines the losses for whichever head is
in use (`train`/`train_continuous`):

```
            9-scalar parameter vector θ
     (existing 8: geometry preset, physio params,
      inlet v -- plus normalized t, new)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ encoder.py                      │
        │ GeometryParamEncoder            │
        │ FiLM-style MLP modulation of    │
        │ a learned base grid + fixed     │
        │ sinusoidal coordinate embedding │
        └─────────────────────────────────┘
                         │ latent spatial grid
                         ▼
        ┌─────────────────────────────────┐
        │ Dropout2d                       │
        │ (MC-dropout UQ tap point)       │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ operator_core.py                │
        │ Fourier Neural Operator (FNO)   │
        │ trunk (SurrogateBackbone) --    │
        │ shared by both heads below      │
        │ [gnn backbone: unimplemented]   │
        └─────────────────────────────────┘
                         │ latent feature grid
                         │ (hidden_channels, H, W) --
                         │ shared trunk output, consumed
                         │ by either head below
                         ▼
        ── two interchangeable heads ──

   (a) grid-projection head
       [today's path, kept as a comparison
        baseline -- neural/model.py]

        ┌─────────────────────────────────┐
        │ ThrombusSurrogate: Conv2d       │
        │ projection -> (11, H, W)        │
        │ predicted field grid            │
        └─────────────────────────────────┘
                         │ predicted field grid
                         ▼
        ┌─────────────────────────────────┐
        │ train.py loss                   │
        │ MSE data loss (vs. (A)'s        │
        │ rasterized fields) +            │
        │ physics_losses.py (finite-      │
        │ difference mass-conservation +  │
        │ non-negativity penalties)       │
        └─────────────────────────────────┘

   (b) CoordinateDecoder head
       [new, primary path -- Phase 1-2,
        neural/coordinate_decoder.py]

       continuous query point (x, y) +
       this sample's geometry parameters
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ coordinate_encoding.py          │
        │ Fourier/SIREN positional        │
        │ encoding of (x, y)              │
        └─────────────────────────────────┘
                         │ positional features, merged with
                         │ grid_sample'd local features
                         │ (bilinear lookup into the trunk's
                         │ latent grid above) and
                         │ geometry_sdf.py's analytic
                         │ signed-distance-to-wall(x, y)
                         ▼
        ┌─────────────────────────────────┐
        │ coordinate_decoder.py           │
        │ CoordinateDecoder: MLP +        │
        │ residual blocks -> field        │
        │ values at exactly (x, y)        │
        └─────────────────────────────────┘
                         │ predicted point values
                         ▼
        ┌─────────────────────────────────┐
        │ train.py: train_continuous loss │
        │ per-point MSE (channel-         │
        │ exclusion for cap-affected      │
        │ species) + physics_losses.py:   │
        │ true autograd mass-conservation │
        │ residual (PINN collocation)     │
        └─────────────────────────────────┘

                         │
                         ▼
                         uncertainty.py: MC-dropout / deep-ensemble
                         wrapper repeats the forward pass for UQ
                         (works unchanged for either head)
```

### The coordinate-decoder design

The grid-projection head (a) always produces a fixed-resolution raster --
training data for it has to be rasterized too (`data/generate_dataset.
_rasterize`, `scipy.interpolate.griddata`), which introduces interpolation
error and wastes the FEM mesh's own resolution (different sampled
geometries mesh differently, so a shared fixed grid was always an
approximation of each mesh's actual resolution). The `CoordinateDecoder`
head (b) instead predicts field values at an arbitrary, continuous query
point `(x, y, t)`, so it can be trained directly against the mechanistic
solver's own mesh node coordinates -- `data/dataset.
PointCloudThrombusDataset` (ragged per-sample node counts, batched via a
flat-points-plus-`batch_index` convention) supplies this point-sampled
training data instead of a raster, and `generate_dataset.py` saves it by
default (`_build_pointcloud_sample`; the legacy raster path is now opt-in,
`--also-save-raster`, for `ThrombusSurrogateDataset`/baseline (a)).

Two side-benefits of this shift, beyond avoiding the interpolation error
itself:

- **No more `_fluid_mask`/`griddata` artifacts in training data.** The
  vessel+aneurysm domain doesn't fill its own rasterization bounding box
  (an L/T-shaped union), so the grid path needs `_fluid_mask` to flag which
  raster cells are real vs. `griddata`-filled background filler. Point-cloud
  training data has no such problem -- every training point is a genuine
  mesh node inside the fluid domain, by construction.
- **A real `"autograd"` physics residual.** Because (b) is a genuinely
  differentiable function of continuous `(x, y)` (and, for the encoder
  input, `t`), PDE residuals can be computed via true `torch.autograd.grad`
  on the model's own output at PINN-style collocation points, rather than
  finite differences on a fixed grid. See `neural/physics_losses.py`'s
  module docstring for exactly which residual is implemented this way
  (`mass_conservation`) and which residual modes remain unimplemented
  (`cdr_residual`, `surface_flux_bc_residual`) -- not duplicated here.

`mechanistic/geometry_sdf.py`'s analytic signed-distance-to-wall function
(reused by both the `CoordinateDecoder`'s SDF input feature and the
autograd residual's collocation-point sampling) only works because this
project's domains are idealized parametric shapes with a closed-form
boundary -- see "Known limitations" below for the scope this doesn't
extend to.

### Benchmark pipeline — dataset → train → report

The three CLI entry points (`thrombus-generate-dataset` →
`thrombus-train` → `thrombus-benchmark`) wire (A) and (B) above into an
end-to-end pipeline. Each entry point now has two paths -- the new
point-cloud path (primary, no `--continuous` flags needed on
`thrombus-train`/`thrombus-benchmark` shown below since it's a
per-command flag, not a separate command) and the legacy raster path
(kept for baseline (a) above, opt-in via `--also-save-raster`/
`--continuous`):

```
data/sampler.py                data/generate_dataset.py
(LHS sampling +      ──▶       batch-runs (A) mechanistic model per
 edge_holdout-tail split)       sample, checkpointed at data.n_snapshots
                                 points in time; writes data/processed/
                                 {split}/sample_NNN.npz as point-cloud data
                                 (Phase 1/3 schema: mesh node coords +
                                 values, no rasterization by default) --
                                 --also-save-raster additionally rasterizes
                                 onto a fixed grid (legacy/comparison path)
                                       │
                                       ▼
                        thrombus-generate-dataset  (CLI)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        data/dataset.py                       data/dataset.py
        PointCloudThrombusDataset              ThrombusSurrogateDataset
        (primary -- ragged mesh                (legacy/comparison --
        nodes + batch_index)                   fixed raster grid)
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                        thrombus-train [--continuous]  (CLI)
                        neural/train.py: train_continuous optimizes (B)
                        ContinuousThrombusSurrogate (primary path) or
                        train optimizes the grid ThrombusSurrogate
                        (comparison baseline), against data loss +
                        physics losses (neural/physics_losses.py)
                                       │
                                       ▼
                        checkpoints/{continuous_,}model.pt
                                       │
                                       ▼
                        thrombus-benchmark [--continuous]  (CLI)
                        benchmark/run_benchmark.py: evaluate on
                        test+edge_holdout, compute point-query or grid
                        field RMSE, edge_holdout_eval/calibration, time a
                        fresh mechanistic re-solve, render viz/plots.py
                        (opt-in extrapolation split: a separate script,
                        scripts/evaluate_extrapolation[_continuous].py --
                        not part of this main flow)
                                       │
                                       ▼
                        results/report[_continuous].md + PNGs
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
   better-sourced value. `verification/sensitivity_study.md` found this
   placeholder's guessed value doesn't materially matter: sweeping 0.5x-2x
   left `max_M_at`/`thrombosed_fraction`/peak-deposition location
   essentially unchanged, for both paper geometries.

6. **`D_a` calibration constant.** Eqs. (14)-(15) include a thrombus
   growth-rate scale factor `D_a` that the paper states was tuned ("different
   values of D_a were tested, and the one specified in Table 1 was
   determined to be the one that ensures smooth convergence and best
   reproduces the experimental data") without reporting the resulting
   numeric value in the table as extracted for this project. Defaulted to
   `1.0` (`configs/physio_params.yaml` `scale_terms.D_a`); should be
   recalibrated against Fig. 4/7/8/9 if closer quantitative agreement is
   desired. Unlike item 5 above, `verification/sensitivity_study.md` found
   this placeholder's guessed value matters a great deal: sweeping a
   10x-each-way bracket (0.1-10) swung `thrombosed_fraction` from ~0 to a
   large fraction of the wall in both paper geometries -- a qualitative,
   not just quantitative, difference in the model's headline "does
   thrombus form" result, within a range the paper itself says it explored.

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

- **Thrombin/fibrin generation is not well-calibrated -- now flagged in the
  data rather than silently clipped.** The surface thrombin-generation flux
  (Eqs. C.5-C.6, via `phi_at`/`phi_rt` on M_at/M_r) combined with Eq.
  (A.10)'s Γ -- which *saturates* rather than accelerates as `[T]` grows,
  since T appears in Γ's denominator -- can produce runaway growth in `[T]`
  (and downstream `[FI]`) over timescales of seconds in the current coupling
  scheme, well beyond physiological ranges (~0.1-10 µM). An isolated
  single-node diagnostic (`scripts/diagnose_thrombin_reaction_stiffness.py`;
  no spatial transport) compared `species_transport.reaction_step`'s
  implicit substepping against an independent high-accuracy
  `scipy.integrate.solve_ivp` reference on the identical local reaction ODE,
  with and without a representative C.5-C.6-shaped term. This ruled out the
  substepping scheme as the cause (the two integrators track each other
  closely) and showed the local reaction system is actually *self-limiting*
  once local PT/FG are exhausted -- not intrinsically divergent on its own.
  The sustained runaway therefore appears to require *this project's*
  transport-coupling (continuous PT/FG resupply at the wall across many
  macro steps, a mechanism the isolated diagnostic can't exercise),
  compounded by how sensitive the C.5-C.6 term's magnitude is to the
  `CM2_TO_M2` areal SI conversion (`coupled_solver.py`; a ~1e4x lever). This
  was not resolved despite the unit reconciliation above (independently
  verified correct via an isolated mass-balance check), and no single
  definite bug has been pinned down -- so rather than inventing an
  unsupported fix, `run_coupled_simulation` now returns a
  `thrombin_fibrin_reliable` flag on its `CoupledSimulationHistory` (False
  whenever `concentration_cap` actually bound for T or FI during the run),
  threaded through `generate_dataset.py`'s saved `.npz` samples and
  `dataset.py`'s `ThrombusSurrogateDataset.__getitem__` output. `[T]`/`[FI]`
  values from `run_coupled_simulation`/`generate_dataset.py` should still
  not be treated as physically meaningful without further recalibration
  against a reference implementation this project does not have access to
  -- but downstream training/evaluation code can now filter or weight
  samples by `thrombin_fibrin_reliable` instead of silently trusting capped
  values. **`[PT]` (prothrombin) is affected too, not just `[T]`/`[FI]`**:
  checking actual `clip_count_*` QC data across 18 LHS-sampled runs
  (`docs/continuous_surrogate_design.md` Phase 4) found the concentration
  cap binds for `PT` in 100% of samples, identically to `T`/`FI` -- and
  never for `RP`/`AP`/`APR`/`APS`/`AT`/`FG` in that same check. This is
  mechanistically consistent with the root cause above: PT is the
  substrate consumed by the same uncalibrated C.5-C.6 thrombin-generation
  pathway (`j_pt_chem_si` in `coupled_solver.py`), so its runaway
  consumption tracks T's runaway generation. `thrombin_fibrin_reliable`
  itself is not renamed/widened to cover this (still only tests T/FI, per
  its docstring) but any code excluding "the unreliable species" by name
  should include `PT` alongside `T`/`FI` -- see `neural/train.py`'s
  `DEFAULT_EXCLUDED_TEMPORAL_CHANNELS` and `configs/continuous.yaml`'s
  `data.excluded_temporal_channels`, both defaulting to `conc_T`,
  `conc_PT`, `conc_FI` excluded (zero-weighted, still loaded/predicted for
  diagnostics) from `train_continuous`'s per-point training loss. The grid
  path's `train()` has no analogous mechanism -- it trains on all 11
  channels unconditionally, so this caveat applies to its `[T]`/`[PT]`/
  `[FI]` output channels unfiltered.
- **The analytic wall-distance function only covers this project's
  idealized geometries.** `mechanistic/geometry_sdf.py`'s closed-form
  signed-distance-to-wall function (used by `CoordinateDecoder`'s SDF input
  feature and the autograd physics residual's collocation-point sampling)
  works by exploiting an exact algebraic description of the vessel+aneurysm
  domain's boundary (a rectangle union a circular-arc sac). This is
  specific to the idealized parametric shapes `mesh.py` generates -- it
  does **not** generalize to patient-specific or otherwise arbitrary
  geometries, which have no closed-form boundary to differentiate through.
  Supporting such geometries would need a different, non-analytic domain
  representation (e.g. a signed-distance field learned from or sampled
  against an actual mesh/point cloud), not an extension of this function.
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
configs/                  Plain YAML configs, loaded via PyYAML (geometry, physio params, training)
src/thrombus_bench/
  mechanistic/             2D FEM thrombus formation solver (scikit-fem)
  data/                    Dataset generation (LHS sampling + batch mechanistic runs) for the surrogate
  neural/                  Neural surrogate (encoder + FNO core [GNN unimplemented] + physics-informed losses + UQ)
  benchmark/               Accuracy/runtime/edge-of-domain/calibration metrics + report generation
  viz/                     Plotting utilities
tests/                     pytest suite
notebooks/                 Exploratory notebooks
scripts/                   One-off analysis/diagnostic scripts (not part of the installed package)
verification/              Committed numerical-verification artifacts (e.g. convergence studies)
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

### Mesh / time-step self-convergence study

```bash
python scripts/convergence_study.py
```

Runs both paper geometries across 3 mesh resolutions x 3 macro time steps
(18 runs, ~1 minute total on CPU) and writes `verification/convergence_study.md`
-- see that script's module docstring for methodology and what's/isn't
validated by this.

### Placeholder-parameter sensitivity sweep

```bash
python scripts/sensitivity_study.py
```

One-at-a-time sweep of the two documented placeholder parameters
(`k1_ADP_per_s`, `D_a` -- "Assumptions & Deviations" items 5/6 above) on
both paper geometries (10 runs, ~25s total on CPU), writing
`verification/sensitivity_study.md` -- see that script's module docstring
for methodology and range choices.

### Run the full benchmark pipeline (dataset → training → report)

Two sizes are available (see "Scale caveat" above): **demo** (the CLI
default, `configs/demo_cpu.yaml` -- a pipeline smoke test, a few minutes,
not a real benchmark) and **pilot** (`configs/pilot.yaml` -- a real,
if still CPU-modest, benchmark, ~15 minutes). Both use the same three
commands; only the config and output directories differ.

```bash
# --- Demo (pipeline smoke test, ~a few minutes; uses the CLI defaults) ---

# 1. Generate the demo-scale dataset:
thrombus-generate-dataset --output-dir data/processed

# 2. Train the neural surrogate:
thrombus-train --dataset-dir data/processed --checkpoint checkpoints/model.pt

# 3. Run the benchmark, producing results/report.md + PNGs:
thrombus-benchmark --checkpoint checkpoints/model.pt --dataset-dir data/processed
```

```bash
# --- Pilot (real benchmark, ~15 minutes -- see configs/pilot.yaml's header
#     comment for the estimate's basis; a separate --output-dir/
#     --dataset-dir keeps this from overwriting the demo dataset above) ---

# 1. Generate the pilot-scale dataset:
thrombus-generate-dataset --training-config configs/pilot.yaml --output-dir data/processed_pilot

# 2. Train the neural surrogate:
thrombus-train --config configs/pilot.yaml --dataset-dir data/processed_pilot --checkpoint checkpoints/model_pilot.pt

# 3. Run the benchmark, producing results/report.md + PNGs:
thrombus-benchmark --training-config configs/pilot.yaml --checkpoint checkpoints/model_pilot.pt --dataset-dir data/processed_pilot --output-dir results_pilot
```

`thrombus-generate-dataset` and `thrombus-benchmark` also accept
`--physio-config`/`--geometry-config` overrides (default to the
`configs/*.yaml` files in this repo, unaffected by the demo/pilot choice
above). See `--help` on each command, or the corresponding module's
`main()` in `src/thrombus_bench/{data,neural,benchmark}/`.

### Run the continuous-surrogate pipeline (point-cloud, primary path)

`configs/continuous.yaml` is the demo-scale config for the continuous
model (`ContinuousThrombusSurrogate`) -- forked from `demo_cpu.yaml`
rather than overloading it with continuous-specific keys, and smaller
still (see that file's header comment: the point-cloud path checkpoints
each sample multiple times, so a comparably-sized dataset costs more
wall-clock time to generate than the raster path's final-checkpoint-only
default).

```bash
# 1. Generate the dataset (point-cloud .npz by default -- add
#    --also-save-raster too if you also want the 4th, grid-FNO comparison
#    row in step 3 below):
thrombus-generate-dataset --training-config configs/continuous.yaml \
    --output-dir data/processed_continuous

# 2. Train the continuous surrogate (--continuous selects train_continuous):
thrombus-train --continuous --config configs/continuous.yaml \
    --dataset-dir data/processed_continuous \
    --checkpoint checkpoints/continuous_model.pt

# 3. Run the benchmark, producing results/report_continuous.md + PNGs
#    (add --grid-checkpoint <path> for the optional 4th comparison row --
#    requires a separately-trained ThrombusSurrogate and a dataset
#    generated with --also-save-raster):
thrombus-benchmark --continuous --training-config configs/continuous.yaml \
    --dataset-dir data/processed_continuous \
    --checkpoint checkpoints/continuous_model.pt
```

**Interpreting the report's RMSE numbers:** `results/report_continuous.md`'s
"Accuracy"/"Model comparison" sections use **point-query RMSE**
(`benchmark.metrics.field_rmse_pointwise`) -- computed directly against
each held-out simulation's own exact mesh node coordinates, no
interpolation involved. If a `--grid-checkpoint` was given, its numbers
appear in a separate "Legacy grid-projection FNO" section using the
original **grid RMSE** (`field_rmse`, `griddata`-interpolated onto a fixed
raster). **These two numbers are not directly comparable in magnitude** --
they're computed against different ground truth (exact nodes vs.
interpolated raster cells) -- only compare within one section, not across
them (e.g. don't conclude the continuous model is more/less accurate than
the FNO just because one number is larger than the other).

### Genuine-extrapolation evaluation (opt-in)

The edge-of-domain holdout above is still drawn from the same sampled
parameter box as train/val/test -- it does not test extrapolation beyond
the trained range. This opt-in variant does: `heparin_conc_uM` is
restricted to `0.1-0.38` uM for train/val/test, and a separate
"extrapolation" split is drawn from the withheld `0.38-0.5` uM remainder
(demo scale, ~1-2 minutes total; requires its own separately-trained
checkpoint -- see `benchmark/extrapolation_eval.py`'s module docstring for
why reusing the default demo/pilot checkpoint wouldn't test extrapolation
at all).

```bash
thrombus-generate-dataset --extrapolation-param heparin_conc_uM \
    --output-dir data/processed_extrap_heparin

thrombus-train --dataset-dir data/processed_extrap_heparin \
    --checkpoint checkpoints/model_extrap_heparin.pt

python scripts/evaluate_extrapolation.py
```

A continuous-path counterpart (`evaluate_extrapolation_degradation_continuous`)
exists too, via `scripts/evaluate_extrapolation_continuous.py` -- same
prerequisite shape, using `--continuous`/`configs/continuous.yaml` for the
generate/train steps:

```bash
thrombus-generate-dataset --extrapolation-param heparin_conc_uM \
    --output-dir data/processed_extrap_heparin_continuous

thrombus-train --continuous --config configs/continuous.yaml \
    --dataset-dir data/processed_extrap_heparin_continuous \
    --checkpoint checkpoints/continuous_model_extrap_heparin.pt

python scripts/evaluate_extrapolation_continuous.py
```

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
- `configs/demo_cpu.yaml`: neural surrogate architecture, physics-loss
  weights, optimizer, and dataset split sizes (including the edge-of-domain
  holdout split), at pipeline-smoke-test scale -- see "Scale caveat" above.
  This is the CLI default for `--training-config`/`--config` across all
  three entry points. `configs/training.yaml` is kept as a symlink to this
  file, for anything still written against the old path.
- `configs/pilot.yaml`: the same structure as `configs/demo_cpu.yaml`, at
  pilot-benchmark scale -- see "Scale caveat" above and this file's own
  header comment for its sizing rationale.
- `configs/ci_smoke.yaml`: an even smaller variant used only by
  `.github/workflows/ci.yml`'s end-to-end smoke test (not meant for
  interactive use).
- `configs/continuous.yaml`: the demo-scale config for the continuous
  surrogate (`ContinuousThrombusSurrogate`/`train_continuous`) -- forked
  from `demo_cpu.yaml` rather than adding continuous-specific keys to it,
  smaller still (see that file's own header comment). New keys beyond the
  grid configs' shape:
  - `data.n_snapshots`: checkpoints saved per sample by
    `generate_dataset.py` (`_output_every_n_steps_for_snapshots`); `1`
    reproduces the original final-checkpoint-only behavior. Also settable
    via that CLI's `--n-snapshots`, which falls back to this value if not
    passed explicitly.
  - `data.points_per_sample`: `PointCloudThrombusDataset`'s random
    per-checkpoint node subsample size (redrawn every access, not fixed at
    dataset construction -- see that class's docstring); `null`/omitted
    means "use every node."
  - `data.excluded_temporal_channels`: `FIELD_NAMES` channel names
    zero-weighted in `train_continuous`'s per-point data loss (still
    loaded/predicted, just not optimized against) -- defaults to
    `[conc_T, conc_PT, conc_FI]`, the species the concentration-cap QC
    check found unreliable (see "Known limitations").
  - `model.encoder.param_dim: 9`: the one required difference in the
    shared-trunk config block vs. the grid path's `8` -- the existing 8
    scalars plus normalized time.
  - `model.coordinate_encoding.num_frequency_bands`: Phase 1's Fourier
    positional encoding's `L` (`neural/coordinate_encoding.
    DEFAULT_N_FREQUENCIES`, default `8` -- see that module's docstring for
    the reasoning).
  - `model.coordinate_decoder.mlp_hidden`/`n_residual_blocks`:
    `CoordinateDecoder`'s MLP width/depth.
  - `model.predict_M_at_wall`: same meaning as the grid config's key of the
    same name (an opt-in extra output channel for `M_at`), but defaults to
    `true` here (vs. `false` for the grid config) since this config exists
    specifically to exercise the continuous path's features end to end.
  - `physics_loss.residual_mode: autograd`: the only mode `train_continuous`
    supports (`finite_difference` is grid-path-only); enables
    `physics_losses.continuous_mass_conservation_loss`.
  - `physics_loss.n_collocation_points`: PINN-style collocation points
    sampled per training-batch sample for the autograd mass-conservation
    residual (`physics_losses.sample_collocation_points`).
  - `physics_loss.weights.mass_conservation`: same meaning as the grid
    config's weight of the same name, applied to the autograd residual
    instead of the finite-difference one.

  `generate_dataset.py` also accepts `--also-save-raster` (CLI-only, not a
  YAML key): additionally computes and saves the legacy rasterized
  representation alongside the default point-cloud data, needed to produce
  data `ThrombusSurrogateDataset`/the grid-projection baseline (or
  `thrombus-benchmark --continuous --grid-checkpoint`) can consume. Off by
  default since it's the expensive part (`griddata`/mask-building).

## Contributors

- **Danilo Dursoniah** ([ddursoniah@gmail.com](mailto:ddursoniah@gmail.com))
  — Data Scientist and Computational Biologist. Implementation,
  calibration, and maintenance of the mechanistic solver, neural
  surrogate, and benchmark pipeline.

_Last updated: 2026-07-25_
