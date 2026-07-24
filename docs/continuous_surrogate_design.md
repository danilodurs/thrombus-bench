## Design summary

This section is the design doc referenced in "How to use this file" above
-- save it (or this whole file) into the repo before starting Phase 0, and
have Claude Code re-read it at the start of every phase, since it has no
memory of any conversation that produced it.

Split the model into two reused/new stages instead of one:

- **Stage 1 (mostly reused): branch/backbone.** `encoder.py`'s
  `GeometryParamEncoder` + `operator_core.py`'s FNO blocks still take
  `params` (now 9 scalars: the existing 8 + normalized `t`) and produce a
  latent feature grid -- but stop one layer earlier than today's
  `ThrombusSurrogate`, keeping `(hidden_channels, H_latent, W_latent)`
  features instead of projecting straight to the 11 physical output
  channels.
- **Stage 2 (new): coordinate decoder head.** A small MLP taking a
  Fourier/sinusoidal-encoded continuous query `(x, y)`, a bilinearly
  interpolated slice of Stage 1's feature grid at that point
  (`torch.nn.functional.grid_sample`), and an analytic signed-distance-to-wall
  value at that point (computed from the sample's geometry parameters,
  since these are idealized parametric shapes) -- outputs the physical
  field values at exactly `(x, y, t, \text{params})`.
- **Training data becomes point samples, not rasters.** Query points are
  the FEM mesh's own node coordinates (ground truth with zero
  interpolation error) -- `_rasterize`'s `griddata(method="nearest")` step
  and `_fluid_mask` become optional/legacy (kept only for a post-hoc
  visualization utility, not for training).
- **`physics_losses.py`'s `"autograd"` residual mode gets implemented for
  real**, since Stage 2 is a genuinely differentiable function of
  `(x, y, t)` -- true pointwise `torch.autograd.grad` derivatives replace
  the finite-difference/mask/erosion machinery for anything computed this
  way.
- **Keep today's grid-projection path available as a comparison baseline**
  (not a hedge -- this repo already has a strong pattern of keeping
  trivial baselines around, `neural/baselines.py`; the old FNO-grid path
  is a legitimate one more baseline to benchmark the new continuous model
  against, and costs little to keep since Stage 1 is shared).
- **Channel-exclusion caveat**: as of the last verified repo state,
  `README.md`'s "Known limitations" section documents that thrombin/fibrin
  concentrations (`[T]`/`[FI]`) hit a safety-clip concentration cap during
  simulation (see `coupled_solver.py`'s `concentration_cap` and
  `thrombin_fibrin_reliable` flag) -- confirm this is still the documented
  state by reading that section and `generate_dataset.py`'s per-species
  `clip_count_*` QC output before finalizing which species to exclude (`T`
  and `FI` are the ones directly named there; whether `FG`/`AT`/`PT` are
  also affected needs checking against the actual clip-count data, not
  assumed). Whichever species end up excluded, they should stay excluded
  from the primary training loss by default in this design too, since it
  doesn't change that underlying mechanistic-model issue -- the exact
  mechanism for building that per-checkpoint exclusion is specified fresh
  in Phase 1/3 below (nothing to reuse from elsewhere, since no
  checkpoint-level reliability tracking exists in the repo yet).

### Finalized Phase 7 decisions (benchmark pipeline integration)

- **`run_benchmark_continuous`** (`benchmark/run_benchmark.py`) supersedes
  Phase 4's `benchmark_continuous_placeholder` (removed): a full report --
  `ContinuousThrombusSurrogate` + both continuous baselines via point-query
  RMSE (primary), an *optional* 4th row for a separately-trained grid FNO
  via its own legacy grid RMSE (`--grid-checkpoint`, requires the dataset
  to have been generated with `--also-save-raster` so both representations
  coexist in the same files) in its own clearly-labeled section, edge-
  holdout degradation, MC-dropout/deep-ensemble calibration, runtime, and
  an M_at/IoU section mirroring the grid path's. Written to
  `results/report_continuous.md` (distinct from the grid path's
  `report.md`, since they're different checkpoints/models).
- `edge_holdout_eval.py`/`extrapolation_eval.py` each got a
  `..._continuous` counterpart function (`PointCloudThrombusDataset` +
  `field_rmse_pointwise`), not a mode-branch -- same established pattern
  as `train`/`train_continuous`. `scripts/evaluate_extrapolation_continuous.py`
  mirrors the existing grid-path script, not wired into the main report
  (matching that script's own precedent of staying separate).
- `neural/uncertainty.py`'s `DeepEnsemble`/`MCDropoutWrapper` needed **no
  changes** -- confirmed by actually running both against
  `ContinuousThrombusSurrogate` (`tests/test_uncertainty.py`), not just
  assumed; both are already `*args`-generic. `_FieldChannelsOnly`
  (`run_benchmark.py`) needed one generalization: `forward(params)` ->
  `forward(*args)`, so it works for both call signatures.
- Confirmed (and tested against real generated files, not just the
  abstract sampler partition) that `split_train_val_test_edge_holdout`'s
  sample-level guarantee still holds for the point-cloud path: a sample's
  `.npz` (every checkpoint together) is written to exactly one split
  directory, so it structurally cannot leak across splits.

### Finalized Phase 6 decisions (baselines, metrics, visualization)

- **`ContinuousMeanFieldBaseline`**: pools every training `(sample,
  checkpoint, node)` row's normalized `(x, y)` position + field value into
  one `scipy.spatial.cKDTree`, predicts the mean of the `k` nearest pooled
  points -- chosen over "rasterize a low-res mean grid, then nearest-cell
  lookup" specifically because point-cloud data has no raster ground truth
  lying around any more (Phase 3 stopped generating it by default), so
  that option would reintroduce the exact `griddata` cost this design
  moved away from, just to build a baseline.
- **`ContinuousNearestNeighborBaseline`**: single combined-distance `k=1`
  search over `[params_with_time (9) ; x_norm, y_norm (2)]`, not a
  two-stage nearest-sample-then-nearest-node lookup -- avoids the failure
  mode where the nearest-by-parameters sample has no node near the query
  location.
- Both pool in **normalized** `[-1, 1]` coordinates
  (`neural.coordinate_decoder.normalize_query_points_to_unit_box`, factored
  out of `ContinuousThrombusSurrogate.forward` this phase for reuse), not
  raw meters -- different sampled geometries have different physical
  extents.
- **Bootstrap resampling unit**: audited (no bootstrap/CI code existed
  anywhere before this phase) and pre-emptively fixed via
  `benchmark.metrics.bootstrap_metric_by_sample`, which resamples whole
  `sample_id` rows only -- guidance for future code, not a fix to
  something broken.
- **`rasterize_continuous_model`** lives in `viz/rasterize_continuous.py`
  (new module, not `viz/plots.py`, to keep that module's pure-matplotlib-
  on-arrays style separate from model-querying logic) -- queries the model
  directly on a regular grid over the analytic bounding box, no FEM mesh
  needed at inference time at all, unlike the legacy `_rasterize` (which
  interpolates existing scattered mesh data). Uses `sdf >= 0` (inclusive)
  for the display mask specifically, since the grid's own edges (x=0, y=0)
  sit exactly on the analytic boundary by construction and a boundary
  pixel is still visually part of the vessel, unlike the `> 0`
  strictly-interior convention used elsewhere (e.g. collocation-point
  rejection sampling, Phase 5) where "exactly on the wall" genuinely should
  not count as "inside."
- `thrombus_height_error`/`time_to_onset_error` (`metrics.py`) are still
  unimplemented, but their docstrings' claim that the data doesn't exist is
  now stale -- the point-cloud path *does* save full per-checkpoint spatial
  fields since Phase 1/3. Noted, not implemented (separate, not-yet-assessed
  scope).

### cdr_residual assessment (Phase 5, deferred)

Checked (not assumed) whether `physics_losses.cdr_residual` is now
tractable for RP/AP/APR/APS via the same autograd machinery
`mass_conservation_penalty_autograd` uses, now that those species are
already excluded from the primary loss for other reasons. Finding:
`activation.chemical_source_terms`'s actual reaction terms for those four
species (as wired in `coupled_solver.py`'s `source_fn`) depend only on
`{RP, AP, APR, APS, bulk shear rate}` -- genuinely decoupled from
`T`/`PT`/`AT`/`FG`/`FI`, more favorable than assumed going in. But three
real gaps remain: `d/dt` needs autograd through the *whole* Stage 1
backbone (time is an encoder input, not a Stage 2 query coordinate --
structurally different/heavier than the spatial derivatives already
built); the bulk shear rate `sqrt(2 D:D)` (Eq. 2) needs porting to
differentiable form, a physics-formula reimplementation with its own
correctness risk; and a Laplacian (second-derivative autograd) is needed
per species, for 4 species. **Deferred, not implemented** -- assessed as
real scope creep on top of the mass-conservation work already in this
phase; needs explicit go-ahead before building.

### Finalized M_at design choice (Phase 4)

Chose (a) -- a 12th `CoordinateDecoder` output channel, active/meaningful
only at wall points -- over (b) a separate wall-only decoder head, for two
reasons: (1) it's a one-line change (`output_channels + 1`) that exactly
mirrors `ThrombusSurrogate`/`neural/model.py`'s existing
`predict_M_at_wall` convention, whereas (b) would need `CoordinateDecoder`
to branch its forward pass on a per-point "is this a wall query" flag and
the dataset/collate layer to route two distinct query populations through
different heads; (2) wall points are not a disjoint geometric set needing
separate SDF/grid_sample plumbing -- `wall_dofs` (new schema field above)
shows they're already a subset of the same bulk mesh nodes `fields`
already covers, so a per-point M_at target array (0 off-wall, real value
on-wall) slots directly into the *existing* single-decoder-head, flat
point-cloud pipeline with no new query population, no mask threaded
through `CoordinateDecoder.forward`, and no change to Phase 2's module at
all -- `neural/train.py`'s loss just concatenates it as target channel 12,
identical in spirit to how `ThrombusSurrogate`'s training loop already
handles `M_at_wall` today (plain MSE, 0-filled off the wall band, no
special masking). The already-present SDF input feature (Phase 2) gives
the network a natural, if soft, "how close to the wall is this point"
signal to condition that 12th channel on, for free.

### Finalized channel-exclusion default (Phase 4)

Checked actual `clip_count_*` QC data across 18 LHS-sampled runs spanning
two seeds/durations (not assumed): the concentration cap binds for `T`,
`PT`, and `FI` in 100% of samples checked, and never for
`RP`/`AP`/`APR`/`APS`/`AT`/`FG`. `PT` is affected as badly as `T`/`FI`,
contradicting a naive reading of README.md's "Known limitations" (which
only names `T`/`FI` -- now corrected there too). `neural/train.py`'s
`configs/*.yaml` `data.excluded_temporal_channels` therefore defaults to
`["conc_T", "conc_PT", "conc_FI"]` (`FIELD_NAMES` channel names).

### Finalized Phase 1 schema (confirmed after review)

Phase 1 implemented and got sign-off on the coordinate encoding (Fourier/
SIREN-style, `neural/coordinate_encoding.FourierFeatureEncoding`, default
`L=8`), the analytic SDF (`mechanistic/geometry_sdf.
signed_distance_to_wall`, sign convention: positive inside the fluid
domain, negative outside, zero on the wall), and this point-cloud `.npz`
schema (implemented as the standalone, tested `data/generate_dataset.
_build_pointcloud_sample` -- not yet wired into the actual
`generate_dataset`/`_run_one_sample` save path as of Phase 1; that wiring
is Phase 3's job):

- ``params``: ``(8,)`` float64 -- unchanged, `PARAM_ORDER`.
- ``node_coords``: ``(n_nodes, 2)`` float64 -- FEM mesh vertex coordinates
  (meters), `tagged_mesh.mesh.p.T`.
- ``triangles``: ``(n_triangles, 3)`` int32 -- mesh connectivity
  (`tagged_mesh.mesh.t.T`), kept only so a post-hoc legacy visualization
  utility can still rasterize a sample the old way without needing the
  original mesh object -- not used in training.
- ``fields``: ``(n_snapshots, n_nodes, 11)`` float32 -- physical field
  values at every node, at every checkpoint, in `data/dataset.FIELD_NAMES`
  channel order (velocity_x, velocity_y, then conc_{species} in
  `generate_dataset._ALL_SPECIES` order -- the two orderings already
  agree).
- ``wall_dofs``: ``(n_wall_nodes,)`` int64 -- indices into
  ``node_coords``/``fields``'s first axis for the wall nodes (added
  Phase 4, for the M_at-as-12th-channel training target below).
- ``wall_node_coords``: ``(n_wall_nodes, 2)`` float64 -- wall DOF
  coordinates (fixed across checkpoints within a sample, since the
  mesh/wall_dofs don't change during a run).
- ``M_at_wall_values``: ``(n_snapshots, n_wall_nodes)`` float32 --
  `surface_ode.SurfaceState.M_at` at each wall node, per checkpoint.
- ``time_s``: ``(n_snapshots,)`` float64 -- checkpoint times.
- ``thrombin_fibrin_reliable_at_checkpoint``: ``(n_snapshots,)`` bool --
  from `CoupledSimulationHistory` (new in Phase 1; monotonic True -> False
  within a run, see that field's docstring in `coupled_solver.py`) -- this
  is the mechanism the channel-exclusion caveat above resolves to: exclude
  a checkpoint's T/FI (and any other affected species, per that caveat's
  verification step) from the primary training loss when this flag is
  `False` for that checkpoint.
- Per-run QC keys, unchanged in meaning from today's raster path:
  ``converged``, ``flow_n_iterations``, ``flow_residual`` (from the
  *final* checkpoint's flow), ``thrombin_fibrin_reliable``,
  ``clip_count_{species}``, ``conc_{species}_min``/``max``
  (final-checkpoint nodal extrema), ``max_M_at``, ``thrombosed_fraction``.

Deliberately dropped from this schema (vs. today's raster `.npz`):
``fluid_mask`` (only meaningful relative to a fixed raster grid) and the
rasterized ``M_at_wall``/bulk-field grids themselves (superseded by the
node-native arrays above; still reconstructable post-hoc from
``node_coords`` + ``triangles`` + ``fields`` for visualization, Phase 6).

Ragged-ness: ``n_nodes``/``n_wall_nodes`` vary per sample (different
sampled geometries mesh differently) -- fine within one sample's `.npz`,
but requires a custom `collate_fn` (flat-points + `batch_index`, matching
Stage 2's own batching convention) rather than PyTorch's default
collation once a `DataLoader` batches multiple samples together. That
dataloader-side handling is Phase 3's job, not Phase 1's.

Also finalized in Phase 1: `generate_dataset._output_every_n_steps_for_
snapshots(n_steps, n_snapshots)` derives `run_coupled_simulation`'s
`output_every_n_steps` from a target checkpoint count (`n_snapshots<=1`
reproduces today's final-only formula exactly); a `--n-snapshots` CLI flag
exists but, like `_build_pointcloud_sample`, was not yet wired into the
actual generation pipeline as of Phase 1.

Run the phases below in order. **Pause for my review after Phase 1
(design lock-in) and after Phase 5 (autograd physics residual)** -- those
are the two places with the most room for a subtle correctness bug to hide
silently. Everything else builds mechanically once those are confirmed
right.

Standing instructions (same as prior task lists; keep/extend
`CLAUDE.md` if present): don't change governing equations/physical
constants without explicit instruction; every behavior change needs a
test; run `pytest` before declaring a task done; keep docstrings in sync
with actual behavior; prefer small reviewable diffs -- for a feature this
size, "small" means "one phase," not "one file."
