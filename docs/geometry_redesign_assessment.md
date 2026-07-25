# Geometry redesign: design assessment (Phase 4a)

**Status: assessment only. No implementation code was written for this
phase.** This document answers the questions Phase 4a was scoped to
investigate; Phase 4b (implementation) starts only after explicit approval
of a specific option below.

## 1. Current geometry model (baseline)

`mesh.py::_aneurysm_geometry_points`/`_build_boundary_polygon` and
`geometry_sdf.py::signed_distance_to_wall` both derive the sac from exactly
one number, `aneurysm_diameter_mm` (`R = aneurysm_diameter_mm / 2`): the
upper half of a circle of radius `R` centered at `(L/2, D)`, sitting flush
on the vessel's top wall. This forces four things to be equal that a real
saccular aneurysm's shape parameters need not be:

- neck width = sac (max) width = `2R`
- sac height = `R` (exactly half the width, by construction of a
  half-circle)
- left/right symmetry about the sac centerline (no asymmetry)
- the sac's local "up" direction = the vessel's normal (no tilt)

Both paper presets (`aneurysm_7mm`, `aneurysm_10mm`) are just two choices
of this one number.

## 2. Proposed shape family: two options, explicit tradeoff

I looked at three families (free-form spline through control points, a
Fourier/"perturbed-circle" descriptor, and a piecewise-ellipse-arc) and
narrowed to two worth presenting, because the choice has a direct,
material consequence on Section 4 (the SDF) and Section 5 (preserving the
paper presets exactly) — it isn't just a cosmetic shape-fitting choice.

### Option A (recommended starting point): asymmetric half-ellipse

Replace the single circular arc with two elliptical quarter-arcs sharing
an apex, each with its own horizontal semi-axis:

```
right half (theta in [0, pi/2]):  x = xc + a_right*cos(theta), y = D + b*sin(theta)
left half  (theta in [pi/2, pi]): x = xc + a_left *cos(theta), y = D + b*sin(theta)
```

- `b` (independent of `a_left`/`a_right`) gives **sac height** decoupled
  from width -- the paper's own aneurysm literature calls `height/neck`
  the "aspect ratio," a real rupture-risk descriptor, so this alone is a
  meaningful upgrade.
- `a_left != a_right` gives **asymmetry** directly.
- Both halves meet at `theta = pi/2` with the same tangent direction
  (horizontal) regardless of `a_left`/`a_right` -- I checked this
  analytically (`dy/dx -> 0` from both sides since `dy/dtheta = b*cos(theta)
  = 0` at `theta=pi/2` independent of the semi-axis values), so the apex is
  automatically smooth (no extra continuity engineering needed).
- **What this option does *not* give you:** neck width and sac (max) width
  stay the same parameter, exactly as today (`a_left`/`a_right` are both
  the base chord half-width *and* the widest point of their respective
  quarter-ellipse -- an ellipse standing on its own diameter cannot bulge
  past its base). **Tilt is not included either** (see Section 6).
- **Degenerates exactly to today's shape** when `a_left = a_right = b = R`
  (a circle is an ellipse with equal semi-axes) -- bit-for-bit, not an
  approximation. This matters a lot for Section 5.
- Nearest-point-to-ellipse-arc has no simple closed form, but it *is* a
  well-known, fast, iteratively-converging problem (a handful of Newton
  steps on the ellipse's own parametrization) -- `geometry_sdf.py` stays
  "semi-analytic": piecewise-parametric with a cheap numeric refinement
  per piece, not a dense-polyline approximation.

### Option B (fuller ask, more engineering): free-form control-point spline

A Catmull-Rom (or natural cubic) spline through 5 control points --
`(-neck/2, 0)`, `(-sac_width/2, h*height)`, `(asymmetry, height)`,
`(+sac_width/2, h*height)`, `(+neck/2, 0)` -- genuinely decouples neck
width from sac width, including a sac that bulges out *past* its own
neck (a real, common saccular-aneurysm silhouette Option A cannot
produce).

Cost:
- No closed-form (or fast-Newton) nearest-point formula exists for a
  general cubic spline in the way it does for an ellipse -- `geometry_sdf.py`
  would become a **numerical** SDF (dense polyline sample of the curve +
  `scipy.spatial.cKDTree` nearest-neighbor query, the same technique
  `test_geometry_sdf.py`'s own brute-force reference already uses to
  *check* the analytic formula). This is a real architectural change to a
  module whose docstring specifically frames itself as closed-form/exact
  with "no mesh lookup."
- `geometry_sdf.py`'s SDF is called on the model's hot path
  (`neural/coordinate_decoder.py::_sdf_per_point`, once per forward pass,
  looped per unique sample geometry in a batch) -- a numerical SDF would
  need per-geometry caching of its polyline/KD-tree to avoid rebuilding it
  on every call, which is new state a currently-pure function doesn't
  have.
- A spline fit through points sampled off today's exact circle does
  **not** reproduce that circle exactly (Catmull-Rom is an approximating,
  not interpolating-with-curvature-matching, spline) -- reproducing the
  two paper presets bit-for-bit would need either a special-cased "if
  parameters match today's defaults, use the old circle formula" branch,
  or accepting a small (probably sub-percent, but non-zero) numeric
  deviation in the two presets' mesh/SDF values. Given the plan's
  explicit "preserved exactly as-is" requirement, I'd lean toward the
  special-cased branch if B is chosen -- but that is itself added
  complexity Option A avoids entirely.
- Free-form control points also need new **self-intersection guarding**:
  nothing currently stops `sac_width >> neck_width` + small `height` from
  producing a boundary polygon that folds back on itself, which would
  silently break `_build_boundary_polygon`'s "simple closed polygon,
  traversed once" assumption. `_aneurysm_geometry_points` already raises
  `ValueError` for one such case (aneurysm too large for the vessel); a
  general spline needs a broader validity check (e.g. a cheap segment-
  intersection scan, or `shapely`'s polygon-validity check -- a new,
  currently-unused dependency).

### My recommendation

Start with **Option A**. It delivers 3 of the 5 requested knobs (width,
height, asymmetry) with no new dependency, an exactly-preserved SDF
architecture, and a mechanically verifiable exact-reduction to both paper
presets. It explicitly does **not** deliver independent neck-vs-sac width
(a real anatomical feature) or tilt. If, after seeing Option A's
qualitative results, independent neck/sac width genuinely matters, Option
B is a well-scoped follow-up with the tradeoffs above made explicit up
front rather than discovered mid-implementation.

(A "perturbed circle" Fourier-descriptor family -- `r(theta) = R*(1 + sum
c_k * cos(k*theta))` -- was also considered: parameter-efficient and
smooth, but its nearest-point problem is just as numerical as the spline's,
so it doesn't avoid Option B's `geometry_sdf.py` cost while also not
fully solving independent neck/sac width without extra terms. Mentioned
for completeness; I don't recommend it over A or B.)

## 3. Does the existing Delaunay mesher survive this?

**Mostly yes, for either option, with two bounded, mechanical changes --
no rewrite.** I checked each stage of `mesh.py::build_aneurysm_mesh`
against a general (non-circular) simple boundary curve:

- `_fill_interior_points`'s point-in-polygon test
  (`matplotlib.path.Path.contains_points`) and `Delaunay`'s own
  triangulation are **already shape-agnostic** -- they work for any simple
  polygon, concave included, not just convex ones. A concave/overhung
  boundary (Option B) is not a problem here.
- `_build_boundary_polygon`'s graded arc-point spacing (the `theta` warp
  that concentrates points near both neck ends) is currently written in
  terms of a circular parameter `theta in [0, pi]`; generalizing it to an
  arbitrary parametric curve `t in [0, 1]` (arc-length or control-parameter
  based) with the same warp is a mechanical substitution, not new
  algorithm design.
- The **one place that genuinely needs new logic**: `build_aneurysm_mesh`'s
  post-hoc boundary tagging (`is_on_arc`, today a closed-form
  `|dist_to_circle_center - R| < tol` test) has to become a "distance to
  the new curve" test. For Option A this is still closed-form-ish (ellipse
  distance, same machinery as the SDF); for Option B this is the same
  dense-polyline distance test the SDF itself would need.

**No concrete limitation was found that requires a CAD/CSG mesher.**
I recommend **not** adding a `gmsh`/`pygmsh` dependency for this feature.
`gmsh` would become genuinely justified for a literal rigid-body tilted
neck (a true 2D boolean of a rotated shape against the straight vessel
wall -- see Section 6) or a curved parent vessel (Section 7) -- neither of
which this phase's shape work actually requires if tilt is deferred or
reframed per Section 6.

## 4. Implications for `geometry_sdf.py`

Covered above per option. Summary: **Option A keeps the module's design
promise (closed-form, no mesh/lookup, cheap per-point) intact**, with the
one addition of an iterative (not closed-form) ellipse-nearest-point
solve per piece. **Option B breaks that promise** and turns the module
into a numerical approximation requiring per-geometry caching to stay
fast on `_sdf_per_point`'s hot path.

One correction to the original plan's premise, found during inspection:
the plan describes the SDF as "used by both the legacy raster
`fluid_mask` and the continuous surrogate's exterior masking." I checked
`data/generate_dataset._fluid_mask` directly -- it uses
`matplotlib.tri.Triangulation.get_trifinder()` on the actual mesh, **not**
`geometry_sdf.py`, at all. `geometry_sdf.signed_distance_to_wall` is
actually used by (a) `viz/rasterize_continuous.py`'s *continuous* display
mask, (b) `neural/coordinate_decoder.py`'s per-query-point model input
feature (the hot-path use that matters for the Option A vs. B tradeoff
above), and (c) `physics_losses.py`'s collocation-point rejection
sampling. This doesn't change either option's viability, but it does mean
the legacy raster path has zero exposure to whichever SDF design is
chosen -- only the continuous surrogate path does.

## 5. Preserving `aneurysm_7mm`/`aneurysm_10mm` exactly

Both presets currently supply only `vessel_diameter_mm`,
`aneurysm_diameter_mm`, `vessel_length_mm`. Plan: give the new shape
parameters defaults derived from `aneurysm_diameter_mm` that reduce
**exactly** to today's half-circle (`neck_width_mm = sac_width_mm =
aneurysm_diameter_mm`, `sac_height_mm = aneurysm_diameter_mm / 2`,
`asymmetry = 0`), so `GeometryConfig.from_preset` needs no changes to
`configs/geometry.yaml` at all for the two existing presets -- they simply
never set the new (optional) keys.

This is exact and mechanically verifiable for **Option A** (`a_left =
a_right = b = R` is a circle, not an approximation of one). For **Option
B**, exact reduction needs either a special-cased fallback to the old
circle formula when parameters match the defaults, or accepting the paper
presets shift by whatever small numeric error the spline's approximation
of a circle introduces -- I'd treat that shift as unacceptable given the
plan's explicit "preserved exactly as-is, not replaced" requirement, so B
would need the special-cased branch.

Either way, Phase 4b should add a regression test asserting the new
parametrization's mesh boundary points and SDF values match today's
circle-based ones to floating-point precision at both presets, run
*before* touching anything else -- this is the one test that, if it ever
fails, means the "special case" story has silently broken.

## 6. Tilt: recommend reframing, not literal rotation

A literal rigid-body tilt (rotating the whole sac about the neck midpoint)
detaches the two neck-attachment points from the horizontal wall line
`y = D` -- the vessel top wall would then need a *tilted chord* stitched
into an otherwise-horizontal wall, which cascades into
`_build_boundary_polygon`'s top-wall-stub logic, the
`is_wall_sac`/`is_wall_vessel` boundary-tagging complement (today defined
as "everything on the boundary that isn't inlet/outlet/sac," which
implicitly assumes the sac's footprint sits cleanly at `y = D`), and
`geometry_sdf.py`'s straight-segment formulas for the two wall stubs (no
longer both horizontal). This is a materially bigger, more error-prone
change than the sac-shape work itself, and it's the closest thing in this
whole assessment to actually needing a real 2D boolean/CSG operation
(hence closest to justifying `gmsh`, per Section 3) -- I would not
recommend it as part of this phase.

**Cheaper alternative that gets the qualitative effect:** keep both neck
attachment points pinned to `y = D` (no wall change at all), and get a
"leaning" look by pushing the apex/asymmetry parameter further than a
symmetric aneurysm normally would (i.e., `tilt` becomes a relabeling of
an extreme `asymmetry` value, not a new geometric primitive). This costs
nothing beyond Option A's existing `asymmetry` parameter and avoids every
complication above. If a literal rotated neck is still wanted later, it's
a well-scoped, separate, `gmsh`-justified follow-up.

**Recommendation: defer literal tilt; reuse `asymmetry` for the
qualitative "leaning sac" effect now.**

## 7. Curved parent vessel: recommend deferring

Honest assessment, as asked for: **defer this, and by a wide margin
relative to the sac-shape work.**

- The source paper itself only validates straight vessels (Fig. 1's both
  geometries, Sec. 2.2's uniform-plug inlet/do-nothing outlet on flat end
  caps) -- a curved vessel has *zero* grounding in the paper, unlike the
  sac shape (which at least varies a paper-reported input, however
  idealized the exact contour is).
- It is not a contained change. It touches: `_wall_branches`'s
  top/bottom split (currently a hardcoded `y > D/2` test, meaningless for
  a curved centerline -- needs an arc-length/local-normal coordinate
  instead), `_axial_shear_gradient`'s x-based finite difference (needs to
  become arc-length-based), `flow_solver.py`'s inlet/outlet Dirichlet BC
  (currently "uniform velocity in `+x`" -- needs "uniform velocity along
  the local inlet normal," a real solver change, not just a mesh change),
  every straight-segment formula in `geometry_sdf.py` (not just the sac's),
  and the coordinate-decoder's whole bounding-box coordinate-normalization
  convention (`normalize_query_points_to_unit_box` assumes an axis-aligned
  `[0, L] x [0, D+R]` box, meaningless once the vessel isn't axis-aligned).
- This is qualitatively a different, larger initiative than "give the sac
  a smoother shape" -- essentially every mechanistic module's coordinate
  assumptions would need revisiting, not just `mesh.py`/`geometry_sdf.py`.

**Recommendation: out of scope for this phase; would need its own
separate design assessment if pursued later.**

## 8. Dataset schema / sampler / neural conditioning implications

- **`data/sampler.py`**: `DEFAULT_RANGES` is a flat, fully generic dict;
  `ParameterSpace`/`normalize_params`/`denormalize_params`/
  `latin_hypercube_sample` have no hardcoded parameter count or names.
  Adding `neck_width_mm`/`sac_height_mm`/`asymmetry` (Option A) is
  strictly additive dict entries -- **zero code changes** in this module.
  Reasonable ranges (e.g. `sac_height_mm` informed by real
  aspect-ratio literature, `asymmetry` as a small fraction) would need
  picking and documenting as a new, explicit project assumption (same
  treatment as the existing 11 numbered deviations in `README.md`), since
  the paper never studied or reported these.
- **`data/generate_dataset.py`**: `PARAM_ORDER` grows from 8 to 8+k
  entries -- **this is a breaking on-disk schema change**: existing
  generated `.npz` datasets (all currently gitignored/regenerable, not
  committed) have `params` arrays of the old length and would need
  regenerating, not migrating. New geometry-shape entries should be
  inserted *right after* `aneurysm_diameter_mm`/`vessel_diameter_mm`
  (positions 2, 3, [4]), not appended at the very end, to preserve the
  "leading N entries of `PARAM_ORDER` are geometry" convention that
  `data/dataset.py`'s `PointCloudThrombusDataset.__getitem__` already
  relies on (`geometry_mm = params_raw[:2]`, a hardcoded slice that would
  need to become `params_raw[:N]`).
- **`neural/encoder.py`**: `GeometryParamEncoder(param_dim, ...)` is a
  plain `nn.Linear(param_dim, ...)` with no assumption about which
  scalars occupy which position -- growing `param_dim` in
  `configs/*.yaml` is the **only** change needed here.
- **`neural/coordinate_decoder.py`**: this is where real code changes
  land. `ContinuousThrombusSurrogate.forward`'s `geometry_mm: (batch, 2)`
  argument is hardcoded to exactly
  `[aneurysm_diameter_mm, vessel_diameter_mm].` and both
  `normalize_query_points_to_unit_box` and `_sdf_per_point` construct a
  `GeometryConfig` from exactly those two fields. `geometry_mm` needs to
  grow to carry the new shape parameters, the bounding-box math needs
  updating for the new shape's actual extent (in particular, `D +
  sac_height_mm` instead of `D + aneurysm_diameter_mm/2`, and checking
  whether an asymmetric sac can ever push outside the nominal `[0, L]`
  box -- it can't for Option A since `a_left`/`a_right <= vessel_length`
  is already enforced the same way `R` is today, but this bound should be
  re-verified explicitly for whatever final parametrization is chosen).

## 9. Summary / what I'd recommend approving for Phase 4b

1. **Shape family: Option A** (asymmetric half-ellipse) -- height,
   asymmetry, and a relabeled `asymmetry`-driven "tilt" effect, no new
   dependency, exact preset preservation, `geometry_sdf.py` stays
   closed-form-with-fast-iteration.
2. **No `gmsh`/`pygmsh` dependency.**
3. **Defer**: independent neck-vs-sac width (Option B), literal rigid
   tilt, curved parent vessel -- each flagged above as its own separately
   scoped follow-up if wanted later.
4. New `GeometryConfig` fields default to exactly reproduce today's
   circle for both existing presets; a floating-point-exact regression
   test for both presets' mesh boundary + SDF values is the first thing
   Phase 4b should add, before any other change.
5. `PARAM_ORDER`/sampler ranges/`encoder.param_dim` updates are low-risk
   and mechanical; `coordinate_decoder.py`'s `geometry_mm`-handling is the
   one place needing genuine new logic, not just a config bump.

Waiting for your decision on (1)-(3) before writing any implementation
code.
