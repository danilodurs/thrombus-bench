"""Physics-informed loss terms: PDE residuals, BC flux residuals, conservation penalties.

Responsibility
---------------
Penalize violations of the governing equations in the neural surrogate's
predicted fields, weighted per `configs/training.yaml` `physics_loss.weights`:

Space each penalty operates in
-------------------------------
`total_physics_loss`'s input (and the model's raw output, `pred`) is
entirely in the **log-compressed space** of `data/dataset.field_to_log`
(`sign(x) * log1p(|x|)`, applied per-channel including velocity -- see that
module's docstring). Which space a penalty must evaluate in depends on
whether the physical law it encodes is linear in the field or not:

* `mass_conservation`: div(u) = du/dx + dv/dy is a *linear* differential
  operator, and `log1p`/`sign` are nonlinear, so differencing the
  log-compressed velocity channels directly does NOT measure physical
  divergence -- `d/dx[sign(u)*log1p(|u|)] != f(du/dx)` for any simple `f`.
  `total_physics_loss` therefore inverts the velocity channels back to
  **physical space** via `data/dataset.log_to_field` before calling
  `mass_conservation_penalty` (steady incompressible flow requires
  div(u)=0 pointwise; `mechanistic/flow_solver.compute_boundary_flux` checks
  the integrated form exactly for the mechanistic solver -- here it is a
  *soft* differentiable training penalty on the surrogate's output).
* `nonnegativity`: non-negativity is a *sign* condition, and
  `field_to_log`/`log_to_field` are sign-preserving and zero-preserving
  (`y >= 0 <=> x >= 0` for `y = field_to_log(x)`), so checking it directly
  on the **log-compressed** predicted channels (no inversion) gives exactly
  the same verdict as checking physical concentrations. All physical
  species/surface fields are non-negative.
* `navier_stokes_residual` / `cdr_residual`: residuals of Eqs. (3)-(4) and
  Eq. (1); like `mass_conservation`, these involve differential operators
  that are not invariant under the log transform, so any caller wiring them
  in must pass **physical-space** velocity/pressure/concentration fields
  (see `total_physics_loss` below for why `navier_stokes_residual` isn't
  currently wired in at all).

Fluid-domain masking (optional `mask` argument)
-------------------------------------------------
`mass_conservation_penalty`, `nonnegativity_penalty`, and `total_physics_loss`
all accept an optional `mask` -- the per-sample fluid/exterior raster
`data/dataset.py`'s `fluid_mask` (True where a grid cell is inside the
actual FEM mesh domain; see `data/generate_dataset._fluid_mask`). The
rasterization grid's bounding box contains genuine exterior cells (the
vessel+aneurysm domain is an L/T-shaped union), and those cells hold
whatever `griddata(method="nearest")` filled in -- not meaningful values.
Without masking, a penalty averaged over the whole grid is diluted by the
(easy, mostly-constant/near-zero) exterior background, understating how
badly it's actually doing on the (hard) fluid interior. When `mask` is
given, `mass_conservation_penalty` additionally erodes it by one pixel
before use (see `_erode_mask_one_pixel`) because it's a spatial-derivative
penalty: a central difference computed one cell inside the mask boundary
still reads its exterior neighbor's meaningless value. `nonnegativity_penalty`
is a pointwise sign check with no derivative, so it uses the mask directly,
unmodified.

Residual computation mode (`configs/training.yaml` `physics_loss.residual_mode`)
-------------------------------------------------------------------------------
* `"finite_difference"`: spatial derivatives via central-difference
  convolution kernels on the regular output grid. **Implemented** -- used
  by `total_physics_loss` below, for the grid-projection baseline
  (`neural/model.ThrombusSurrogate`) only.
* `"autograd"`: derivatives via `torch.autograd.grad` on
  `neural.coordinate_decoder.ContinuousThrombusSurrogate`'s genuinely
  continuous `(x, y) -> field` function -- **implemented** (`docs/
  continuous_surrogate_design.md` Phase 5), for the continuous model only,
  via `mass_conservation_penalty_autograd` (generic, model-agnostic core
  mechanics) + `sample_collocation_points` (PINN-style SDF-rejection
  sampling) + `continuous_mass_conservation_loss` (wires the two together
  for `neural.train.train_continuous`) below. Deliberately a separate code
  path from `total_physics_loss` rather than a mode-branch inside it:
  `total_physics_loss` operates on a fixed `(B, C, H, W)` raster tensor,
  which a ragged, continuous point cloud has no natural way to become
  without re-rasterizing (defeating the point of the continuous model);
  the two residual modes correspond to two structurally different data
  representations (grid vs. point cloud), not just two ways of computing
  the same derivative on the same data.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import log_to_field
from ..mechanistic.geometry_sdf import signed_distance_to_wall
from ..mechanistic.mesh import GeometryConfig


def _central_diff(field: torch.Tensor, axis: int) -> torch.Tensor:
    """d(field)/d(axis), axis=2 -> y (rows), axis=3 -> x (cols); (B,C,H,W).

    Central differences in the interior; one-sided forward/backward
    differences at the two boundary rows/columns of that axis. The grid is
    a bounded vessel+aneurysm domain (inlet/outlet/wall boundaries), not a
    periodic one, so a `torch.roll`-based implementation (wrapping the
    boundary around) is physically wrong -- it treats a value at one edge of
    the grid as adjacent to the opposite edge. (Padding with
    `mode="replicate"` and reusing the central-difference formula everywhere
    would halve the boundary derivative -- e.g. `(f[1]-f[0])/2` instead of
    `f[1]-f[0]` -- since the replicated edge value counts as one of the two
    central-difference neighbors; slicing explicitly avoids that.)
    """

    n = field.size(axis)
    if n == 1:
        return torch.zeros_like(field)

    first = field.narrow(axis, 0, 1)
    second = field.narrow(axis, 1, 1)
    last = field.narrow(axis, n - 1, 1)
    second_last = field.narrow(axis, n - 2, 1)

    left = second - first  # forward difference at the first boundary point
    right = last - second_last  # backward difference at the last boundary point

    if n == 2:
        return torch.cat([left, right], dim=axis)

    interior = (field.narrow(axis, 2, n - 2) - field.narrow(axis, 0, n - 2)) / 2.0
    return torch.cat([left, interior, right], dim=axis)


def _prepare_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-sample fluid mask, shape (B, H, W) (as saved/exposed
    by `data/dataset.py`'s `fluid_mask`), to `reference`'s (B, C, H, W)
    shape by inserting the channel axis, matching its dtype/device."""

    if mask.dim() == reference.dim() - 1:
        mask = mask.unsqueeze(1)
    return mask.to(dtype=reference.dtype, device=reference.device)


def _erode_mask_one_pixel(mask: torch.Tensor) -> torch.Tensor:
    """Erode a 0/1 fluid mask by one pixel (4-connected: up/down/left/right
    neighbors) -- see module docstring's "Fluid-domain masking" section for
    why `mass_conservation_penalty` needs this and `nonnegativity_penalty`
    does not.

    A cell survives erosion only if it and every one of its *existing*
    neighbors are also fluid. Missing neighbors (true array edges, not mask
    edges) are treated as trivially satisfied rather than as exterior: at a
    genuine grid edge, `_central_diff` already falls back to a one-sided
    stencil that never reads anything outside the array, so there is no
    fluid/exterior mixing to guard against there -- only at an internal
    mask boundary, where a central difference reads across it.
    """

    up = torch.ones_like(mask)
    up[..., 1:, :] = mask[..., :-1, :]
    down = torch.ones_like(mask)
    down[..., :-1, :] = mask[..., 1:, :]
    left = torch.ones_like(mask)
    left[..., :, 1:] = mask[..., :, :-1]
    right = torch.ones_like(mask)
    right[..., :, :-1] = mask[..., :, 1:]
    return mask * up * down * left * right


def mass_conservation_penalty(
    velocity_x: torch.Tensor, velocity_y: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Soft penalty on the discrete divergence div(u) = du/dx + dv/dy of the
    predicted velocity field (should vanish for steady incompressible flow,
    Eq. 4).

    If `mask` is given (fluid/exterior raster, see module docstring's
    "Fluid-domain masking" section), the mean is taken only over fluid
    cells, eroded by one pixel first (`_erode_mask_one_pixel`) so that
    cells whose central difference reads an exterior neighbor are excluded
    too, not just genuinely exterior cells themselves.
    """

    div = _central_diff(velocity_x, axis=3) + _central_diff(velocity_y, axis=2)
    if mask is None:
        return div.pow(2).mean()

    eroded = _erode_mask_one_pixel(_prepare_mask(mask, div))
    return (div.pow(2) * eroded).sum() / eroded.sum().clamp_min(1.0)


def mass_conservation_penalty_autograd(
    velocity_fn: Callable[[torch.Tensor], torch.Tensor], query_points: torch.Tensor
) -> torch.Tensor:
    """Autograd counterpart of `mass_conservation_penalty` for a genuinely
    continuous `(x, y) -> velocity` function, rather than a fixed raster
    (classic PINN-style collocation-point residual) -- see module
    docstring's "Residual computation mode" section.

    `velocity_fn`: maps `(N, 2)` physical `(x, y)` query points to `(N, 2)`
    physical `[u, v]` velocity, evaluated **pointwise** -- `velocity_fn(p)[i]`
    must depend only on `p[i]`, never on any other row. This holds for
    `ContinuousThrombusSurrogate`: `grid_sample`'s bilinear interpolation and
    the per-point MLP decoder head are both pointwise operations with no
    cross-point mixing (confirmed directly, not just assumed, by
    `test_mass_conservation_penalty_autograd_matches_analytic_divergence_
    pointwise_not_averaged` in the test suite -- a spatially-*varying*
    divergence field, since a spatially-uniform one like solid-body rotation
    can't distinguish "correct per-point residual" from "residuals
    mixed/averaged across the batch," both being zero either way).

    Given that pointwise property, `torch.autograd.grad(u, points,
    grad_outputs=torch.ones_like(u), ...)` (the standard "vector-Jacobian"
    trick, equivalent to backpropagating from `u.sum()`) correctly recovers
    each point's own local `du/dx` (and `dv/dy`), since
    `d(sum_j u_j)/d(points[i])` collapses to exactly `d(u_i)/d(points[i])`
    -- every cross term `d(u_j)/d(points[i])` for `j != i` is identically
    zero.

    `create_graph=True` on both `autograd.grad` calls is required -- without
    it, the returned residual would be detached from the *outer* graph
    (model parameters -> latent -> velocity -> [this gradient]), so a
    training loss that includes this penalty would silently fail to
    backpropagate into the model's parameters through this term at all (no
    error; the contribution would just be zero). Checked directly, not just
    asserted here, by `test_gradient_flows_to_model_parameters_through_
    autograd_mass_conservation_loss` in the test suite.

    Uses a fresh internal leaf tensor (`query_points.detach().clone()
    .requires_grad_(True)`) rather than mutating the caller's `query_points`
    in place, so this call has no side effect on the tensor the caller
    passed in.
    """

    points = query_points.detach().clone().requires_grad_(True)
    velocity = velocity_fn(points)  # (N, 2): [u, v]
    u, v = velocity[:, 0], velocity[:, 1]

    du = torch.autograd.grad(u, points, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    dv = torch.autograd.grad(v, points, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]

    div = du[:, 0] + dv[:, 1]
    return div.pow(2).mean()


def sample_collocation_points(
    geometry_mm: torch.Tensor,
    vessel_length_mm: float,
    n_points_per_sample: int,
    rng: np.random.Generator | None = None,
    oversample_factor: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reject-sample `n_points_per_sample` PINN-style collocation points per
    batch sample, strictly inside the fluid domain (`signed_distance_to_wall
    (...) > 0`, Phase 1's analytic SDF, reused exactly) -- points where the
    PDE residual is enforced without needing ground-truth field values
    there.

    `geometry_mm`: `(batch, 2)` raw `[aneurysm_diameter_mm,
    vessel_diameter_mm]`, same convention as `ContinuousThrombusSurrogate.
    forward`'s argument of the same name. Returns `(query_points_m,
    batch_index)` in this project's usual flat + `batch_index` ragged
    convention (`neural/coordinate_decoder.py`, `data/dataset.
    pointcloud_collate_fn`).

    Uniformly oversamples each sample's own analytic bounding box `[0, L] x
    [0, D + R]` (`geometry_sdf.py`'s module docstring) by `oversample_factor`
    and keeps only positive-SDF points, rather than a rejection loop with an
    unbounded worst case: the vessel+aneurysm domain (a long rectangle plus
    a half-disk bulge) occupies a large majority of its own bounding box, so
    under-filling is not expected at `oversample_factor=4` in practice; if
    it happens anyway, this returns however many points were actually
    accepted for that sample rather than raising or padding.
    """

    rng = rng if rng is not None else np.random.default_rng()
    geometry_np = geometry_mm.detach().cpu().numpy()

    all_points, all_batch_index = [], []
    for b in range(geometry_np.shape[0]):
        geom = GeometryConfig(
            vessel_diameter_mm=float(geometry_np[b, 1]),
            aneurysm_diameter_mm=float(geometry_np[b, 0]),
            vessel_length_mm=vessel_length_mm,
        )
        L_m = vessel_length_mm * 1.0e-3
        D_m = geometry_np[b, 1] * 1.0e-3
        R_m = geometry_np[b, 0] * 0.5e-3

        n_candidates = n_points_per_sample * oversample_factor
        xs = rng.uniform(0.0, L_m, n_candidates)
        ys = rng.uniform(0.0, D_m + R_m, n_candidates)
        inside = signed_distance_to_wall(xs, ys, geom) > 0.0

        accepted = np.column_stack([xs[inside], ys[inside]])[:n_points_per_sample]
        all_points.append(accepted)
        all_batch_index.append(np.full(len(accepted), b, dtype=np.int64))

    query_points_m = torch.from_numpy(np.concatenate(all_points, axis=0).astype(np.float32))
    batch_index = torch.from_numpy(np.concatenate(all_batch_index, axis=0))
    return query_points_m, batch_index


def continuous_mass_conservation_loss(
    model: torch.nn.Module,
    params_with_time: torch.Tensor,
    geometry_mm: torch.Tensor,
    n_points_per_sample: int,
    vessel_length_mm: float,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Wires `sample_collocation_points` + `mass_conservation_penalty_autograd`
    to a `ContinuousThrombusSurrogate` (`model`) -- the continuous path's
    counterpart of `total_physics_loss`'s (grid-only, finite-difference)
    `mass_conservation` term. Used by `neural.train.train_continuous` when
    `physics_loss.residual_mode: autograd`.

    `model`'s raw output channels 0/1 are log-compressed velocity (same
    convention as everywhere else in this project -- see module docstring's
    "Space each penalty operates in"), inverted back to physical space via
    `log_to_field` before computing the divergence, exactly like
    `total_physics_loss` does for the grid path.

    Note: this calls `model(...)` once per training step just for these
    collocation points, separately from whatever forward pass computes the
    data loss at the batch's own query points -- i.e. `SurrogateBackbone`
    runs twice per step (once per query-point set). This is the simplest
    correct implementation; sharing one backbone forward pass's latent grid
    across both would need `ContinuousThrombusSurrogate.forward`'s API to
    accept a precomputed latent, which is a valid future optimization but
    out of scope here (correctness first).
    """

    query_points_m, batch_index = sample_collocation_points(geometry_mm, vessel_length_mm, n_points_per_sample, rng)

    def velocity_fn(points: torch.Tensor) -> torch.Tensor:
        pred = model(params_with_time, points, batch_index, geometry_mm)
        return log_to_field(pred[:, :2])

    return mass_conservation_penalty_autograd(velocity_fn, query_points_m)


def nonnegativity_penalty(*fields: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """sum of relu(-field)^2 over all physical (non-negative) predicted fields.

    If `mask` is given (fluid/exterior raster, see module docstring's
    "Fluid-domain masking" section), each field's mean is taken only over
    fluid cells -- unlike `mass_conservation_penalty`, no erosion: this is a
    pointwise sign check with no spatial derivative, so a fluid cell's own
    value is trustworthy regardless of its neighbors.
    """

    if mask is None:
        return sum(F.relu(-f).pow(2).mean() for f in fields)

    def _masked_term(f: torch.Tensor) -> torch.Tensor:
        m = _prepare_mask(mask, f)
        return (F.relu(-f).pow(2) * m).sum() / m.sum().clamp_min(1.0)

    return sum(_masked_term(f) for f in fields)


def navier_stokes_residual(velocity_x: torch.Tensor, velocity_y: torch.Tensor, pressure: torch.Tensor, viscosity: float) -> torch.Tensor:
    """Simplified steady-Stokes momentum residual (Eqs. 3-4, inertia
    dropped, matching `mechanistic/flow_solver.py`'s own simplification):
    `-grad(p) + mu*laplacian(u) ~= 0`, penalized as a soft residual norm."""

    lap_u = _central_diff(_central_diff(velocity_x, 3), 3) + _central_diff(_central_diff(velocity_x, 2), 2)
    lap_v = _central_diff(_central_diff(velocity_y, 3), 3) + _central_diff(_central_diff(velocity_y, 2), 2)
    dp_dx = _central_diff(pressure, 3)
    dp_dy = _central_diff(pressure, 2)
    res_x = -dp_dx + viscosity * lap_u
    res_y = -dp_dy + viscosity * lap_v
    return res_x.pow(2).mean() + res_y.pow(2).mean()


def cdr_residual(*args, **kwargs) -> torch.Tensor:
    """Still not implemented -- see module docstring's scope note, and
    `docs/continuous_surrogate_design.md` Phase 5's assessment (reported to
    the user, deferred pending explicit sign-off, not implemented here) of
    whether an autograd-based residual for RP/AP/APR/APS specifically
    (excluding T/AT/PT/FG/FI, the concentration-cap-affected species) is
    now tractable given `ContinuousThrombusSurrogate`'s autograd machinery.

    Summary of that assessment: `activation.chemical_source_terms`'s actual
    reaction terms for RP/AP/APR/APS (as wired in `coupled_solver.py`)
    depend only on {RP, AP, APR, APS, bulk shear rate} -- genuinely
    decoupled from T/PT/AT/FG/FI, more favorable than assumed. But a full
    residual still needs: (1) `d/dt` via autograd through the *entire*
    Stage 1 backbone (time is a `SurrogateBackbone` encoder input, not a
    Stage 2 query coordinate -- structurally heavier than the spatial
    derivatives `mass_conservation_penalty_autograd` uses), (2) the bulk
    shear rate `sqrt(2 D:D)` (Eq. 2, `flow_solver.shear_rate`) reimplemented
    in differentiable form from the velocity-gradient tensor -- a physics
    formula port with its own correctness risk, and (3) a Laplacian
    (second-derivative autograd) per species, for 4 species. Assessed as
    real scope creep for one phase on top of the mass-conservation work
    already in it, not implemented without explicit go-ahead.
    """

    raise NotImplementedError(
        "physics_losses.cdr_residual: not implemented in this project (scope note in module docstring)"
    )


def surface_flux_bc_residual(*args, **kwargs) -> torch.Tensor:
    raise NotImplementedError(
        "physics_losses.surface_flux_bc_residual: not implemented in this project (scope note in module docstring)"
    )


def total_physics_loss(
    predicted_fields: torch.Tensor,
    weights: dict,
    mode: str,
    viscosity: float = 0.0035,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted physics-informed loss terms computable without ground-truth
    reference data, evaluated on `predicted_fields` (B, C, H, W) ordered per
    `data/dataset.FIELD_NAMES`: [velocity_x, velocity_y, conc_RP, conc_AP,
    conc_APR, conc_APS, conc_T, conc_AT, conc_PT, conc_FG, conc_FI].

    Returns a dict of named (unweighted) loss terms; `train.py` applies
    `weights` and sums. Only `mass_conservation` and `nonnegativity` are
    implemented here (see module docstring for `navier_stokes_residual`,
    which is implemented but not wired into this default set: besides a
    representative viscosity scalar, it needs a predicted pressure field,
    and pressure is not one of the surrogate's output channels -- there is
    no pressure entry in `data/dataset.FIELD_NAMES`. Wiring it in would mean
    adding a pressure output channel through the dataset/model pipeline, not
    just this function).

    `predicted_fields` is in `data/dataset.field_to_log`'s log-compressed
    space (same as the model's raw output and training target). See the
    module docstring's "Space each penalty operates in" section:
    `mass_conservation` needs the velocity channels inverted back to
    physical space first (`data/dataset.log_to_field`); `nonnegativity` is
    evaluated directly on the log-compressed species channels, which is
    equivalent to physical-space non-negativity since the transform is
    sign-preserving.

    `mask` (optional, `data/dataset.py`'s `fluid_mask`, shape (B, H, W)) is
    forwarded to both penalties -- see the module docstring's "Fluid-domain
    masking" section.
    """

    if mode != "finite_difference":
        raise NotImplementedError(
            f"physics_losses.total_physics_loss: residual_mode={mode!r} not implemented (only 'finite_difference' is)"
        )

    velocity_x_log, velocity_y_log = predicted_fields[:, 0:1], predicted_fields[:, 1:2]
    species_fields_log = predicted_fields[:, 2:]

    velocity_x_physical = log_to_field(velocity_x_log)
    velocity_y_physical = log_to_field(velocity_y_log)

    return {
        "mass_conservation": mass_conservation_penalty(velocity_x_physical, velocity_y_physical, mask=mask),
        "nonnegativity": nonnegativity_penalty(species_fields_log, mask=mask),
    }
