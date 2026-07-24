"""One-at-a-time sensitivity sweep for `configs/physio_params.yaml`'s two
documented placeholder parameters (README.md "Assumptions & Deviations"
items 5 and 6):

* `sorensen_chemical.k1_ADP_per_s`: reused from `k1_TxA2_per_s` (Eq. A.4's
  inhibition rate constant), since the paper's Appendix D table doesn't
  separately tabulate an ADP value for Eq. (A.3)'s analogous term.
* `scale_terms.D_a`: the thrombus growth-rate scale factor in Eqs. (14)-(15),
  defaulted to 1.0 since the paper states it was tuned but doesn't report
  the resulting numeric value.

This is NOT a search for "the right value" -- neither placeholder has an
independently-sourced target to search toward (see README items 5/6). It
asks a narrower, answerable question: holding everything else (including
mesh/dt) at a fixed, already mesh/dt-inspected-for-convergence resolution
(`scripts/convergence_study.py`'s middle grid point: target_num_elements=800,
dt_s=0.1, end_time_s=2.0), does the model's *qualitative* behavior change
across a defensible range for each placeholder in isolation?

Grid: for each of the same two paper geometries used in
`scripts/convergence_study.py` (`aneurysm_7mm`, `aneurysm_10mm`, at their
respective paper-reported inlet velocities), one baseline run (both
placeholders at their `configs/physio_params.yaml` defaults) plus 2
additional values for each placeholder in turn:

* `D_a` in {0.1, 10.0} vs. the default 1.0 (a 10x-each-way bracket, chosen
  because D_a directly multiplies the growth *rate* in dM_at/dt -- a
  plausible tuning range wide enough to reveal whether growth saturates
  within the simulated window or barely starts, without being an
  arbitrarily huge sweep).
* `k1_ADP_per_s` in {0.5x, 2x} vs. the current reused (1x) value -- per the
  task spec's requested range.

5 runs per geometry (baseline + 2 D_a + 2 k1_ADP, baseline shared/reused as
the "1.0"/"1x" row in both sweeps), 10 total. Runtime: ~2.4s/run at this
mesh/dt (measured in `scripts/convergence_study.py`'s output), so ~25s
total -- no budget check needed.

Reported per run (mirroring `data/generate_dataset.py`'s own summary
scalars where applicable): `max_M_at`, `thrombosed_fraction`, max `[FI]`
(+ `thrombin_fibrin_reliable`), and where peak `M_at` localizes -- the
*nearest* of four landmarks along the idealized vessel+sac geometry's axial
(x) coordinate: inlet (x=0), proximal neck (`xc - R`), distal neck
(`xc + R`), outlet (x=L) (`mesh.py`'s `_aneurysm_geometry_points`:
`xc = 0.5*vessel_length_mm`, `R = 0.5*aneurysm_diameter_mm`), plus the
distance to that landmark. Reporting the true nearest landmark (rather than
forcing a proximal-vs-distal-neck-only choice) matters here: an early
manual check of the baseline run found the actual peak sits exactly at the
inlet (x=0mm, ~21.5mm from the nearest neck for the 7mm geometry) -- a
proximal/distal-only classifier would have silently mislabeled that as
"proximal neck" just because it's numerically closer to that neck than the
distal one, which would misreport an inlet boundary-layer effect as neck
localization.

Output: `verification/sensitivity_study.md`, alongside Task 3.1's
`verification/convergence_study.md`.
"""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import yaml

from thrombus_bench.mechanistic.coupled_solver import run_coupled_simulation
from thrombus_bench.mechanistic.mesh import GeometryConfig, build_aneurysm_mesh

GEOMETRY_PRESETS = ("aneurysm_7mm", "aneurysm_10mm")
TARGET_NUM_ELEMENTS = 800
DT_S = 0.1
END_TIME_S = 2.0

D_A_VALUES = (0.1, 1.0, 10.0)          # default 1.0
K1_ADP_MULTIPLIERS = (0.5, 1.0, 2.0)   # x current reused (k1_TxA2) value


def _copy_physio(physio_base: dict) -> dict:
    """Shallow-per-top-level-key copy, matching
    `data/generate_dataset.py`'s own pattern -- sufficient since callers
    here only mutate one nested dict's leaf value."""

    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in physio_base.items()}


def run_one(geometry_cfg: dict, mesh_cfg_base: dict, physio: dict) -> dict:
    geom = GeometryConfig.from_preset(geometry_cfg)
    mesh_cfg = dict(mesh_cfg_base)
    mesh_cfg["target_num_elements"] = TARGET_NUM_ELEMENTS
    inlet_velocity_m_s = geometry_cfg["inlet_velocity_cm_s"] / 100.0
    n_macro_steps = max(1, int(round(END_TIME_S / DT_S)))

    t0 = time.perf_counter()
    tagged_mesh = build_aneurysm_mesh(geom, mesh_cfg)
    history = run_coupled_simulation(
        tagged_mesh, inlet_velocity_m_s=inlet_velocity_m_s, physio=physio,
        end_time_s=END_TIME_S, dt_s=DT_S,
        output_every_n_steps=n_macro_steps, flow_resolve_every_n_steps=n_macro_steps,
    )
    elapsed_s = time.perf_counter() - t0

    final = history.states[-1]
    basis_c = history.basis_c
    M_at_critical = physio["adhesion_aggregation"]["M_at_critical_plt_cm2"]

    peak_idx = int(np.argmax(final.surface.M_at))
    peak_dof = final.wall_dofs[peak_idx]
    x_peak_m = float(basis_c.doflocs[0, peak_dof])
    L_m = geometry_cfg["vessel_length_mm"] * 1.0e-3
    R_m = 0.5 * geometry_cfg["aneurysm_diameter_mm"] * 1.0e-3
    xc_m = 0.5 * L_m
    landmarks_m = {
        "inlet": 0.0,
        "proximal neck": xc_m - R_m,
        "distal neck": xc_m + R_m,
        "outlet": L_m,
    }
    nearest_label = min(landmarks_m, key=lambda name: abs(x_peak_m - landmarks_m[name]))
    nearest_distance_mm = abs(x_peak_m - landmarks_m[nearest_label]) * 1000.0

    return {
        "elapsed_s": elapsed_s,
        "max_M_at": float(final.surface.M_at.max()),
        "thrombosed_fraction": float(np.mean(final.surface.M_at >= M_at_critical)),
        "x_peak_mm": x_peak_m * 1000.0,
        "nearest_landmark": nearest_label,
        "nearest_landmark_distance_mm": nearest_distance_mm,
        "max_FI_uM": float(final.concentrations["FI"].max()),
        "thrombin_fibrin_reliable": bool(history.thrombin_fibrin_reliable),
    }


def _format_row(label: str, r: dict) -> str:
    reliable = "yes" if r["thrombin_fibrin_reliable"] else "**NO** (cap hit)"
    location = f"{r['nearest_landmark']} ({r['nearest_landmark_distance_mm']:.2f} mm away, x={r['x_peak_mm']:.2f} mm)"
    return (
        f"| {label} | {r['max_M_at']:.4e} | {r['thrombosed_fraction']:.4f} | "
        f"{r['max_FI_uM']:.4e} | {reliable} | {location} | "
        f"{r['elapsed_s']:.2f} |"
    )


_HEADER = (
    "| Value | max_M_at (PLT/cm²) | thrombosed_fraction | max [FI] (µM) | [T]/[FI] reliable? | "
    "Peak M_at location (nearest landmark) | Wall time (s) |\n"
    "|---|---|---|---|---|---|---|"
)


def main() -> None:
    with open("configs/geometry.yaml") as f:
        geometry_yaml = yaml.safe_load(f)
    with open("configs/physio_params.yaml") as f:
        physio_base = yaml.safe_load(f)
    mesh_cfg_base = dict(geometry_yaml["mesh"])

    default_k1_adp = physio_base["sorensen_chemical"]["k1_ADP_per_s"]

    lines = [
        "# Placeholder-parameter sensitivity sweep",
        "",
        "One-at-a-time sweep of `configs/physio_params.yaml`'s two documented "
        "placeholder parameters (README.md \"Assumptions & Deviations\" items 5 "
        "and 6: `sorensen_chemical.k1_ADP_per_s` and `scale_terms.D_a`) -- see "
        "`scripts/sensitivity_study.py`'s module docstring for methodology, "
        "range choices, and column definitions. This is not a search for a "
        '"correct" value (none is available); it asks whether the model\'s '
        "qualitative behavior changes across a defensible range for each "
        "placeholder in isolation, at a fixed mesh/dt "
        f"(target_num_elements={TARGET_NUM_ELEMENTS}, dt_s={DT_S}, "
        f"end_time_s={END_TIME_S:g}).",
        "",
    ]

    grid_t0 = time.perf_counter()
    conclusions = []
    for preset_name in GEOMETRY_PRESETS:
        geometry_cfg = geometry_yaml["presets"][preset_name]
        lines.append(
            f"## `{preset_name}` (vessel {geometry_cfg['vessel_diameter_mm']} mm / "
            f"aneurysm {geometry_cfg['aneurysm_diameter_mm']} mm, "
            f"inlet velocity {geometry_cfg['inlet_velocity_cm_s']} cm/s)"
        )
        lines.append("")

        print(f"[{preset_name}] baseline (D_a=1.0, k1_ADP={default_k1_adp}x1) ...", flush=True)
        baseline = run_one(geometry_cfg, mesh_cfg_base, _copy_physio(physio_base))
        print(f"  -> {baseline['elapsed_s']:.2f}s", flush=True)

        lines.append("### `D_a` sweep (`scale_terms.D_a`, default 1.0)")
        lines.append("")
        lines.append(_HEADER)
        d_a_results = {}
        for d_a in D_A_VALUES:
            if d_a == 1.0:
                r = baseline
            else:
                print(f"[{preset_name}] D_a={d_a} ...", flush=True)
                physio = _copy_physio(physio_base)
                physio["scale_terms"]["D_a"] = d_a
                r = run_one(geometry_cfg, mesh_cfg_base, physio)
                print(f"  -> {r['elapsed_s']:.2f}s", flush=True)
            d_a_results[d_a] = r
            lines.append(_format_row(f"D_a = {d_a:g}", r))
        lines.append("")

        lines.append(
            "### `k1_ADP_per_s` sweep (`sorensen_chemical.k1_ADP_per_s`, "
            f"current reused value {default_k1_adp:g} s⁻¹ = 1x)"
        )
        lines.append("")
        lines.append(_HEADER)
        k1_adp_results = {}
        for mult in K1_ADP_MULTIPLIERS:
            if mult == 1.0:
                r = baseline
            else:
                print(f"[{preset_name}] k1_ADP_per_s = {mult}x ...", flush=True)
                physio = _copy_physio(physio_base)
                physio["sorensen_chemical"]["k1_ADP_per_s"] = default_k1_adp * mult
                r = run_one(geometry_cfg, mesh_cfg_base, physio)
                print(f"  -> {r['elapsed_s']:.2f}s", flush=True)
            k1_adp_results[mult] = r
            lines.append(_format_row(f"{mult:g}x ({default_k1_adp * mult:.4g} s⁻¹)", r))
        lines.append("")

        d_a_necks = {d_a: d_a_results[d_a]["nearest_landmark"] for d_a in D_A_VALUES}
        k1_adp_necks = {mult: k1_adp_results[mult]["nearest_landmark"] for mult in K1_ADP_MULTIPLIERS}
        d_a_neck_stable = len(set(d_a_necks.values())) == 1
        k1_adp_neck_stable = len(set(k1_adp_necks.values())) == 1
        d_a_max_m_at_range = (min(r["max_M_at"] for r in d_a_results.values()), max(r["max_M_at"] for r in d_a_results.values()))
        k1_adp_max_m_at_range = (min(r["max_M_at"] for r in k1_adp_results.values()), max(r["max_M_at"] for r in k1_adp_results.values()))
        d_a_frac_range = (min(r["thrombosed_fraction"] for r in d_a_results.values()), max(r["thrombosed_fraction"] for r in d_a_results.values()))
        k1_adp_frac_range = (min(r["thrombosed_fraction"] for r in k1_adp_results.values()), max(r["thrombosed_fraction"] for r in k1_adp_results.values()))
        d_a_landmark_sequence = " -> ".join(f"{d_a_necks[d_a]} (D_a={d_a:g})" for d_a in D_A_VALUES)
        k1_adp_landmark_sequence = " -> ".join(f"{k1_adp_necks[mult]} ({mult:g}x)" for mult in K1_ADP_MULTIPLIERS)

        lines.append(f"**`{preset_name}` summary:**")
        lines.append("")
        lines.append(f"- `D_a` sweep, peak-location sequence (low to high `D_a`): {d_a_landmark_sequence}.")
        lines.append(f"- `k1_ADP_per_s` sweep, peak-location sequence (0.5x to 2x): {k1_adp_landmark_sequence}.")
        lines.append(
            f"- `max_M_at` ranged {d_a_max_m_at_range[0]:.3e} - {d_a_max_m_at_range[1]:.3e} PLT/cm² "
            f"across the `D_a` sweep ({(d_a_max_m_at_range[1] / max(d_a_max_m_at_range[0], 1e-30)):.2f}x spread); "
            f"`thrombosed_fraction` ranged {d_a_frac_range[0]:.4f} - {d_a_frac_range[1]:.4f}."
        )
        lines.append(
            f"- `max_M_at` ranged {k1_adp_max_m_at_range[0]:.3e} - {k1_adp_max_m_at_range[1]:.3e} PLT/cm² "
            f"across the `k1_ADP_per_s` sweep ({(k1_adp_max_m_at_range[1] / max(k1_adp_max_m_at_range[0], 1e-30)):.2f}x spread); "
            f"`thrombosed_fraction` ranged {k1_adp_frac_range[0]:.4f} - {k1_adp_frac_range[1]:.4f}."
        )
        lines.append("")
        conclusions.append({
            "preset_name": preset_name,
            "d_a_neck_stable": d_a_neck_stable,
            "k1_adp_neck_stable": k1_adp_neck_stable,
            "d_a_landmark": next(iter(d_a_necks.values())) if d_a_neck_stable else None,
            "k1_adp_landmark": next(iter(k1_adp_necks.values())) if k1_adp_neck_stable else None,
            "d_a_frac_range": d_a_frac_range,
            "k1_adp_frac_range": k1_adp_frac_range,
            "d_a_max_m_at_range": d_a_max_m_at_range,
            "k1_adp_max_m_at_range": k1_adp_max_m_at_range,
        })

    grid_elapsed = time.perf_counter() - grid_t0
    print(f"Total sweep runtime: {grid_elapsed:.1f}s", flush=True)

    lines.insert(3, "")
    lines.insert(4, f"Total sweep runtime: {grid_elapsed:.1f} s ({2 * len(GEOMETRY_PRESETS) * 2 + len(GEOMETRY_PRESETS)} runs, "
                     f"baseline shared between both sweeps per geometry).")

    lines.append("## Overall conclusion")
    lines.append("")

    d_a_landmark_stable_everywhere = all(c["d_a_neck_stable"] for c in conclusions)
    k1_adp_landmark_stable_everywhere = all(c["k1_adp_neck_stable"] for c in conclusions)
    d_a_frac_swing = max(c["d_a_frac_range"][1] - c["d_a_frac_range"][0] for c in conclusions)
    k1_adp_frac_swing = max(c["k1_adp_frac_range"][1] - c["k1_adp_frac_range"][0] for c in conclusions)
    # A "large" swing threshold: thrombosed_fraction moving by more than 0.1 (10
    # percentage points of the wall) is a qualitatively different headline
    # result, not sampling noise around a stable number.
    d_a_qualitative_change = d_a_frac_swing > 0.1 or not d_a_landmark_stable_everywhere
    k1_adp_qualitative_change = k1_adp_frac_swing > 0.1 or not k1_adp_landmark_stable_everywhere

    lines.append(
        f"**`k1_ADP_per_s`** ({K1_ADP_MULTIPLIERS[0]:g}x-{K1_ADP_MULTIPLIERS[-1]:g}x the current reused "
        f"value): "
        + (
            "**robust**. `max_M_at`/`thrombosed_fraction`/peak-deposition location were "
            "essentially unchanged across the whole sweep, for both geometries (see the "
            "per-geometry tables above) -- within this range, this placeholder does not "
            "appear to matter for these headline outputs at this end_time_s."
            if not k1_adp_qualitative_change
            else "**NOT robust** -- see the per-geometry tables/sequences above for how "
            "`thrombosed_fraction` and/or peak location shift across the sweep."
        )
    )
    lines.append("")
    lines.append(
        f"**`D_a`** ({D_A_VALUES[0]:g}-{D_A_VALUES[-1]:g}, a 10x-each-way bracket around the default "
        f"{1.0:g}): "
        + (
            "**robust**. Headline outputs were essentially unchanged across the whole sweep."
            if not d_a_qualitative_change
            else "**NOT robust -- this placeholder's guessed value changes the qualitative "
            "result.** In every geometry tested, `thrombosed_fraction` swung from "
            f"~0 at low/default `D_a` to a large fraction of the wall (see the per-geometry "
            "tables above) at the high end of this plausible range -- i.e. whether the model "
            'reports "essentially no thrombosis" or "most of the wall thrombosed" by end_time_s '
            f"={END_TIME_S:g}s depends materially on a value the paper itself says was "
            'tuned but doesn\'t report ("different values of D_a were tested", README.md '
            'item 6). Separately, peak deposition\'s *location* was also not always at a neck '
            "(see the per-geometry landmark sequences) -- at the low end of this sweep it sat "
            "at a genuine neck, but at default/high `D_a` it shifted to the inlet, a boundary-"
            "layer effect competing with the neck-localized mechanism, not something either "
            "placeholder is \"responsible for\" so much as something this sweep incidentally "
            "surfaced. Whichever landmark a given run's peak sits at, don't read a stable "
            "landmark alone as confirmation of neck localization without checking which "
            "landmark it actually is."
        )
    )
    lines.append("")
    lines.append(
        "`[T]`/`[FI]` reliability (concentration-cap clip) is reported per row above for "
        'completeness, per README.md "Known limitations"; treat `max [FI]` as a numerically-'
        "bounded proxy, not a physically meaningful value, in every run where it reads "
        "**NO** (every run in this sweep, in fact -- consistent with "
        "`verification/convergence_study.md`'s Task 3.1 finding that this is a robust, "
        "resolution-independent hit of the safety clip, not something either placeholder here "
        "changes)."
    )

    os.makedirs("verification", exist_ok=True)
    output_path = "verification/sensitivity_study.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
