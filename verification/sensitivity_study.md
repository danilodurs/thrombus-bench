# Placeholder-parameter sensitivity sweep

One-at-a-time sweep of `configs/physio_params.yaml`'s two documented placeholder parameters (README.md "Assumptions & Deviations" items 5 and 6: `sorensen_chemical.k1_ADP_per_s` and `scale_terms.D_a`) -- see `scripts/sensitivity_study.py`'s module docstring for methodology, range choices, and column definitions. This is not a search for a "correct" value (none is available); it asks whether the model's qualitative behavior changes across a defensible range for each placeholder in isolation, at a fixed mesh/dt (target_num_elements=800, dt_s=0.1, end_time_s=2).

Total sweep runtime: 24.1 s (10 runs, baseline shared between both sweeps per geometry).

## `aneurysm_7mm` (vessel 3.2 mm / aneurysm 7.0 mm, inlet velocity 47.0 cm/s)

### `D_a` sweep (`scale_terms.D_a`, default 1.0)

| Value | max_M_at (PLT/cm²) | thrombosed_fraction | max [FI] (µM) | [T]/[FI] reliable? | Peak M_at location (nearest landmark) | Wall time (s) |
|---|---|---|---|---|---|---|
| D_a = 0.1 | 1.0570e+07 | 0.0000 | 1.4000e+01 | **NO** (cap hit) | proximal neck (0.01 mm away, x=21.51 mm) | 2.42 |
| D_a = 1 | 1.3819e+07 | 0.0000 | 1.4000e+01 | **NO** (cap hit) | inlet (0.00 mm away, x=0.00 mm) | 2.29 |
| D_a = 10 | 1.0404e+08 | 0.7771 | 1.4000e+01 | **NO** (cap hit) | inlet (0.00 mm away, x=0.00 mm) | 2.46 |

### `k1_ADP_per_s` sweep (`sorensen_chemical.k1_ADP_per_s`, current reused value 0.0161 s⁻¹ = 1x)

| Value | max_M_at (PLT/cm²) | thrombosed_fraction | max [FI] (µM) | [T]/[FI] reliable? | Peak M_at location (nearest landmark) | Wall time (s) |
|---|---|---|---|---|---|---|
| 0.5x (0.00805 s⁻¹) | 1.3819e+07 | 0.0000 | 1.4000e+01 | **NO** (cap hit) | inlet (0.00 mm away, x=0.00 mm) | 2.47 |
| 1x (0.0161 s⁻¹) | 1.3819e+07 | 0.0000 | 1.4000e+01 | **NO** (cap hit) | inlet (0.00 mm away, x=0.00 mm) | 2.29 |
| 2x (0.0322 s⁻¹) | 1.3819e+07 | 0.0000 | 1.4000e+01 | **NO** (cap hit) | inlet (0.00 mm away, x=0.00 mm) | 2.48 |

**`aneurysm_7mm` summary:**

- `D_a` sweep, peak-location sequence (low to high `D_a`): proximal neck (D_a=0.1) -> inlet (D_a=1) -> inlet (D_a=10).
- `k1_ADP_per_s` sweep, peak-location sequence (0.5x to 2x): inlet (0.5x) -> inlet (1x) -> inlet (2x).
- `max_M_at` ranged 1.057e+07 - 1.040e+08 PLT/cm² across the `D_a` sweep (9.84x spread); `thrombosed_fraction` ranged 0.0000 - 0.7771.
- `max_M_at` ranged 1.382e+07 - 1.382e+07 PLT/cm² across the `k1_ADP_per_s` sweep (1.00x spread); `thrombosed_fraction` ranged 0.0000 - 0.0000.

## `aneurysm_10mm` (vessel 4.0 mm / aneurysm 10.0 mm, inlet velocity 75.0 cm/s)

### `D_a` sweep (`scale_terms.D_a`, default 1.0)

| Value | max_M_at (PLT/cm²) | thrombosed_fraction | max [FI] (µM) | [T]/[FI] reliable? | Peak M_at location (nearest landmark) | Wall time (s) |
|---|---|---|---|---|---|---|
| D_a = 0.1 | 1.1204e+07 | 0.0000 | 1.4000e+01 | **NO** (cap hit) | proximal neck (0.01 mm away, x=20.01 mm) | 2.39 |
| D_a = 1 | 1.3763e+07 | 0.0000 | 1.4000e+01 | **NO** (cap hit) | inlet (0.00 mm away, x=0.00 mm) | 2.38 |
| D_a = 10 | 1.0362e+08 | 0.7030 | 1.4000e+01 | **NO** (cap hit) | inlet (0.00 mm away, x=0.00 mm) | 2.40 |

### `k1_ADP_per_s` sweep (`sorensen_chemical.k1_ADP_per_s`, current reused value 0.0161 s⁻¹ = 1x)

| Value | max_M_at (PLT/cm²) | thrombosed_fraction | max [FI] (µM) | [T]/[FI] reliable? | Peak M_at location (nearest landmark) | Wall time (s) |
|---|---|---|---|---|---|---|
| 0.5x (0.00805 s⁻¹) | 1.3763e+07 | 0.0000 | 1.4000e+01 | **NO** (cap hit) | inlet (0.00 mm away, x=0.00 mm) | 2.39 |
| 1x (0.0161 s⁻¹) | 1.3763e+07 | 0.0000 | 1.4000e+01 | **NO** (cap hit) | inlet (0.00 mm away, x=0.00 mm) | 2.38 |
| 2x (0.0322 s⁻¹) | 1.3763e+07 | 0.0000 | 1.4000e+01 | **NO** (cap hit) | inlet (0.00 mm away, x=0.00 mm) | 2.40 |

**`aneurysm_10mm` summary:**

- `D_a` sweep, peak-location sequence (low to high `D_a`): proximal neck (D_a=0.1) -> inlet (D_a=1) -> inlet (D_a=10).
- `k1_ADP_per_s` sweep, peak-location sequence (0.5x to 2x): inlet (0.5x) -> inlet (1x) -> inlet (2x).
- `max_M_at` ranged 1.120e+07 - 1.036e+08 PLT/cm² across the `D_a` sweep (9.25x spread); `thrombosed_fraction` ranged 0.0000 - 0.7030.
- `max_M_at` ranged 1.376e+07 - 1.376e+07 PLT/cm² across the `k1_ADP_per_s` sweep (1.00x spread); `thrombosed_fraction` ranged 0.0000 - 0.0000.

## Overall conclusion

**`k1_ADP_per_s`** (0.5x-2x the current reused value): **robust**. `max_M_at`/`thrombosed_fraction`/peak-deposition location were essentially unchanged across the whole sweep, for both geometries (see the per-geometry tables above) -- within this range, this placeholder does not appear to matter for these headline outputs at this end_time_s.

**`D_a`** (0.1-10, a 10x-each-way bracket around the default 1): **NOT robust -- this placeholder's guessed value changes the qualitative result.** In every geometry tested, `thrombosed_fraction` swung from ~0 at low/default `D_a` to a large fraction of the wall (see the per-geometry tables above) at the high end of this plausible range -- i.e. whether the model reports "essentially no thrombosis" or "most of the wall thrombosed" by end_time_s =2s depends materially on a value the paper itself says was tuned but doesn't report ("different values of D_a were tested", README.md item 6). Separately, peak deposition's *location* was also not always at a neck (see the per-geometry landmark sequences) -- at the low end of this sweep it sat at a genuine neck, but at default/high `D_a` it shifted to the inlet, a boundary-layer effect competing with the neck-localized mechanism, not something either placeholder is "responsible for" so much as something this sweep incidentally surfaced. Whichever landmark a given run's peak sits at, don't read a stable landmark alone as confirmation of neck localization without checking which landmark it actually is.

`[T]`/`[FI]` reliability (concentration-cap clip) is reported per row above for completeness, per README.md "Known limitations"; treat `max [FI]` as a numerically-bounded proxy, not a physically meaningful value, in every run where it reads **NO** (every run in this sweep, in fact -- consistent with `verification/convergence_study.md`'s Task 3.1 finding that this is a robust, resolution-independent hit of the safety clip, not something either placeholder here changes).