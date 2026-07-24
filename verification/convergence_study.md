# Mechanistic solver: mesh / time-step self-convergence study

Self-convergence only (no analytic/reference solution exists for this idealized geometry+physics combination) -- see `scripts/convergence_study.py`'s module docstring for methodology and column definitions, and README.md "Known limitations" for the project-wide validation caveat. Read each table top-to-bottom-right: values should stabilize (stop changing much) as mesh resolution increases and dt decreases, not match any particular target value.

End time: 2 s. Total grid runtime: 53.1 s (18 runs).

**Peak wall shear rate is a pointwise (max-over-nodes) quantity near the proximal/distal neck's geometric corners, where the true continuum shear field is singular/near-singular** -- it can converge much more slowly, and less monotonically, with mesh refinement than the integrated/domain quantities (velocity L2 norm, total wall M, ∫RP/∫AP) in the same table. A large jump in that one column between rows is expected mesh sensitivity at a sharp corner, not necessarily a sign the rest of the run is unreliable.

## `aneurysm_7mm` (vessel 3.2 mm / aneurysm 7.0 mm, inlet velocity 47.0 cm/s)

| Mesh (target / actual elements) | dt (s) | Wall time (s) | Velocity L2 norm | Pressure drop (Pa) | Peak wall shear (s⁻¹) | Total wall M (PLT/cm) | ∫RP (PLT·m²/mL) | ∫AP (PLT·m²/mL) | Max [T] (µM) | Max [FI] (µM) | [T]/[FI] reliable? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 400 (556) | 0.2 | 1.18 | 6.3948e-03 | 8.4409e+03 | 1.9751e+03 | 4.4957e+07 | 6.2610e+04 | 3.0176e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 400 (556) | 0.1 | 1.83 | 6.3947e-03 | 8.4413e+03 | 1.9749e+03 | 4.5160e+07 | 6.2613e+04 | 3.0174e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 400 (556) | 0.05 | 3.13 | 6.3945e-03 | 8.4422e+03 | 1.9745e+03 | 4.5815e+07 | 6.2617e+04 | 3.0168e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 800 (871) | 0.2 | 1.61 | 6.4025e-03 | 8.0006e+03 | 3.2245e+03 | 4.1945e+07 | 6.2640e+04 | 3.0063e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 800 (871) | 0.1 | 2.48 | 6.3995e-03 | 8.0315e+03 | 3.2224e+03 | 4.2517e+07 | 6.2646e+04 | 3.0060e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 800 (871) | 0.05 | 4.22 | 6.3916e-03 | 8.0647e+03 | 3.2199e+03 | 4.2449e+07 | 6.2644e+04 | 3.0058e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 1600 (1466) | 0.2 | 2.44 | 6.9383e-03 | 6.5319e+03 | 3.6422e+03 | 3.9787e+07 | 6.2551e+04 | 2.9844e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 1600 (1466) | 0.1 | 3.53 | 6.7993e-03 | 6.8150e+03 | 3.6422e+03 | 3.9655e+07 | 6.2548e+04 | 2.9846e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 1600 (1466) | 0.05 | 5.87 | 6.5730e-03 | 7.2733e+03 | 3.6432e+03 | 3.9576e+07 | 6.2546e+04 | 2.9847e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |

**Note:** every run above hit the `[T]`/`[FI]` concentration-cap safety clip (`thrombin_fibrin_reliable=False`), and both columns are identical to 4 significant figures across the whole grid -- that's the clip ceiling, not a converged physical value. See README.md "Known limitations"; these two columns cannot be used to assess convergence here.

## `aneurysm_10mm` (vessel 4.0 mm / aneurysm 10.0 mm, inlet velocity 75.0 cm/s)

| Mesh (target / actual elements) | dt (s) | Wall time (s) | Velocity L2 norm | Pressure drop (Pa) | Peak wall shear (s⁻¹) | Total wall M (PLT/cm) | ∫RP (PLT·m²/mL) | ∫AP (PLT·m²/mL) | Max [T] (µM) | Max [FI] (µM) | [T]/[FI] reliable? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 400 (556) | 0.2 | 1.02 | 1.1327e-02 | 7.8749e+03 | 2.6649e+03 | 4.8350e+07 | 8.3558e+04 | 4.0756e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 400 (556) | 0.1 | 1.64 | 1.1352e-02 | 6.9344e+03 | 5.9739e+03 | 4.8267e+07 | 8.3558e+04 | 4.0755e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 400 (556) | 0.05 | 3.03 | 1.1326e-02 | 7.8532e+03 | 2.6695e+03 | 4.9095e+07 | 8.3564e+04 | 4.0746e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 800 (859) | 0.2 | 1.58 | 1.2446e-02 | 6.3203e+03 | 3.4449e+03 | 4.3864e+07 | 8.3696e+04 | 4.0663e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 800 (859) | 0.1 | 2.45 | 1.2365e-02 | 6.4428e+03 | 3.4352e+03 | 4.4608e+07 | 8.3702e+04 | 4.0657e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 800 (859) | 0.05 | 4.21 | 1.2136e-02 | 6.6065e+03 | 3.4352e+03 | 4.4550e+07 | 8.3702e+04 | 4.0657e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 1600 (1459) | 0.2 | 2.59 | 1.3607e-02 | 5.0585e+03 | 5.4005e+03 | 4.3267e+07 | 8.3627e+04 | 4.0438e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 1600 (1459) | 0.1 | 3.83 | 1.3246e-02 | 5.4245e+03 | 5.3996e+03 | 4.3148e+07 | 8.3625e+04 | 4.0439e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |
| 1600 (1459) | 0.05 | 6.38 | 1.2491e-02 | 6.1056e+03 | 5.3935e+03 | 4.3242e+07 | 8.3626e+04 | 4.0438e+03 | 1.0000e+03 | 1.4000e+01 | **NO** (cap hit) |

**Note:** every run above hit the `[T]`/`[FI]` concentration-cap safety clip (`thrombin_fibrin_reliable=False`), and both columns are identical to 4 significant figures across the whole grid -- that's the clip ceiling, not a converged physical value. See README.md "Known limitations"; these two columns cannot be used to assess convergence here.
