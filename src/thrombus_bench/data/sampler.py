"""Latin hypercube sampling over the physiological + geometric parameter space.

Responsibility
---------------
Define the parameter space used to generate the training/val/test/OOD
dataset for the neural surrogate, and draw stratified (Latin hypercube)
samples from it. Parameters, with physiologically-motivated ranges drawn
from the paper's sensitivity studies (Sec. 3.3):

* Geometry: aneurysm diameter (7-10 mm, interpolated/extrapolated around the
  paper's two studied geometries), vessel diameter.
* Inlet velocity: 30-100 cm/s (paper explores 45/75/100 cm/s, Fig. 9;
  physiological range 30-70 cm/s per Sec. 3.3.3 citation).
* Resting platelet concentration: 1e8-5e8 PLT/ml (Sec. 3.3.1 citation range).
* Heparin concentration: 0.1-0.5 uM (Fig. 8 range).
* Prothrombin / antithrombin / fibrinogen concentrations: physiological
  ranges around Table 1 nominal values.

The OOD (out-of-distribution) split (`benchmark/ood_eval.py`) is carved out
of the *extremes* of these ranges (see `configs/training.yaml`
`data.ood_quantile`), e.g. very high platelet count + very low heparin +
very high inlet velocity simultaneously -- combinations expected to stress
the surrogate's extrapolation behavior.

Not yet implemented -- this is a scaffolding stub.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParameterSpace:
    """Named (low, high) ranges for each sampled parameter."""

    ranges: dict  # name -> (low, high)


def latin_hypercube_sample(space: ParameterSpace, n_samples: int, seed: int = 0):
    """Draw `n_samples` Latin hypercube samples from `space`.

    Returns a structured array / list of dicts, one per sample, with keys
    matching `space.ranges`.
    """

    raise NotImplementedError("sampler.latin_hypercube_sample: not yet implemented")


def split_train_val_test_ood(samples, ood_quantile: float, n_train: int, n_val: int, n_test: int, n_ood: int, seed: int = 0):
    """Partition `samples` into train/val/test/OOD sets.

    OOD samples are drawn from the outer `(1 - ood_quantile)` tail of the
    per-parameter distributions (see module docstring); train/val/test are
    drawn from the remaining "in-distribution" pool.
    """

    raise NotImplementedError("sampler.split_train_val_test_ood: not yet implemented")
