"""CLI: run both models on the test set, produce the comparison report + plots.

Responsibility
---------------
End-to-end benchmark entrypoint:

1. Load the trained neural surrogate (`neural/model.py`) and run it on
   `data/dataset.py` `split="test"`.
2. Run the mechanistic solver (`mechanistic/coupled_solver.py`) on the same
   test-set parameter samples (or load cached mechanistic outputs from
   `data/generate_dataset.py`, since those *are* the mechanistic solver's
   output).
3. Compute `benchmark/metrics.py` accuracy/runtime metrics,
   `benchmark/ood_eval.py` OOD degradation, and `benchmark/calibration.py`
   UQ calibration.
4. Render all plots via `viz/plots.py` and assemble a single Markdown +
   PNG bundle at `results/report.md` (accuracy table, runtime table, OOD
   degradation plot, calibration plot), per project scope.

CLI: `thrombus-benchmark benchmark=default`.

Not yet implemented -- this is a scaffolding stub. Depends on every other
module in `mechanistic/`, `neural/`, and `benchmark/`.
"""

from __future__ import annotations

import argparse


def run_benchmark(cfg: dict, output_dir: str = "results") -> None:
    raise NotImplementedError("run_benchmark.run_benchmark: not yet implemented")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.parse_args()
    raise NotImplementedError(
        "run_benchmark CLI: pending mechanistic/coupled_solver.py and neural/model.py implementation"
    )


if __name__ == "__main__":
    main()
