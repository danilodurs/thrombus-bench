"""Batch-run the mechanistic solver over sampled parameters to build the surrogate dataset.

Responsibility
---------------
For each parameter sample from `sampler.py`:

1. Build the mesh (`mechanistic/mesh.py`) for the sampled geometry.
2. Run the full coupled mechanistic simulation (`mechanistic/coupled_solver.py`)
   for the sampled physiological parameters and inlet velocity.
3. Save the resulting time series (velocity, pressure, viscosity, species
   concentrations, surface coverage fields) to `data/processed/` in a
   format consumable by `dataset.py`.
4. Route each sample to train/val/test/ood per `sampler.split_train_val_test_ood`.

Intended to run as an embarrassingly-parallel batch job (each sample is an
independent simulation); a `--n-workers` flag should map samples across a
`multiprocessing.Pool` since the mechanistic solver here is CPU-only.

Depends on `mechanistic/coupled_solver.py` (not yet implemented) for the
full transient simulation -- until then, this can only run the flow-only
path via `mechanistic/run_simulation.run_flow_only`.

Not yet implemented -- this is a scaffolding stub.
"""

from __future__ import annotations

import argparse


def generate_dataset(config: dict, output_dir: str, n_workers: int = 1) -> None:
    """Run `sampler.latin_hypercube_sample` + `sampler.split_train_val_test_ood`,
    then batch-run the mechanistic solver over every sample, writing results
    under `output_dir/{train,val,test,ood}/`."""

    raise NotImplementedError("generate_dataset.generate_dataset: not yet implemented")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.parse_args()
    raise NotImplementedError(
        "generate_dataset CLI: pending mechanistic/coupled_solver.py implementation"
    )


if __name__ == "__main__":
    main()
