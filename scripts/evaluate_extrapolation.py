"""CLI: evaluate a surrogate trained on a genuine-extrapolation split.

Prerequisite pipeline (mirrors `thrombus-generate-dataset` ->
`thrombus-train` -> `thrombus-benchmark`, but for the opt-in extrapolation
split -- see `data/generate_dataset.generate_extrapolation_dataset` and
`benchmark/extrapolation_eval.py`'s module docstrings for why this needs
its own separately-trained checkpoint, not the default demo/pilot one):

```bash
thrombus-generate-dataset --extrapolation-param heparin_conc_uM \\
    --output-dir data/processed_extrap_heparin
thrombus-train --dataset-dir data/processed_extrap_heparin \\
    --checkpoint checkpoints/model_extrap_heparin.pt
python scripts/evaluate_extrapolation.py
```

Prints the same `label`/`test`/`extrapolation`/`degradation_ratio` dict
`benchmark/extrapolation_eval.evaluate_extrapolation_degradation` returns,
labeled explicitly (e.g. "heparin_conc_uM extrapolation (trained on
0.1-0.38, tested on 0.38-0.5)") so it is never confused with
`edge_holdout_eval.py`'s same-sampled-box edge-of-domain degradation.
"""

from __future__ import annotations

import argparse

import torch

from thrombus_bench.benchmark.extrapolation_eval import evaluate_extrapolation_degradation
from thrombus_bench.data.dataset import ThrombusSurrogateDataset
from thrombus_bench.data.generate_dataset import DEFAULT_EXTRAPOLATION_TRAIN_FRACTION
from thrombus_bench.data.sampler import DEFAULT_RANGES
from thrombus_bench.neural.model import ThrombusSurrogate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, default="data/processed_extrap_heparin")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_extrap_heparin.pt")
    parser.add_argument("--extrapolation-param", type=str, default="heparin_conc_uM", choices=["heparin_conc_uM"])
    parser.add_argument(
        "--extrapolation-split-fraction", type=float, default=DEFAULT_EXTRAPOLATION_TRAIN_FRACTION,
        help="Must match the value used when the dataset was generated (thrombus-generate-dataset's "
        "--extrapolation-split-fraction, same default) -- only used to label the report; the actual "
        "split is whatever's already on disk in --dataset-dir.",
    )
    args = parser.parse_args()

    lo, hi = DEFAULT_RANGES[args.extrapolation_param]
    split_point = lo + args.extrapolation_split_fraction * (hi - lo)
    train_range, extrapolate_range = (lo, split_point), (split_point, hi)

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model = ThrombusSurrogate(checkpoint["cfg"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_ds = ThrombusSurrogateDataset(args.dataset_dir, "test")
    extrapolation_ds = ThrombusSurrogateDataset(args.dataset_dir, "extrapolation")

    result = evaluate_extrapolation_degradation(
        model, test_ds, extrapolation_ds,
        extrapolate_param=args.extrapolation_param, train_range=train_range, extrapolate_range=extrapolate_range,
    )

    print(f"\n{result['label']}")
    print(f"  Test (in-range) field RMSE:            {result['test']['overall']:.4f}")
    print(f"  Extrapolation (withheld-range) field RMSE: {result['extrapolation']['overall']:.4f}")
    print(f"  Degradation ratio (extrapolation / test): {result['degradation_ratio']:.2f}x")


if __name__ == "__main__":
    main()
