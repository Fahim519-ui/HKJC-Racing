"""Run all chronological ablations and write their combined metrics table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from model_logic.aggregate_folds import aggregate_folds
from model_logic.run_folds import run_folds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run every ablation config over chronological folds, then aggregate results."
    )
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/clean data/runners_merged.parquet"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/folds"))
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--inner-tune-months", type=int, default=4)
    parser.add_argument("--min-gain", type=float, default=0.001)
    parser.add_argument("--fold-ids", nargs="*")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    reports = run_folds(
        folds_path=args.folds,
        data_path=args.data,
        seed_count=args.seeds,
        output=args.out,
        inner_tune_months=args.inner_tune_months,
        minimum_gain=args.min_gain,
        fold_ids=args.fold_ids,
        all_configs=True,
        resume=args.resume,
    )
    metrics = aggregate_folds(root=args.out, all_configs=True)
    if reports:
        print(pd.DataFrame(reports).to_string(index=False))
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
