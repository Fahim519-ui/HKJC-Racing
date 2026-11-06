"""Aggregate saved chronological fold predictions without selecting a winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from model_logic.validation.evaluation import paired_bootstrap_gain, race_losses


def metric_rows(frame: pd.DataFrame, fold_id: str) -> list[dict]:
    valid = frame.loc[frame["role"].eq("valid")].copy()
    valid["race_day"] = pd.to_datetime(valid["race_date"], errors="raise").dt.normalize()
    market = race_losses(valid, "p_market")
    calibrated = race_losses(valid, "p_market_calibrated")
    rows = []
    for column in sorted(item for item in valid if item.startswith("p_")):
        loss = race_losses(valid, column)
        lo, hi = paired_bootstrap_gain(market, loss, valid.groupby("race_id")["race_day"].first())
        rows.append({
            "fold_id": fold_id, "variant": column.removeprefix("p_"), "n_races": len(loss),
            "loss": loss.mean(), "market_loss": market.mean(), "market_calibrated_loss": calibrated.mean(),
            "gain_vs_market": market.mean() - loss.mean(),
            "gain_vs_calibrated": calibrated.mean() - loss.mean(), "ci_low": lo, "ci_high": hi,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/folds"))
    parser.add_argument("--config", default="primary")
    args = parser.parse_args()
    root = args.root / args.config
    frames, selected, rows = [], [], []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        prediction = pd.read_parquet(directory / "predictions.parquet")
        rows.extend(metric_rows(prediction, directory.name))
        frames.append(prediction.assign(fold_id=directory.name))
        selected.extend(json.loads((directory / "cl_selected_features.json").read_text()))
    metrics = pd.DataFrame(rows)
    pooled = metric_rows(pd.concat(frames, ignore_index=True), "pooled")
    metrics = pd.concat([metrics, pd.DataFrame(pooled)], ignore_index=True)
    metrics.to_csv(root / "fold_metrics.csv", index=False)
    pd.Series(selected).value_counts().rename_axis("feature").reset_index(name="fold_count").to_csv(
        root / "feature_selection_frequency.csv", index=False
    )
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
