"""Aggregate saved chronological fold predictions without selecting a winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from model_logic.ablation_configs import CONFIGS
from model_logic.validation.evaluation import paired_bootstrap_gain, race_losses


def metric_rows(frame: pd.DataFrame, fold_id: str, config: str) -> list[dict]:
    valid = frame.loc[frame["role"].eq("valid")].copy()
    valid["race_day"] = pd.to_datetime(valid["race_date"], errors="raise").dt.normalize()
    market = race_losses(valid, "p_market")
    calibrated = race_losses(valid, "p_market_calibrated")
    rows = []
    for column in sorted(item for item in valid if item.startswith("p_")):
        loss = race_losses(valid, column)
        lo, hi = paired_bootstrap_gain(market, loss, valid.groupby("race_id")["race_day"].first())
        rows.append({
            "config": config, "fold_id": fold_id, "variant": column.removeprefix("p_"),
            "n_races": len(loss),
            "loss": loss.mean(), "market_loss": market.mean(), "market_calibrated_loss": calibrated.mean(),
            "gain_vs_market": market.mean() - loss.mean(),
            "gain_vs_calibrated": calibrated.mean() - loss.mean(), "ci_low": lo, "ci_high": hi,
        })
    return rows


def aggregate_config(root: Path, config: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_root = root / config
    frames, selected, rows, metadata = [], [], [], []
    for directory in sorted(path for path in config_root.iterdir() if path.is_dir()):
        prediction = pd.read_parquet(directory / "predictions.parquet")
        meta = json.loads((directory / "meta.json").read_text())
        rows.extend(metric_rows(prediction, directory.name, config))
        frames.append(prediction.assign(fold_id=directory.name))
        selected.extend(json.loads((directory / "cl_selected_features.json").read_text()))
        metadata.append({
            "config": config,
            "fold_id": directory.name,
            "comparison": "primary" if meta["primary"] else "exploratory",
            "unavailable_exclusions": ",".join(meta["unavailable_exclusions"]) or None,
        })

    metrics = pd.DataFrame(rows)
    pooled = pd.DataFrame(metric_rows(pd.concat(frames, ignore_index=True), "pooled", config))
    metrics = pd.concat([metrics, pooled], ignore_index=True)
    labels = pd.DataFrame(metadata)
    metrics = metrics.merge(labels, on=["config", "fold_id"], how="left")
    metrics.loc[metrics["fold_id"].eq("pooled"), "comparison"] = (
        "primary" if CONFIGS[config].primary else "exploratory"
    )
    metrics.loc[metrics["fold_id"].eq("pooled"), "unavailable_exclusions"] = (
        ",".join(
            sorted({
                item
                for values in labels["unavailable_exclusions"].dropna()
                for item in values.split(",")
            })
        )
        or None
    )
    selection = pd.Series(selected).value_counts().rename_axis("feature").reset_index(name="fold_count")
    selection.insert(0, "config", config)
    return metrics, selection


def aggregate_folds(
    root: Path = Path("artifacts/folds"), config: str = "primary", all_configs: bool = False
) -> pd.DataFrame:
    configs = sorted(CONFIGS) if all_configs else [config]
    metrics_by_config, selections_by_config = [], []
    for selected_config in configs:
        metrics, selection = aggregate_config(root, selected_config)
        metrics.to_csv(root / selected_config / "fold_metrics.csv", index=False)
        selection.to_csv(root / selected_config / "feature_selection_frequency.csv", index=False)
        metrics_by_config.append(metrics)
        selections_by_config.append(selection)

    metrics = pd.concat(metrics_by_config, ignore_index=True)
    selections = pd.concat(selections_by_config, ignore_index=True)
    if all_configs:
        metrics.to_csv(root / "ablation_metrics.csv", index=False)
        selections.to_csv(root / "ablation_feature_selection_frequency.csv", index=False)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/folds"))
    parser.add_argument("--config", default="primary")
    parser.add_argument("--all-configs", action="store_true")
    args = parser.parse_args()
    metrics = aggregate_folds(args.root, args.config, args.all_configs)
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
