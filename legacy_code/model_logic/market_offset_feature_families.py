"""Attribute market-offset LightGBM residual gain to feature families."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from model_logic.feature_builder import build_features
    from model_logic.feature_registry import get_features
except ModuleNotFoundError:
    from feature_builder import build_features
    from feature_registry import get_features
from market_diagnostics import FEATURE_GROUPS
from market_offset_lightgbm import FOLDS, train_fold

ALL_FEATURES = tuple(get_features("conditional_logit"))


def walk_forward(
    data: pd.DataFrame, features: list[str], max_rounds: int, name: str
) -> pd.DataFrame:
    rows = []
    for valid_start, valid_end in FOLDS:
        train = data.loc[data["race_date"] < valid_start].copy()
        valid = data.loc[
            data["race_date"].ge(valid_start) & data["race_date"].lt(valid_end)
        ].copy()
        result = train_fold(train, valid, max_rounds, features)
        rows.append(
            {
                "feature_set": name,
                "validation_start": valid_start,
                "validation_end": valid_end,
                "n_features": len(features),
                "market_gain": result["market_log_loss"] - result["residual_log_loss"],
                **result,
            }
        )
    return pd.DataFrame(rows)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, result in metrics.groupby("feature_set", sort=False):
        weights = result["valid_races"]
        market_loss = np.average(result["market_log_loss"], weights=weights)
        residual_loss = np.average(result["residual_log_loss"], weights=weights)
        rows.append(
            {
                "feature_set": name,
                "n_features": int(result["n_features"].iloc[0]),
                "positive_gain_folds": int((result["market_gain"] > 0).sum()),
                "weighted_market_log_loss": market_loss,
                "weighted_residual_log_loss": residual_loss,
                "weighted_market_gain": market_loss - residual_loss,
                "mean_fold_gain": result["market_gain"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    full_gain = summary.loc[
        summary["feature_set"].eq("full"), "weighted_market_gain"
    ].iloc[0]
    summary["gain_vs_full"] = summary["weighted_market_gain"] - full_gain
    return summary.sort_values("weighted_market_gain", ascending=False).reset_index(drop=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data/clean data/runners_merged_with_features.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/market_offset_feature_families",
    )
    parser.add_argument("--max-rounds", type=int, default=4000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = build_features(pd.read_parquet(args.data))
    data = data.loc[data["source"].eq("kaggle") & data["mkt_ok"]].copy()
    data = data.loc[data.groupby("race_id")["is_winner"].transform("max").gt(0)].copy()

    sets = {"full": ALL_FEATURES}
    sets.update({f"only_{name}": features for name, features in FEATURE_GROUPS.items()})
    for name, features in FEATURE_GROUPS.items():
        sets[f"without_{name}"] = [feature for feature in ALL_FEATURES if feature not in features]
    track_distance = FEATURE_GROUPS["track_distance_interactions"]
    sets.update({f"only_{feature}": [feature] for feature in track_distance})
    for feature in track_distance:
        sets[f"track_distance_without_{feature}"] = [
            candidate for candidate in track_distance if candidate != feature
        ]

    metrics = pd.concat(
        [
            walk_forward(data, features, args.max_rounds, name)
            for name, features in sets.items()
        ],
        ignore_index=True,
    )
    summary = summarise(metrics)
    metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    report = [
        "# Market-offset LightGBM feature-family attribution",
        "",
        "Each model uses `init_score = log(market probability)` and identical "
        "walk-forward folds. `only_*` tests standalone residual signal; `without_*` "
        "tests the full model after removing that family. Positive `gain_vs_full` "
        "means a set outperformed the full model.",
        "",
        "## Summary",
        "",
        "```",
        summary.round(6).to_string(index=False),
        "```",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
