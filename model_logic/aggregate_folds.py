"""Aggregate saved chronological fold predictions without selecting a winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from model_logic.ablation_configs import CONFIGS
from model_logic.validation.evaluation import paired_bootstrap_gain, race_losses


PRIMARY_CONFIG = "primary"
PRIMARY_VARIANT = "mkt+cl+lgbm"
MIN_INNER_TUNE_RACES = 150


def metric_rows(frame: pd.DataFrame, fold_id: str, config: str) -> list[dict]:
    valid = frame.loc[frame["role"].eq("valid")].copy()
    valid["race_day"] = pd.to_datetime(valid["race_date"], errors="raise").dt.normalize()
    market = race_losses(valid, "p_market")
    calibrated = race_losses(valid, "p_market_calibrated")
    rows = []
    for column in sorted(item for item in valid if item.startswith("p_")):
        loss = race_losses(valid, column)
        day_ids = valid.groupby("race_id")["race_day"].first()
        market_lo, market_hi = paired_bootstrap_gain(market, loss, day_ids)
        calibrated_lo, calibrated_hi = paired_bootstrap_gain(calibrated, loss, day_ids)
        rows.append({
            "config": config, "fold_id": fold_id, "variant": column.removeprefix("p_"),
            "n_races": len(loss),
            "loss": loss.mean(), "market_loss": market.mean(), "market_calibrated_loss": calibrated.mean(),
            "gain_vs_market": market.mean() - loss.mean(),
            "gain_vs_calibrated": calibrated.mean() - loss.mean(),
            "ci_low": market_lo, "ci_high": market_hi,
            "ci_vs_market_low": market_lo, "ci_vs_market_high": market_hi,
            "ci_vs_calibrated_low": calibrated_lo, "ci_vs_calibrated_high": calibrated_hi,
        })
    return rows


def primary_evidence_status(
    metrics: pd.DataFrame, inner_tune_races: dict[str, int]
) -> dict[str, object]:
    """Assess the pre-registered primary comparison without selecting a configuration."""
    primary = metrics.loc[
        metrics["config"].eq(PRIMARY_CONFIG)
        & metrics["variant"].eq(PRIMARY_VARIANT)
        & ~metrics["fold_id"].eq("pooled")
    ].copy()
    primary["inner_tune_races"] = primary["fold_id"].map(inner_tune_races)
    if primary["inner_tune_races"].isna().any():
        missing = primary.loc[primary["inner_tune_races"].isna(), "fold_id"].tolist()
        raise ValueError(f"Missing inner-tune race counts for folds: {missing}")

    primary["watch_only"] = primary["inner_tune_races"].lt(MIN_INNER_TUNE_RACES)
    primary["positive_gain"] = primary["gain_vs_calibrated"].gt(0)
    primary["interval_excludes_zero"] = primary["ci_vs_calibrated_low"].gt(0)
    eligible = primary.loc[~primary["watch_only"]]
    required_positive_folds = len(eligible)
    positive_folds = int(eligible["positive_gain"].sum())
    intervals_pass = bool(eligible["interval_excludes_zero"].all()) if required_positive_folds else False
    eligible_for_freeze = (
        required_positive_folds > 0
        and positive_folds == required_positive_folds
        and intervals_pass
    )
    reasons = []
    if primary["watch_only"].any():
        reasons.append(f"watch-only folds below {MIN_INNER_TUNE_RACES} inner-tune races")
    if positive_folds < required_positive_folds:
        reasons.append("not every eligible fold has positive calibrated-market gain")
    if not intervals_pass:
        reasons.append("not every eligible fold excludes zero versus calibrated market")
    return {
        "config": PRIMARY_CONFIG,
        "variant": PRIMARY_VARIANT,
        "positive_fold_rule": "all_eligible",
        "minimum_inner_tune_races": MIN_INNER_TUNE_RACES,
        "eligible_folds": eligible["fold_id"].tolist(),
        "watch_only_folds": primary.loc[primary["watch_only"], "fold_id"].tolist(),
        "positive_folds": positive_folds,
        "required_positive_folds": required_positive_folds,
        "all_eligible_intervals_exclude_zero": intervals_pass,
        "eligible_for_freeze": eligible_for_freeze,
        "reasons": reasons,
    }


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
        inner_tune_races = {
            path.parent.name: json.loads(path.read_text())["race_counts"]["inner_tune"]
            for path in (root / PRIMARY_CONFIG).glob("*/meta.json")
        }
        evidence = primary_evidence_status(metrics, inner_tune_races)
        (root / "evidence_summary.json").write_text(json.dumps(evidence, indent=2) + "\n")
        csv_evidence = {
            key: ",".join(value) if isinstance(value, list) else value
            for key, value in evidence.items()
        }
        pd.DataFrame([csv_evidence]).to_csv(root / "evidence_summary.csv", index=False)
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
