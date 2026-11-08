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
LADDER = (
    (1, "uniform", "baseline"),
    (2, "draw_only", "baseline"),
    (3, "original_cl", "baseline"),
    (4, "market", "benchmark"),
    (5, "market_calibrated", "benchmark"),
    (6, "cl_only", "model"),
    (7, "lgbm_only", "model"),
    (8, "mkt+cl", "blend"),
    (9, "mkt+lgbm", "blend"),
    (10, "mkt+cl+lgbm", "blend"),
)
LADDER_STEPS = (
    ("draw_only vs uniform", "uniform", "draw_only"),
    ("original_cl vs draw_only", "draw_only", "original_cl"),
    ("market vs original_cl", "original_cl", "market"),
    ("market_calibrated vs market", "market", "market_calibrated"),
    ("cl_only vs original_cl", "original_cl", "cl_only"),
    ("lgbm_only vs cl_only", "cl_only", "lgbm_only"),
    ("mkt+cl+lgbm vs market_calibrated", "market_calibrated", "mkt+cl+lgbm"),
)
PROVENANCE_KEYS = ("data_hash", "registry_hash", "git_commit")


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


def assert_baseline_provenance(config_meta: dict, baseline_meta: dict, fold_id: str) -> None:
    mismatched = [
        key for key in PROVENANCE_KEYS
        if config_meta.get(key) != baseline_meta.get(key)
    ]
    if mismatched:
        raise ValueError(
            f"{fold_id}: baseline and config artifacts have mismatched provenance keys: {mismatched}"
        )


def ladder_frame(
    config_prediction: pd.DataFrame, baseline_prediction: pd.DataFrame, fold_id: str
) -> pd.DataFrame:
    config_valid = config_prediction.loc[config_prediction["role"].eq("valid")].copy()
    baseline_valid = baseline_prediction.loc[baseline_prediction["role"].eq("valid")].copy()
    config_valid = config_valid.drop(
        columns=["p_uniform", "p_draw_only", "p_original_cl"], errors="ignore"
    )
    keys = ["race_id", "horse_id", "role"]
    if config_valid.duplicated(keys).any() or baseline_valid.duplicated(keys).any():
        raise ValueError(f"{fold_id}: duplicate baseline or config prediction keys")
    merged = config_valid.merge(
        baseline_valid[keys + ["is_winner", "p_uniform", "p_draw_only", "p_original_cl"]],
        on=keys,
        how="inner",
        suffixes=("_config", "_baseline"),
        validate="one_to_one",
    )
    config_races = set(config_valid["race_id"])
    merged_races = set(merged["race_id"])
    dropped = len(config_races - merged_races)
    if dropped:
        print(f"{fold_id}: dropped {dropped} market-complete valid races missing baseline predictions")
    if config_races != merged_races:
        raise ValueError(f"{fold_id}: baseline predictions do not cover every market-complete valid race")
    if not merged["is_winner_config"].equals(merged["is_winner_baseline"]):
        raise ValueError(f"{fold_id}: baseline and config winner labels differ")
    merged = merged.drop(columns="is_winner_baseline").rename(
        columns={"is_winner_config": "is_winner"}
    )
    required = [f"p_{variant}" for _, variant, _ in LADDER]
    missing = [column for column in required if column not in merged]
    if missing:
        raise ValueError(f"{fold_id}: unified ladder is missing probabilities: {missing}")
    return merged


def ladder_rows(frame: pd.DataFrame, fold_id: str, era: str, config: str) -> list[dict]:
    losses = {
        variant: race_losses(frame, f"p_{variant}")
        for _, variant, _ in LADDER
    }
    counts = {len(loss) for loss in losses.values()}
    if len(counts) != 1:
        raise ValueError(f"{fold_id}: ladder rungs do not share an identical race set")
    rows = []
    previous_loss = None
    for rung_order, variant, role in LADDER:
        loss = losses[variant].mean()
        rows.append(
            {
                "config": config,
                "fold_id": fold_id,
                "era": era,
                "rung_order": rung_order,
                "variant": variant,
                "role": role,
                "is_primary": (
                    config == PRIMARY_CONFIG
                    and fold_id == "pooled"
                    and variant == PRIMARY_VARIANT
                ),
                "n_races": len(losses[variant]),
                "loss": loss,
                "gain_vs_uniform": losses["uniform"].mean() - loss,
                "gain_vs_market": losses["market"].mean() - loss,
                "gain_vs_calibrated": losses["market_calibrated"].mean() - loss,
                "delta_vs_previous_rung": (
                    None if previous_loss is None else previous_loss - loss
                ),
            }
        )
        previous_loss = loss
    return rows


def ladder_step_rows(
    frame: pd.DataFrame, fold_id: str, config: str
) -> list[dict]:
    days = pd.to_datetime(frame.groupby("race_id")["race_date"].first(), errors="raise").dt.normalize()
    rows = []
    for step, variant_a, variant_b in LADDER_STEPS:
        loss_a = race_losses(frame, f"p_{variant_a}")
        loss_b = race_losses(frame, f"p_{variant_b}")
        low, high = paired_bootstrap_gain(loss_a, loss_b, days.reindex(loss_a.index))
        rows.append(
            {
                "config": config,
                "fold_id": fold_id,
                "step": step,
                "loss_a": loss_a.mean(),
                "loss_b": loss_b.mean(),
                "gain": loss_a.mean() - loss_b.mean(),
                "ci_low": low,
                "ci_high": high,
                "n_races": len(loss_a),
            }
        )
    return rows


def write_ladder(
    root: Path,
    config: str,
    fold_frames: list[tuple[str, str, pd.DataFrame]],
) -> None:
    rows, steps = [], []
    for fold_id, era, frame in fold_frames:
        rows.extend(ladder_rows(frame, fold_id, era, config))
        steps.extend(ladder_step_rows(frame, fold_id, config))
    pooled_frame = pd.concat([frame for _, _, frame in fold_frames], ignore_index=True)
    rows.extend(ladder_rows(pooled_frame, "pooled", "pooled", config))
    steps.extend(ladder_step_rows(pooled_frame, "pooled", config))
    ladder = pd.DataFrame(rows)
    ladder_steps = pd.DataFrame(steps)
    positive = (
        ladder_steps.loc[~ladder_steps["fold_id"].eq("pooled")]
        .assign(positive=lambda frame: frame["gain"].gt(0))
        .groupby("step", sort=False)["positive"]
        .sum()
        .astype(int)
    )
    ladder_steps["n_folds_positive"] = ladder_steps["step"].map(positive).fillna(0).astype(int)
    output = root / config
    ladder.to_csv(output / "ladder.csv", index=False)
    ladder_steps.to_csv(output / "ladder_steps.csv", index=False)
    lines = [
        "# Unified baseline ladder",
        "",
        "All rows use valid-period races only. h1's s1 period mixes Kaggle and HKJC data.",
        "",
        ladder.to_markdown(index=False),
        "",
        "## Paired steps",
        "",
        ladder_steps.to_markdown(index=False),
        "",
        "Each positive gain means the second named rung has lower log loss than the first; "
        "intervals are paired race-day bootstrap intervals.",
    ]
    (output / "ladder.md").write_text("\n".join(lines) + "\n")


def aggregate_config(root: Path, config: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_root = root / config
    frames, selected, rows, metadata, ladder_frames = [], [], [], [], []
    for directory in sorted(path for path in config_root.iterdir() if path.is_dir()):
        prediction = pd.read_parquet(directory / "predictions.parquet")
        meta = json.loads((directory / "meta.json").read_text())
        baseline_directory = root / "_baselines" / directory.name
        baseline_prediction = pd.read_parquet(baseline_directory / "baseline_predictions.parquet")
        baseline_meta = json.loads((baseline_directory / "meta.json").read_text())
        assert_baseline_provenance(meta, baseline_meta, directory.name)
        rows.extend(metric_rows(prediction, directory.name, config))
        frames.append(prediction.assign(fold_id=directory.name))
        ladder_frames.append(
            (directory.name, str(meta["fold"]["era"]), ladder_frame(
                prediction, baseline_prediction, directory.name
            ))
        )
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
    write_ladder(root, config, ladder_frames)
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
