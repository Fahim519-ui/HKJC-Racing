"""Tune a leakage-safe conditional logit against the fixed Kaggle validation window.

The script uses the latest Week-3 merged feature table when it is available.
It recomputes its eligible historical rates to enforce project data rules:
history comes only from earlier race days in the same source. It never uses
odds as a Stage-1 feature and never scores the Apr-Oct 2026 held-out test
window. It always saves the best selected model. Its report explicitly states
whether that model beats the market baseline on the Kaggle tuning window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

BASE_FEATURES = [
    "jt_jockey_win365_rel",
    "hf_avg3_fin_pct_rel",
    "rc_draw_pct",
    "rc_draw_x_hv",
    "rc_weight_rel",
    "hf_days_since",
]
CANDIDATE_FEATURES = [
    "hf_last_fin_pct_rel",
    "jt_trainer_win365_rel",
    "hf_win5_s",
    "hf_first_start",
    "rc_draw_x_sprint",
    "rc_weight_x_dist",
    "hf_class_change",
    "w3_horse_past_win_rate",
    "w3_horse_past_top3_rate",
]


def assign_split(data: pd.DataFrame) -> pd.Series:
    split = pd.Series("warmup", index=data.index)
    kaggle = data["source"].eq("kaggle")
    hkjc = data["source"].eq("hkjc")
    date = data["race_date"]
    split[kaggle & (date >= "2017-11-01") & (date < "2019-08-01")] = "k_train"
    split[kaggle & (date >= "2019-08-01")] = "k_tune"
    split[hkjc & (date >= "2025-11-01") & (date < "2026-04-01")] = "h_s2"
    split[hkjc & (date >= "2026-04-01")] = "h_test"
    return split


def smoothed_rate(
    data: pd.DataFrame, key: str, target: str, prior: float, *, weight: float = 30.0
) -> np.ndarray:
    daily = (
        data.groupby(["source", key, "race_date"], as_index=False)
        .agg(hits=(target, "sum"), n=(target, "size"))
        .sort_values(["source", key, "race_date"])
    )
    rolling = (
        daily.set_index("race_date")
        .groupby(["source", key])[["hits", "n"]]
        .rolling("365D", closed="left")
        .sum()
        .fillna(0)
        .reset_index()
    )
    rolling["value"] = (rolling["hits"] + weight * prior) / (rolling["n"] + weight)
    return data.merge(
        rolling[["source", key, "race_date", "value"]],
        on=["source", key, "race_date"],
        how="left",
    )["value"].to_numpy()


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.loc[raw["race_valid"] & raw["started"]].copy()
    data["race_date"] = pd.to_datetime(data["race_date"])
    data = data.sort_values(["source", "race_date", "race_no"]).reset_index(drop=True)
    data["is_winner"] = data["is_winner"].astype(int)
    data["is_top3"] = data["is_top3"].astype(int)
    data["split"] = assign_split(data)

    field_size = data["field_size"].astype(float)
    data["rc_draw_pct"] = (data["draw"].astype(float) - 1) / (field_size - 1)
    is_happy_valley = data["track_code"].eq("HV").astype(float)
    is_sprint = (data["distance_m"].astype(float) <= 1200).astype(float)
    data["rc_draw_x_hv"] = data["rc_draw_pct"] * is_happy_valley
    data["rc_draw_x_sprint"] = data["rc_draw_pct"] * is_sprint
    weight = data["carried_weight_lb"].astype(float)
    data["rc_weight_rel"] = weight - weight.groupby(data["race_id"]).transform("mean")
    data["rc_weight_x_dist"] = (
        data["rc_weight_rel"] * (data["distance_m"].astype(float) - 1400) / 1000
    )

    data["finish_percentile"] = (data["finish_pos"].astype(float) - 1) / (field_size - 1)
    horse = data.groupby(["source", "horse_id"], sort=False)
    data["hf_days_since"] = np.log1p(
        (data["race_date"] - horse["race_date"].shift(1)).dt.days.clip(upper=365)
    )
    data["hf_first_start"] = horse.cumcount().eq(0).astype(float)
    data["hf_last_fin_pct"] = horse["finish_percentile"].shift(1)
    data["hf_avg3_fin_pct"] = horse["finish_percentile"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    prior = 1 / 12
    wins = horse["is_winner"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).sum().fillna(0)
    )
    starts = horse["is_winner"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).count()
    )
    data["hf_win5_s"] = (wins + 5 * prior) / (starts + 5)
    data["w3_horse_past_win_rate"] = horse["is_winner"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    data["w3_horse_past_top3_rate"] = horse["is_top3"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    data["hf_class_change"] = np.log(
        data["prize_hkd"].astype(float) / horse["prize_hkd"].shift(1).astype(float)
    )
    for column in ("hf_last_fin_pct", "hf_avg3_fin_pct"):
        data[f"{column}_rel"] = data[column] - data.groupby("race_id")[column].transform("mean")

    data["jt_jockey_win365"] = smoothed_rate(data, "jockey_key", "is_winner", prior)
    data["jt_trainer_win365"] = smoothed_rate(data, "trainer_key", "is_winner", prior)
    for column in ("jt_jockey_win365", "jt_trainer_win365"):
        data[f"{column}_rel"] = data[column] - data.groupby("race_id")[column].transform("mean")

    return data.drop(columns=["finish_percentile"])


def race_softmax(scores: np.ndarray, race_id: pd.Series) -> pd.Series:
    score = pd.Series(np.asarray(scores, dtype=float), index=race_id.index)
    exponentiated = np.exp(score - score.groupby(race_id).transform("max"))
    return exponentiated / exponentiated.groupby(race_id).transform("sum")


def per_race_log_loss(probability: pd.Series, data: pd.DataFrame) -> float:
    winners = data["is_winner"].eq(1)
    return float(
        -np.log(probability[winners].groupby(data.loc[winners, "race_id"]).mean()).mean()
    )


def fit_model(
    train: pd.DataFrame, features: list[str]
) -> tuple[ConditionalLogit, pd.Series, pd.Series]:
    mean = train[features].astype(float).mean()
    std = train[features].astype(float).std().replace(0, 1)
    train_x = ((train[features].astype(float) - mean) / std).fillna(0.0).to_numpy()
    model = ConditionalLogit(
        train["is_winner"].astype(float).to_numpy(),
        train_x,
        groups=train["race_id"].to_numpy(),
    ).fit(disp=0)
    return model, mean, std


def fit_and_score(
    train: pd.DataFrame, tune: pd.DataFrame, features: list[str]
) -> tuple[ConditionalLogit, pd.Series, float, pd.Series, pd.Series]:
    model, mean, std = fit_model(train, features)
    tune_x = ((tune[features].astype(float) - mean) / std).fillna(0.0).to_numpy()
    probability = race_softmax(tune_x @ model.params, tune["race_id"])
    return model, probability, per_race_log_loss(probability, tune), mean, std


def select_features(
    train: pd.DataFrame, tune: pd.DataFrame, minimum_gain: float
) -> tuple[list[str], pd.DataFrame, float]:
    selected = list(BASE_FEATURES)
    _, _, current_loss, _, _ = fit_and_score(train, tune, selected)
    rows = [
        {
            "step": 0,
            "candidate": "(base)",
            "log_loss": current_loss,
            "improvement": np.nan,
            "accepted": True,
            "selected_features": ", ".join(selected),
        }
    ]
    remaining = list(CANDIDATE_FEATURES)

    while remaining:
        scored = []
        for candidate in remaining:
            _, _, loss, _, _ = fit_and_score(train, tune, selected + [candidate])
            scored.append((loss, candidate))
        loss, candidate = min(scored)
        improvement = current_loss - loss
        accepted = improvement >= minimum_gain
        rows.append(
            {
                "step": len(rows),
                "candidate": candidate,
                "log_loss": loss,
                "improvement": improvement,
                "accepted": accepted,
                "selected_features": ", ".join(selected + [candidate]),
            }
        )
        remaining.remove(candidate)
        if not accepted:
            break
        selected.append(candidate)
        current_loss = loss
    return selected, pd.DataFrame(rows), current_loss


def write_report(
    output_dir: Path,
    market_loss: float,
    selected_loss: float,
    selected: list[str],
    selection: pd.DataFrame,
    correlation: pd.DataFrame,
    accepted: bool,
) -> None:
    status = "BEATS MARKET" if accepted else "DOES NOT BEAT MARKET"
    text = [
        "# Conditional-logit tuning report",
        "",
        f"## Status: SELECTED — {status}",
        "",
        "Validation window: Kaggle 2019/20 only (`k_tune`).",
        f"- Market per-race log loss: **{market_loss:.4f}**",
        f"- Selected conditional-logit per-race log loss: **{selected_loss:.4f}**",
        f"- Market improvement: **{market_loss - selected_loss:.4f}** (positive means the model is better)",
        "",
        "The model used no odds or market-derived values as Stage-1 features. "
        "The Apr-Oct 2026 HKJC test window was predicted but not scored.",
        "",
        "## Week-3 feature integration",
        "",
        "The input table includes the Week-3 engineered columns. The model recomputes "
        "`w3_horse_past_win_rate` and `w3_horse_past_top3_rate` from earlier starts "
        "within each source before considering them for selection.",
        "",
        "It excludes `early_position` and `position_change`: they are parsed from the "
        "current race's running positions and are available only after that race. It also "
        "excludes the pushed `jockey_past_win_rate` and `trainer_past_win_rate`: their "
        "values change between a person's rides on the same day. The model instead uses "
        "the existing leakage-safe 365-day `jt_*` features.",
        "",
        "## Selected features",
        "",
        *[f"- `{feature}`" for feature in selected],
        "",
        "## Selection trace",
        "",
        "```",
        selection.round(4).to_string(index=False),
        "```",
        "",
        "## Correlation check",
        "",
        "See `feature_correlations.csv`. Any absolute Spearman correlation above 0.9 must be "
        "resolved before using a feature set for the final model. The conditional-logit "
        "coefficient estimates fitted on `k_train` are in `tuning_coefficients.csv`.",
    ]
    (output_dir / "tuning_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    selection.to_csv(output_dir / "selection_trace.csv", index=False)
    correlation.to_csv(output_dir / "feature_correlations.csv")


def main() -> int:
    parser = argparse.ArgumentParser()
    local_feature_data = Path("data/clean data/runners_merged_with_features.parquet")
    local_data = Path("data/clean data/runners_merged.parquet")
    sagemaker_data = Path("data/processed/runners_merged.parquet")
    parser.add_argument(
        "--data",
        type=Path,
        default=(
            local_feature_data
            if local_feature_data.exists()
            else local_data
            if local_data.exists()
            else sagemaker_data
        ),
        help="Path to the cleaned merged Parquet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/conditional_logit_tuning"),
        help="Directory for generated reports and accepted-model outputs.",
    )
    parser.add_argument(
        "--minimum-gain",
        type=float,
        default=0.001,
        help="Minimum k_tune log-loss reduction to accept an added feature.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_features(pd.read_parquet(args.data))
    train = data.loc[data["split"].eq("k_train")].copy()
    tune = data.loc[data["split"].eq("k_tune")].copy()
    selected, trace, selected_loss = select_features(train, tune, args.minimum_gain)
    market_loss = per_race_log_loss(tune["market_prob"].astype(float), tune)
    correlation = train[selected].astype(float).corr(method="spearman")
    tuning_model, _, _, _, _ = fit_and_score(train, tune, selected)
    pd.DataFrame(
        {
            "feature": selected,
            "coefficient": tuning_model.params,
            "standard_error": tuning_model.bse,
        }
    ).to_csv(args.output_dir / "tuning_coefficients.csv", index=False)
    accepted = selected_loss < market_loss
    write_report(
        args.output_dir,
        market_loss,
        selected_loss,
        selected,
        trace,
        correlation,
        accepted,
    )

    final = data.loc[data["split"].isin(["k_train", "k_tune"])].copy()
    model, mean, std = fit_model(final, selected)
    coefficients = pd.DataFrame(
        {"feature": selected, "coefficient": model.params, "standard_error": model.bse}
    )
    coefficients.to_csv(args.output_dir / "coefficients.csv", index=False)
    output = data.loc[data["split"].isin(["h_s2", "h_test"])].copy()
    output_x = ((output[selected].astype(
        
        
        
        
        
        
        
        float) - mean) / std).fillna(0.0).to_numpy()
    predictions = output[["race_id", "horse_id", "split", "is_winner", "market_prob"]].copy()
    predictions["p_cl"] = race_softmax(output_x @ model.params, output["race_id"])
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    (args.output_dir / "selected_features.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    comparison = "beats" if accepted else "does not beat"
    print(
        f"SELECTED: conditional logit {selected_loss:.4f} {comparison} market "
        f"{market_loss:.4f} on k_tune. Outputs written to {args.output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
