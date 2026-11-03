"""Train an odds-free LightGBM win model and evaluate a market-aware final blend.

Stage 1 uses only pre-race physical, race-condition, and strictly lagged
historical form variables.  It never receives odds, implied probabilities, or
market-derived variables.  Its binary probabilities are normalized within each
race to produce a coherent win-probability distribution.

Stage 2 is deliberately separate: it fits a conditional-logit blend of market,
LightGBM, and conditional-logit probabilities on ``h_s2`` and evaluates it once
on the later ``h_test`` window.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

EPSILON = 1e-12
SEED = 17

# These are all constructed by the leakage-safe feature builder from physical
# race conditions and information strictly before the current race day.
FEATURES = [
    "rc_draw_pct",
    "rc_draw_x_hv",
    "rc_draw_x_sprint",
    "rc_weight_rel",
    "rc_weight_x_dist",
    "hf_days_since",
    "hf_first_start",
    "hf_last_fin_pct_rel",
    "hf_avg3_fin_pct_rel",
    "hf_win5_s",
    "hf_class_change",
    "jt_jockey_win365_rel",
    "jt_trainer_win365_rel",
    "w3_horse_past_win_rate",
    "w3_horse_past_top3_rate",
]

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.02,
    "num_leaves": 15,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
    "random_state": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def load_feature_builder() -> object:
    """Load the established v2 builder without duplicating leakage-sensitive code."""
    path = Path(__file__).with_name("conditional_logit_tuning_v2.py")
    spec = importlib.util.spec_from_file_location("conditional_logit_tuning_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load feature builder from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_within_race(values: np.ndarray, race_id: pd.Series) -> pd.Series:
    probability = pd.Series(np.asarray(values, dtype=float), index=race_id.index)
    probability = probability.clip(lower=EPSILON)
    return probability / probability.groupby(race_id).transform("sum")


def race_log_loss(probability: pd.Series, data: pd.DataFrame) -> float:
    winners = data["is_winner"].eq(1)
    winner_probability = probability.loc[winners].groupby(
        data.loc[winners, "race_id"]
    ).mean()
    return float(-np.log(winner_probability.clip(lower=EPSILON)).mean())


def predict(model: lgb.LGBMClassifier, data: pd.DataFrame) -> pd.Series:
    raw_probability = model.predict_proba(data[FEATURES].astype(float))[:, 1]
    return normalize_within_race(raw_probability, data["race_id"])


def fit_stage1(train: pd.DataFrame, tune: pd.DataFrame) -> tuple[lgb.LGBMClassifier, int]:
    probe = lgb.LGBMClassifier(n_estimators=4_000, **LGB_PARAMS)
    probe.fit(
        train[FEATURES].astype(float),
        train["is_winner"].astype(int),
        eval_set=[(tune[FEATURES].astype(float), tune["is_winner"].astype(int))],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return probe, int(probe.best_iteration_ or 300)


def fit_blend(data: pd.DataFrame) -> tuple[ConditionalLogit, pd.Series]:
    columns = ["p_market", "p_lgbm", "p_cl"]
    matrix = np.log(data[columns].clip(lower=EPSILON).to_numpy(float))
    model = ConditionalLogit(
        data["is_winner"].astype(float).to_numpy(),
        matrix,
        groups=data["race_id"].to_numpy(),
    ).fit(disp=0)
    return model, pd.Series(model.params, index=columns)


def blend_probability(data: pd.DataFrame, coefficients: np.ndarray) -> pd.Series:
    matrix = np.log(data[["p_market", "p_lgbm", "p_cl"]].clip(lower=EPSILON).to_numpy(float))
    score = matrix @ coefficients
    score = pd.Series(score, index=data.index)
    exponentiated = np.exp(score - score.groupby(data["race_id"]).transform("max"))
    return exponentiated / exponentiated.groupby(data["race_id"]).transform("sum")


def merge_conditional_logit(
    stage1: pd.DataFrame, conditional_logit_predictions: Path
) -> pd.DataFrame:
    cl = pd.read_parquet(conditional_logit_predictions)[
        ["race_id", "horse_id", "split", "p_cl"]
    ]
    merged = stage1.merge(cl, on=["race_id", "horse_id", "split"], how="left", validate="one_to_one")
    if merged["p_cl"].isna().any():
        missing = int(merged["p_cl"].isna().sum())
        raise ValueError(f"{missing} LightGBM rows have no matching conditional-logit prediction")
    return merged


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data/clean data/runners_merged_with_features.parquet",
    )
    parser.add_argument(
        "--conditional-logit-predictions",
        type=Path,
        default=root / "artifacts/conditional_logit_tuning_v2/predictions.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "artifacts/lightgbm_independent_probability"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    builder = load_feature_builder()
    data = builder.build_features(pd.read_parquet(args.data))
    train = data.loc[data["split"].eq("k_train")].copy()
    tune = data.loc[data["split"].eq("k_tune")].copy()
    stage2 = data.loc[data["split"].eq("h_s2")].copy()
    test = data.loc[data["split"].eq("h_test")].copy()
    if min(len(train), len(tune), len(stage2), len(test)) == 0:
        raise ValueError("All k_train, k_tune, h_s2, and h_test splits must contain rows")

    probe, n_estimators = fit_stage1(train, tune)
    p_tune = predict(probe, tune)
    tune_market = tune["mkt_p"].astype(float)

    final_data = pd.concat([train, tune], ignore_index=True)
    final = lgb.LGBMClassifier(n_estimators=n_estimators, **LGB_PARAMS).fit(
        final_data[FEATURES].astype(float), final_data["is_winner"].astype(int)
    )
    final.booster_.save_model(str(args.output_dir / "lightgbm.txt"))

    prediction_frames = []
    for window in (stage2, test):
        prediction_frames.append(
            pd.DataFrame(
                {
                    "race_id": window["race_id"],
                    "horse_id": window["horse_id"],
                    "split": window["split"],
                    "is_winner": window["is_winner"],
                    "p_market": window["mkt_p"],
                    "market_complete": window["mkt_ok"],
                    "p_lgbm": predict(final, window),
                }
            )
        )
    stage1 = pd.concat(prediction_frames, ignore_index=True)
    stage1.to_parquet(args.output_dir / "stage1_predictions.parquet", index=False)

    joined = merge_conditional_logit(stage1, args.conditional_logit_predictions)
    blend_train = joined.loc[joined["split"].eq("h_s2") & joined["market_complete"]].copy()
    blend_test = joined.loc[joined["split"].eq("h_test") & joined["market_complete"]].copy()
    blend, coefficients = fit_blend(blend_train)
    joined["p_joint"] = blend_probability(joined, blend.params)
    joined.to_parquet(args.output_dir / "joint_predictions.parquet", index=False)

    importances = pd.DataFrame(
        {"feature": FEATURES, "gain": final.booster_.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False)
    importances.to_csv(args.output_dir / "feature_importance.csv", index=False)
    coefficients.rename("coefficient").to_csv(args.output_dir / "joint_coefficients.csv")
    (args.output_dir / "stage1_features.json").write_text(
        json.dumps(FEATURES, indent=2) + "\n", encoding="utf-8"
    )

    report_rows = [
        ("k_tune", "market", race_log_loss(tune_market, tune)),
        ("k_tune", "lightgbm", race_log_loss(p_tune, tune)),
        ("h_s2", "market", race_log_loss(blend_train["p_market"], blend_train)),
        ("h_s2", "lightgbm", race_log_loss(blend_train["p_lgbm"], blend_train)),
        ("h_s2", "conditional_logit", race_log_loss(blend_train["p_cl"], blend_train)),
        ("h_s2", "joint_in_sample", race_log_loss(blend_probability(blend_train, blend.params), blend_train)),
        ("h_test", "market", race_log_loss(blend_test["p_market"], blend_test)),
        ("h_test", "lightgbm", race_log_loss(blend_test["p_lgbm"], blend_test)),
        ("h_test", "conditional_logit", race_log_loss(blend_test["p_cl"], blend_test)),
        ("h_test", "joint", race_log_loss(blend_probability(blend_test, blend.params), blend_test)),
    ]
    metrics = pd.DataFrame(report_rows, columns=["window", "model", "per_race_log_loss"])
    metrics["market_gain"] = metrics.groupby("window")["per_race_log_loss"].transform(
        lambda value: value.iloc[0] - value
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    report = [
        "# Odds-free LightGBM probability model",
        "",
        "Stage 1 uses only the listed physical and strictly lagged historical features. "
        "It excludes odds, implied probability, normalized market probability, and every "
        "other market-derived column. The binary classifier output is normalized within race.",
        "",
        f"- LightGBM early-stopping trees: {n_estimators}",
        "- `k_tune` selects only the tree count; it is not an unbiased model comparison.",
        "- The joint conditional-logit blend is fit on `h_s2` and evaluated on the later `h_test` window.",
        "",
        "## Per-race log loss",
        "",
        "```",
        metrics.round(6).to_string(index=False),
        "```",
        "",
        "## Joint blend coefficients",
        "",
        "```",
        coefficients.round(6).to_string(),
        "```",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(metrics.round(6).to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
