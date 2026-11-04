"""Audit market probabilities and measure feature groups' incremental value."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from conditional_logit_tuning_v2 import MIN_PROB, build_features, prepare, seg_lse


FEATURE_GROUPS = {
    "horse_form": [
        "hf_avg3_fin_pct_rel",
        "hf_last_fin_pct_rel",
        "hf_win5_s",
        "hf_first_start",
        "hf_days_since",
        "hf_class_change",
        "w3_horse_past_win_rate",
        "w3_horse_past_top3_rate",
    ],
    "jockey_trainer": [
        "jt_jockey_win365_rel",
        "jt_trainer_win365_rel",
        "jtp_pair_effect365_rel",
        "jtp_pair_effect_last_50_rel",
    ],
    "draw_weight": ["rc_draw_pct", "rc_weight_rel"],
    # Track and distance are race-level constants. Only runner-varying
    # interactions can be identified by a conditional-logit likelihood.
    "track_distance_interactions": [
        "rc_draw_x_hv",
        "rc_draw_x_sprint",
        "rc_weight_x_dist",
    ],
}


def _offset_nll_gradient(beta: np.ndarray, races) -> tuple[float, np.ndarray]:
    """Conditional-logit NLL and gradient with market log probabilities fixed."""
    utility = races.mkt_logit + races.X @ beta
    total = 0.0
    utility_gradient = np.zeros_like(utility)
    for stage in races.stages:
        valid_races = stage.valid_r & races.mkt_ok
        valid_rows = valid_races[races.seg].astype(float)
        log_sum_exp, probability = seg_lse(utility, races, stage.mask)
        target_utility = np.add.reduceat(stage.y * utility, races.starts)
        total += stage.w * float(
            np.sum(np.where(valid_races, log_sum_exp - target_utility, 0.0))
        )
        utility_gradient += stage.w * valid_rows * (probability - stage.y)
    return total, races.X.T @ utility_gradient


def _fit_offset_model(races) -> tuple[np.ndarray, float]:
    valid_count = int((races.stages[0].valid_r & races.mkt_ok).sum())

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        loss, gradient = _offset_nll_gradient(beta, races)
        return loss / valid_count, gradient / valid_count

    result = minimize(
        objective,
        np.zeros(races.X.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Offset conditional logit failed: {result.message}")
    nll, _ = _offset_nll_gradient(result.x, races)
    return result.x, nll


def _offset_nll(races, beta: np.ndarray) -> tuple[float, float]:
    """Return aggregate and mean per-race NLL under a market-offset model."""
    utility = races.mkt_logit + races.X @ beta
    _, probability = seg_lse(utility, races)
    stage = races.stages[0]
    valid_races = stage.valid_r & races.mkt_ok
    race_loss = np.add.reduceat(
        -stage.y * np.log(np.clip(probability, MIN_PROB, None)), races.starts
    )
    return float(race_loss[valid_races].sum()), float(race_loss[valid_races].mean())


def offset_group_tests(data: pd.DataFrame) -> pd.DataFrame:
    """Fit each group on k_train and score its incremental market value on k_tune."""
    train = data.loc[data["split"].eq("k_train")].copy()
    tune = data.loc[data["split"].eq("k_tune")].copy()
    rows = []

    for group, features in FEATURE_GROUPS.items():
        mean = train[features].astype(float).mean()
        std = train[features].astype(float).std().replace(0, 1).fillna(1)
        train_races = prepare(train, features, mean, std, [1.0])
        tune_races = prepare(tune, features, mean, std, [1.0])

        zero = np.zeros(len(features))
        market_train_nll, _ = _offset_nll(train_races, zero)
        beta, fitted_train_nll = _fit_offset_model(train_races)
        _, market_tune_loss = _offset_nll(tune_races, zero)
        _, fitted_tune_loss = _offset_nll(tune_races, beta)
        lr_statistic = 2 * (market_train_nll - fitted_train_nll)

        rows.append(
            {
                "group": group,
                "features": ", ".join(features),
                "n_features": len(features),
                "train_market_nll": market_train_nll,
                "train_offset_nll": fitted_train_nll,
                "train_lr_statistic": lr_statistic,
                "train_lr_p_value": chi2.sf(lr_statistic, len(features)),
                "k_tune_market_log_loss": market_tune_loss,
                "k_tune_offset_log_loss": fitted_tune_loss,
                "k_tune_gain": market_tune_loss - fitted_tune_loss,
                **{f"coefficient_{feature}": value for feature, value in zip(features, beta)},
            }
        )
    return pd.DataFrame(rows)


def market_baseline(data: pd.DataFrame) -> pd.DataFrame:
    sums = data.groupby("race_id", sort=False)["market_prob"].sum()
    return pd.DataFrame(
        [
            {
                "races": len(sums),
                "rows": len(data),
                "missing_market_probabilities": int(data["market_prob"].isna().sum()),
                "nonpositive_market_probabilities": int((data["market_prob"] <= 0).sum()),
                "mean_race_probability_sum": sums.mean(),
                "min_race_probability_sum": sums.min(),
                "max_race_probability_sum": sums.max(),
                "max_absolute_sum_error": (sums - 1).abs().max(),
            }
        ]
    )


def calibration_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row-level calibration by market-probability decile and race segment."""
    calibrated = data.loc[data["mkt_ok"]].copy()
    calibrated["market_probability"] = calibrated["mkt_p"].astype(float)
    calibrated["outcome"] = calibrated["is_winner"].astype(float)
    calibrated["distance_band"] = pd.cut(
        calibrated["distance_m"].astype(float),
        bins=[0, 1200, 1650, np.inf],
        labels=["sprint_1000_1200", "middle_1400_1650", "staying_1800_plus"],
    )
    calibrated["market_decile"] = pd.qcut(
        calibrated["market_probability"], q=10, labels=False, duplicates="drop"
    ).astype(int) + 1

    def summarise(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        result = frame.groupby(by, observed=True).agg(
            runners=("outcome", "size"),
            predicted_rate=("market_probability", "mean"),
            observed_rate=("outcome", "mean"),
        ).reset_index()
        result["calibration_error"] = result["observed_rate"] - result["predicted_rate"]
        result["observed_to_predicted"] = (
            result["observed_rate"] / result["predicted_rate"].clip(lower=MIN_PROB)
        )
        return result

    by_decile = summarise(calibrated, ["split", "market_decile"])
    # Track and distance apply to every runner in a race. Their aggregate
    # calibration is therefore an identity after probabilities are normalized;
    # nesting probability decile exposes calibration differences within segment.
    by_segment = summarise(
        calibrated,
        ["split", "source", "track_code", "distance_band", "market_decile"],
    )
    return by_decile, by_segment


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data/clean data/runners_merged_with_features.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "artifacts/market_diagnostics"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = build_features(pd.read_parquet(args.data))
    baseline = market_baseline(data)
    groups = offset_group_tests(data)
    deciles, segments = calibration_tables(data)
    baseline.to_csv(args.output_dir / "market_baseline.csv", index=False)
    groups.to_csv(args.output_dir / "offset_feature_groups.csv", index=False)
    deciles.to_csv(args.output_dir / "market_calibration_deciles.csv", index=False)
    segments.to_csv(args.output_dir / "market_calibration_segments.csv", index=False)

    report = [
        "# Market diagnostics",
        "",
        "## Baseline probability audit",
        "",
        "```",
        baseline.to_string(index=False),
        "```",
        "",
        "## Market-offset feature-group tests",
        "",
        "Each group is fitted on `k_train` with `log(market probability)` as a fixed "
        "conditional-logit offset. The likelihood-ratio test is in-sample; `k_tune_gain` "
        "is the out-of-sample per-race log-loss improvement over the market.",
        "",
        "```",
        groups[
            [
                "group",
                "n_features",
                "train_lr_statistic",
                "train_lr_p_value",
                "k_tune_market_log_loss",
                "k_tune_offset_log_loss",
                "k_tune_gain",
            ]
        ].round(6).to_string(index=False),
        "```",
        "",
        "## Calibration",
        "",
        "See `market_calibration_deciles.csv` and `market_calibration_segments.csv`. "
        "Segment results are nested by probability decile because whole-race segment "
        "averages are mechanically calibrated after race-level normalization. Positive "
        "calibration error means the market underestimates that group.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
