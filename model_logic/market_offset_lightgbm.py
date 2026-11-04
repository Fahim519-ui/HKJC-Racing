"""Walk-forward diagnostic: can LightGBM add nonlinear signal beyond the market?"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from conditional_logit_tuning_v2 import ALL_FEATURES, build_features

EPSILON = 1e-12
FOLDS = [
    ("2018-08-01", "2019-02-01"),
    ("2019-02-01", "2019-08-01"),
    ("2019-08-01", "2020-02-01"),
    ("2020-02-01", "2020-08-01"),
]
PARAMS = {
    "learning_rate": 0.01,
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 100.0,
    "max_bin": 63,
    "deterministic": True,
    "force_row_wise": True,
    "seed": 17,
    "bagging_seed": 17,
    "feature_fraction_seed": 17,
    "verbose": -1,
}


class Races:
    """Contiguous race segments for per-race softmax losses."""

    def __init__(self, race_id: pd.Series) -> None:
        codes = pd.factorize(race_id)[0]
        change = np.r_[True, codes[1:] != codes[:-1]]
        self.starts = np.flatnonzero(change)
        self.seg = np.cumsum(change) - 1

    def softmax(self, utility: np.ndarray) -> np.ndarray:
        maximum = np.maximum.reduceat(utility, self.starts)
        exponent = np.exp(utility - maximum[self.seg])
        return exponent / np.add.reduceat(exponent, self.starts)[self.seg]

    def target(self, winner: np.ndarray) -> np.ndarray:
        winner_count = np.add.reduceat(winner, self.starts)
        return winner / np.maximum(winner_count, 1.0)[self.seg]

    def loss(self, probability: np.ndarray, winner: np.ndarray) -> float:
        target = self.target(winner)
        valid = np.add.reduceat(winner, self.starts) > 0
        race_loss = np.add.reduceat(
            -target * np.log(np.clip(probability, EPSILON, None)), self.starts
        )
        return float(race_loss[valid].mean())


def residual_objective(races: Races, target: np.ndarray):
    """Train trees against race-softmax residuals around the market init_score."""

    def objective(predictions: np.ndarray, _dataset):
        probability = races.softmax(np.asarray(predictions, float))
        gradient = probability - target
        hessian = np.maximum(2.0 * probability * (1.0 - probability), 1e-6)
        return gradient, hessian

    return objective


def train_fold(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    max_rounds: int,
    features: list[str] = ALL_FEATURES,
) -> dict:
    mean = train[features].astype(float).mean()
    std = train[features].astype(float).std().replace(0, 1).fillna(1)

    def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, Races]:
        matrix = ((frame[features].astype(float) - mean) / std).fillna(0.0).to_numpy()
        winners = frame["is_winner"].to_numpy(float)
        market_offset = np.log(frame["mkt_p"].clip(lower=EPSILON).to_numpy(float))
        return matrix, winners, market_offset, Races(frame["race_id"])

    x_train, y_train, init_train, races_train = arrays(train)
    x_valid, y_valid, init_valid, races_valid = arrays(valid)
    train_set = lgb.Dataset(x_train, label=y_train, init_score=init_train)
    valid_set = lgb.Dataset(x_valid, label=y_valid, init_score=init_valid, reference=train_set)
    target_train = races_train.target(y_train)

    def evaluate(predictions: np.ndarray, _dataset):
        probability = races_valid.softmax(np.asarray(predictions, float))
        return "race_log_loss", races_valid.loss(probability, y_valid), False

    booster = lgb.train(
        {
            **PARAMS,
            "objective": residual_objective(races_train, target_train),
            "metric": "None",
        },
        train_set,
        num_boost_round=max_rounds,
        valid_sets=[valid_set],
        feval=evaluate,
        callbacks=[lgb.early_stopping(200, first_metric_only=True, verbose=False)],
    )
    best_iteration = int(booster.best_iteration or max_rounds)

    # Dataset init_score is used during training but is not persisted in the
    # Booster. Reattach the fixed market offset when scoring predictions.
    tree_utility = booster.predict(x_valid, raw_score=True)
    residual_probability = races_valid.softmax(init_valid + tree_utility)
    market_probability = races_valid.softmax(init_valid)
    return {
        "market_log_loss": races_valid.loss(market_probability, y_valid),
        "residual_log_loss": races_valid.loss(residual_probability, y_valid),
        "best_iteration": best_iteration,
        "train_races": len(races_train.starts),
        "valid_races": len(races_valid.starts),
        "train_rows": len(train),
        "valid_rows": len(valid),
    }


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
        default=root / "artifacts/market_offset_lightgbm",
    )
    parser.add_argument("--max-rounds", type=int, default=4000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = build_features(pd.read_parquet(args.data))
    data = data.loc[data["source"].eq("kaggle") & data["mkt_ok"]].copy()
    data = data.loc[data.groupby("race_id")["is_winner"].transform("max").gt(0)].copy()

    rows = []
    for valid_start, valid_end in FOLDS:
        train = data.loc[data["race_date"] < valid_start].copy()
        valid = data.loc[
            data["race_date"].ge(valid_start) & data["race_date"].lt(valid_end)
        ].copy()
        if train.empty or valid.empty:
            raise ValueError(f"Fold {valid_start} to {valid_end} has no train or validation rows")
        result = train_fold(train, valid, args.max_rounds)
        rows.append(
            {
                "validation_start": valid_start,
                "validation_end": valid_end,
                **result,
                "market_gain": result["market_log_loss"] - result["residual_log_loss"],
            }
        )

    metrics = pd.DataFrame(rows)
    weighted_market = np.average(metrics["market_log_loss"], weights=metrics["valid_races"])
    weighted_residual = np.average(metrics["residual_log_loss"], weights=metrics["valid_races"])
    summary = pd.DataFrame(
        [
            {
                "folds": len(metrics),
                "positive_gain_folds": int((metrics["market_gain"] > 0).sum()),
                "weighted_market_log_loss": weighted_market,
                "weighted_residual_log_loss": weighted_residual,
                "weighted_market_gain": weighted_market - weighted_residual,
            }
        ]
    )
    metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    report = [
        "# Market-offset LightGBM diagnostic",
        "",
        "Trees are trained with `init_score = log(normalized market probability)` and a "
        "per-race softmax loss. This is diagnostic only: it uses the market inside the "
        "learner and is not a valid odds-free Stage-1 model.",
        "",
        "## Walk-forward folds",
        "",
        "```",
        metrics.round(6).to_string(index=False),
        "```",
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
