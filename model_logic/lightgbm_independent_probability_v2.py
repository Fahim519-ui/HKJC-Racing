"""Odds-free LightGBM win model trained on a per-race softmax likelihood."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

EPSILON = 1e-12

PARAMS = {
    "learning_rate": 0.02,
    "num_leaves": 15,
    "min_data_in_leaf": 200,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.7,
    "lambda_l2": 10.0,
    "max_bin": 63,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
RANK_SOURCES = {
    "jt_jockey_win365": 1,
    "jt_trainer_win365": 1,
    "hf_avg3_fin_pct": -1,
    "hf_win5_s": 1,
    "hf_career_win_s": 1,
    "hf_class_change": 0,
}
MONOTONE = {
    "jt_jockey_win365_rel": 1,
    "jt_trainer_win365_rel": 1,
    "jt_jockey_top3_365_rel": 1,
    "hf_avg3_fin_pct_rel": -1,
    "hf_last_fin_pct_rel": -1,
    "hf_avg5_fin_pct_rel": -1,
    "hf_win5_s": 1,
    "hf_career_win_s": 1,
    "w3_horse_past_win_rate": 1,
    "w3_horse_past_top3_rate": 1,
}


def load_feature_builder() -> object:
    path = Path(__file__).with_name("conditional_logit_tuning_v2.py")
    spec = importlib.util.spec_from_file_location("conditional_logit_tuning_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load feature builder from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Races:
    """Contiguous race segments; every reduction is a reduceat, so races never mix."""

    def __init__(self, race_id) -> None:
        codes = pd.factorize(pd.Series(np.asarray(race_id)))[0]
        change = np.r_[True, codes[1:] != codes[:-1]]
        self.starts = np.flatnonzero(change)
        assert len(self.starts) == codes.max() + 1, "rows of a race must be contiguous"
        self.seg = np.cumsum(change) - 1

    def log_softmax(self, utility: np.ndarray) -> np.ndarray:
        maximum = np.maximum.reduceat(utility, self.starts)
        log_sum_exp = maximum + np.log(
            np.add.reduceat(np.exp(utility - maximum[self.seg]), self.starts)
        )
        return utility - log_sum_exp[self.seg]

    def softmax(self, utility: np.ndarray) -> np.ndarray:
        return np.exp(self.log_softmax(utility))

    def target(self, win: np.ndarray) -> np.ndarray:
        winners = np.add.reduceat(win, self.starts)
        return win / np.maximum(winners, 1.0)[self.seg]

    def loss_vec(self, probability: np.ndarray, win: np.ndarray) -> np.ndarray:
        winner = (win > 0).astype(float)
        n_winners = np.add.reduceat(winner, self.starts)
        mean_probability = np.add.reduceat(probability * winner, self.starts) / np.maximum(
            n_winners, 1.0
        )
        return -np.log(np.clip(mean_probability, EPSILON, None))[n_winners > 0]


def race_loss(probability: np.ndarray | pd.Series, data: pd.DataFrame) -> float:
    races = Races(data["race_id"])
    return float(races.loss_vec(np.asarray(probability, float), data["is_winner"].to_numpy(float)).mean())


def drop_void_races(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[data.groupby("race_id")["is_winner"].transform("max") > 0]


def softmax_objective(races: Races, target: np.ndarray):
    def objective(predictions, _dataset):
        probability = races.softmax(np.asarray(predictions, float))
        return probability - target, np.maximum(2.0 * probability * (1.0 - probability), 1e-6)

    return objective


def train_booster(
    params: dict,
    features: np.ndarray,
    win: np.ndarray,
    races: Races,
    rounds: int,
    valid: tuple | None = None,
    patience: int = 150,
) -> lgb.Booster:
    params = {**params, "objective": softmax_objective(races, races.target(win)), "metric": "None"}
    train_set = lgb.Dataset(features, label=win)
    if valid is None:
        return lgb.train(params, train_set, num_boost_round=rounds)
    valid_features, valid_win, valid_races = valid

    def feval(predictions, _dataset):
        probability = valid_races.softmax(np.asarray(predictions, float))
        return "race_logloss", float(valid_races.loss_vec(probability, valid_win).mean()), False

    return lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=[lgb.Dataset(valid_features, label=valid_win, reference=train_set)],
        feval=feval,
        callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)],
    )


def bag_utility(boosters: list[lgb.Booster], features: np.ndarray) -> np.ndarray:
    return np.mean([booster.predict(features, raw_score=True) for booster in boosters], axis=0)


def add_context_features(data: pd.DataFrame, features: list[str]) -> list[str]:
    result = list(features)
    for column in RANK_SOURCES:
        if column in data.columns:
            data[f"{column}_rk"] = data.groupby("race_id")[column].rank(pct=True)
            result.append(f"{column}_rk")
    result.append("n_runners")
    return result


def monotone_vector(features: list[str]) -> list[int]:
    signs = dict(MONOTONE)
    for column, sign in RANK_SOURCES.items():
        signs[f"{column}_rk"] = sign
    return [signs.get(feature, 0) for feature in features]


def fit_blend(frame: pd.DataFrame, columns: list[str], lam: float) -> np.ndarray:
    """Fit a ridge blend centered on market-only conditional-logit coefficients."""
    races = Races(frame["race_id"])
    features = np.log(frame[columns].clip(lower=EPSILON).to_numpy(float))
    target = races.target(frame["is_winner"].to_numpy(float))
    n_races = float((np.add.reduceat(target, races.starts) > 0).sum())
    prior = np.zeros(len(columns))
    prior[0] = 1.0

    def loss_and_gradient(theta):
        log_probability = races.log_softmax(features @ theta)
        difference = theta - prior
        gradient = features.T @ (np.exp(log_probability) - target) / n_races + 2 * lam * difference
        return -(target * log_probability).sum() / n_races + lam * difference @ difference, gradient

    return minimize(loss_and_gradient, prior, jac=True, method="L-BFGS-B").x


def blend_probability(frame: pd.DataFrame, columns: list[str], theta: np.ndarray) -> np.ndarray:
    features = np.log(frame[columns].clip(lower=EPSILON).to_numpy(float))
    return Races(frame["race_id"]).softmax(features @ theta)


def month_cv(frame: pd.DataFrame, columns: list[str], lambdas: list[float]) -> dict[float, float]:
    month = frame["race_date"].dt.strftime("%Y-%m").to_numpy()
    scores = {}
    for lam in lambdas:
        fold_losses = []
        for held_out in np.unique(month):
            train, test = frame.loc[month != held_out], frame.loc[month == held_out]
            probability = blend_probability(test, columns, fit_blend(train, columns, lam))
            fold_losses.append(Races(test["race_id"]).loss_vec(
                probability, test["is_winner"].to_numpy(float)
            ))
        scores[lam] = float(np.concatenate(fold_losses).mean())
    return scores


def merge_conditional_logit(stage1: pd.DataFrame, path: Path) -> pd.DataFrame:
    conditional_logit = pd.read_parquet(path)[["race_id", "horse_id", "split", "p_cl"]]
    merged = stage1.merge(
        conditional_logit,
        on=["race_id", "horse_id", "split"],
        how="left",
        validate="one_to_one",
    )
    if merged["p_cl"].isna().any():
        raise ValueError(f"{int(merged['p_cl'].isna().sum())} rows lack a conditional-logit prediction")
    return merged


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=root / "data/clean data/runners_merged_with_features.parquet"
    )
    parser.add_argument(
        "--conditional-logit-predictions",
        type=Path,
        default=root / "artifacts/conditional_logit_tuning_v2/predictions.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "artifacts/lightgbm_independent_probability_v2",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=4000)
    parser.add_argument("--no-monotone", action="store_true")
    parser.add_argument(
        "--blend-lambdas", type=float, nargs="+", default=[0.0, 0.003, 0.01, 0.03, 0.1]
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    builder = load_feature_builder()
    data = builder.build_features(pd.read_parquet(args.data))
    features = add_context_features(data, list(builder.ALL_FEATURES))
    params = dict(PARAMS)
    if not args.no_monotone:
        params["monotone_constraints"] = monotone_vector(features)
        params["monotone_constraints_method"] = "intermediate"

    train = drop_void_races(data.loc[data["split"].eq("k_train")])
    tune = drop_void_races(data.loc[data["split"].eq("k_tune")])
    stage2 = data.loc[data["split"].eq("h_s2")]
    test = data.loc[data["split"].eq("h_test")]
    if min(len(train), len(tune), len(stage2), len(test)) == 0:
        raise ValueError("k_train, k_tune, h_s2 and h_test must all contain rows")

    def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, Races]:
        return (
            frame[features].to_numpy(float),
            frame["is_winner"].to_numpy(float),
            Races(frame["race_id"]),
        )

    train_features, train_win, train_races = arrays(train)
    tune_features, tune_win, tune_races = arrays(tune)
    seeds = [17 + 101 * index for index in range(args.seeds)]

    def seeded(seed: int) -> dict:
        return {
            **params,
            "seed": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
        }

    first = train_booster(
        seeded(seeds[0]),
        train_features,
        train_win,
        train_races,
        args.max_rounds,
        (tune_features, tune_win, tune_races),
    )
    best = int(first.best_iteration or 300)
    probe = [first] + [
        train_booster(seeded(seed), train_features, train_win, train_races, best)
        for seed in seeds[1:]
    ]
    tune_probability = tune_races.softmax(bag_utility(probe, tune_features))

    final_data = pd.concat([train, tune], ignore_index=True)
    final_features, final_win, final_races = arrays(final_data)
    rounds = max(50, int(round(best * len(final_races.starts) / len(train_races.starts))))
    boosters = [
        train_booster(seeded(seed), final_features, final_win, final_races, rounds)
        for seed in seeds
    ]
    for index, booster in enumerate(boosters):
        booster.save_model(str(args.output_dir / f"lightgbm_seed{index}.txt"))

    prediction_frames = []
    for window in (stage2, test):
        utility = bag_utility(boosters, window[features].to_numpy(float))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "race_id": window["race_id"].to_numpy(),
                    "horse_id": window["horse_id"].to_numpy(),
                    "split": window["split"].to_numpy(),
                    "race_date": window["race_date"].to_numpy(),
                    "is_winner": window["is_winner"].to_numpy(),
                    "p_market": window["mkt_p"].to_numpy(),
                    "market_complete": window["mkt_ok"].to_numpy(),
                    "p_lgbm": Races(window["race_id"]).softmax(utility),
                }
            )
        )
    stage1 = pd.concat(prediction_frames, ignore_index=True)
    stage1.to_parquet(args.output_dir / "stage1_predictions.parquet", index=False)

    joined = merge_conditional_logit(stage1, args.conditional_logit_predictions)
    complete_market = joined["market_complete"].astype(bool)
    blend_train = drop_void_races(joined.loc[joined["split"].eq("h_s2") & complete_market])
    blend_test = drop_void_races(joined.loc[joined["split"].eq("h_test") & complete_market])

    rows = [
        ("k_tune", "market_raw", race_loss(tune["market_prob"], tune)),
        ("k_tune", "market_norm", race_loss(tune["mkt_p"], tune)),
        ("k_tune", "lightgbm", race_loss(tune_probability, tune)),
    ]
    for name, column in (
        ("market", "p_market"),
        ("lightgbm", "p_lgbm"),
        ("conditional_logit", "p_cl"),
    ):
        rows.append(("h_s2", name, race_loss(blend_train[column], blend_train)))
        rows.append(("h_test", name, race_loss(blend_test[column], blend_test)))

    variants = {
        "mkt+lgbm": ["p_market", "p_lgbm"],
        "mkt+cl": ["p_market", "p_cl"],
        "mkt+lgbm+cl": ["p_market", "p_lgbm", "p_cl"],
    }
    coefficient_rows, cv_rows = [], []
    for name, columns in variants.items():
        cv = month_cv(blend_train, columns, args.blend_lambdas)
        lam = min(cv, key=cv.get)
        theta = fit_blend(blend_train, columns, lam)
        rows.append(("h_s2", f"{name} (month-CV)", cv[lam]))
        rows.append((
            "h_test",
            f"{name} (fit on h_s2)",
            race_loss(blend_probability(blend_test, columns, theta), blend_test),
        ))
        coefficient_rows.append({"variant": name, "lambda": lam, **dict(zip(columns, theta))})
        cv_rows.extend({"variant": name, "lambda": value, "cv_loss": loss} for value, loss in cv.items())
        if name == "mkt+lgbm+cl":
            joined["p_joint"] = blend_probability(joined, columns, theta)
    joined.to_parquet(args.output_dir / "joint_predictions.parquet", index=False)

    metrics = pd.DataFrame(rows, columns=["window", "model", "per_race_log_loss"])
    metrics["market_gain"] = metrics.groupby("window")["per_race_log_loss"].transform(
        lambda value: value.iloc[0] - value
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_csv(args.output_dir / "joint_coefficients.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(args.output_dir / "blend_cv.csv", index=False)
    pd.DataFrame(
        {
            "feature": features,
            "gain": np.sum([booster.feature_importance("gain") for booster in boosters], axis=0),
        }
    ).sort_values("gain", ascending=False).to_csv(args.output_dir / "feature_importance.csv", index=False)
    (args.output_dir / "stage1_features.json").write_text(json.dumps(features, indent=2) + "\n")

    report = [
        "# Odds-free listwise LightGBM",
        "",
        "Stage 1 optimises per-race softmax likelihood on Kaggle data and never uses odds "
        "or market-derived columns. `k_tune` selects only the round count "
        f"({best} early-stopped, {rounds} on train+tune, {args.seeds}-seed bag).",
        "Stage-2 ridge strength is selected by leave-one-month-out CV within `h_s2`; "
        "`h_test` is scored once per variant.",
        "",
        "## Per-race log loss",
        "",
        "```",
        metrics.round(6).to_string(index=False),
        "```",
        "",
        "## Blend coefficients",
        "",
        "```",
        coefficients.round(6).to_string(index=False),
        "```",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(metrics.round(6).to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
