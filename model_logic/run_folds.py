"""Chronological, fold-isolated model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from model_logic.ablation_configs import get_config
from model_logic.conditional_logit_tuning_v2 import (
    RaceSet,
    fit_beta,
    make_stages,
    moments,
    penalty_mats,
    seg_lse,
    select_features,
    standard_errors,
    tune_loss,
)
from model_logic.feature_builder import build_features
from model_logic.feature_registry import (
    FORBIDDEN_INPUTS,
    REGISTRY_VERSION,
    get_features,
    get_spec,
    registry_hash,
)
from model_logic.lightgbm_independent_probability_v2 import (
    PARAMS,
    Races,
    add_context_features,
    bag_utility,
    blend_probability,
    fit_blend,
    monotone_vector,
    train_booster,
)


TEST_START = pd.Timestamp("2026-04-01")
EPSILON = 1e-12
PENALTIES = (0.0, 0.003, 0.01)
BLEND_LAMBDAS = (0.0, 0.003, 0.01, 0.03, 0.1)
REQUIRED_FOLD_COLUMNS = (
    "fold_id", "era", "s1_start", "s1_end", "s2_start", "s2_end", "valid_start", "valid_end",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_folds(path: Path) -> pd.DataFrame:
    folds = pd.read_csv(path)
    missing = set(REQUIRED_FOLD_COLUMNS).difference(folds.columns)
    if missing:
        raise ValueError(f"Fold table missing required columns: {sorted(missing)}")
    folds = folds.loc[:, REQUIRED_FOLD_COLUMNS].copy()
    for column in REQUIRED_FOLD_COLUMNS[2:]:
        folds[column] = pd.to_datetime(folds[column], errors="raise")
    if not folds["era"].isin(("kaggle", "hkjc")).all():
        raise ValueError("Fold era must be kaggle or hkjc")
    if folds["fold_id"].duplicated().any():
        raise ValueError("Fold ids must be unique")
    invalid = ~(
        (folds["s1_end"] < folds["s2_start"])
        & (folds["s2_start"] <= folds["s2_end"])
        & (folds["s2_end"] < folds["valid_start"])
        & (folds["valid_start"] <= folds["valid_end"])
        & (folds["valid_end"] < TEST_START)
    )
    if invalid.any():
        raise ValueError(f"Invalid chronological fold rows: {folds.loc[invalid, 'fold_id'].tolist()}")
    if not folds["valid_start"].is_monotonic_increasing:
        raise ValueError("Folds must be ordered by valid_start")
    return folds


def select_period(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return data.loc[data["race_date"].between(start, end)].copy()


def validate_periods(periods: dict[str, pd.DataFrame]) -> None:
    seen: set[object] = set()
    for name, frame in periods.items():
        ids = set(frame["race_id"])
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"Race ids overlap period {name}: {sorted(overlap)[:3]}")
        seen.update(ids)


def valid_races(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.groupby("race_id")["is_winner"].transform("max").gt(0)].copy()


def market_complete(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    complete = frame.loc[frame["mkt_ok"].astype(bool)].copy()
    return complete, int(frame["race_id"].nunique() - complete["race_id"].nunique())


def stage1_races(
    frame: pd.DataFrame, features: list[str], mean: pd.Series, std: pd.Series, labels: bool = True
) -> RaceSet:
    """Prepare a CL choice set without reading any market-derived values."""
    codes = pd.factorize(frame["race_id"])[0]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    counts = np.diff(np.r_[starts, len(frame)])
    seg = np.cumsum(np.r_[True, codes[1:] != codes[:-1]]) - 1
    matrix = ((frame[features].astype(float) - mean[features]) / std[features]).fillna(0.0)
    races = RaceSet(
        X=np.ascontiguousarray(matrix.to_numpy(), dtype=np.float64),
        starts=starts,
        counts=counts,
        seg=seg,
        mkt_logit=np.zeros(len(frame)),
        mkt_ok=np.zeros(len(starts), dtype=bool),
    )
    if labels:
        races.stages = make_stages(frame["finish_pos"].to_numpy(float), races, [1.0])
        races.S, _ = penalty_mats(races)
    return races


def probability_sum_check(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = frame[column].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains NaN or infinite values")
        sums = frame.groupby("race_id")[column].sum()
        if not np.allclose(sums, 1.0, atol=1e-9):
            raise ValueError(f"{column} does not sum to one per race")


def inner_split(s1: pd.DataFrame, months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = s1["race_date"].max().normalize() + pd.Timedelta(days=1)
    begin = end - pd.DateOffset(months=months)
    return s1.loc[s1["race_date"] < begin].copy(), s1.loc[s1["race_date"] >= begin].copy()


def fit_conditional_logit(
    inner_train: pd.DataFrame,
    inner_tune: pd.DataFrame,
    s1: pd.DataFrame,
    targets: list[pd.DataFrame],
    features: list[str],
    base_features: list[str],
    minimum_gain: float,
) -> tuple[dict, list[pd.DataFrame]]:
    mean, std = moments(inner_train, features)
    train_races = stage1_races(inner_train, features, mean, std)
    tune_races = stage1_races(inner_tune, features, mean, std)
    index, trace = select_features(
        train_races, tune_races, inner_train[features].corr(method="spearman"),
        minimum_gain, features, base_features,
    )
    selected = [features[item] for item in index]
    train_selected, tune_selected = train_races.select(index), tune_races.select(index)
    rows = []
    for lam in PENALTIES:
        beta = fit_beta(train_selected, lam=lam, l1_ratio=0.5, lam_mkt=0.0)
        rows.append({"lam": lam, "loss": tune_loss(beta, tune_selected)})
    grid = pd.DataFrame(rows)
    lam = float(grid.loc[grid["loss"].idxmin(), "lam"])

    final_mean, final_std = moments(s1, selected)
    final = stage1_races(s1, selected, final_mean, final_std)
    beta = fit_beta(final, lam=lam, l1_ratio=0.5, lam_mkt=0.0)
    coefficients = pd.DataFrame(
        {
            "feature": selected,
            "coefficient": beta,
            "standard_error": standard_errors(beta, final, lam, 0.5, 0.0),
        }
    )
    predictions = []
    for target in targets:
        target_races = stage1_races(target, selected, final_mean, final_std, labels=False)
        _, probability = seg_lse(target_races.X @ beta, target_races)
        predictions.append(pd.DataFrame({"p_cl": probability}, index=target.index))
    return {
        "selected": selected, "trace": trace, "grid": grid, "coefficients": coefficients,
        "lambda": lam,
    }, predictions


def lgbm_matrix(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, Races]:
    return (
        frame[features].to_numpy(float),
        frame["is_winner"].to_numpy(float),
        Races(frame["race_id"]),
    )


def fit_lgbm(
    inner_train: pd.DataFrame,
    inner_tune: pd.DataFrame,
    s1: pd.DataFrame,
    targets: list[pd.DataFrame],
    features: list[str],
    seeds: list[int],
) -> tuple[dict, list[pd.DataFrame]]:
    train_x, train_y, train_races = lgbm_matrix(inner_train, features)
    tune_x, tune_y, tune_races = lgbm_matrix(inner_tune, features)
    params = {**PARAMS, "monotone_constraints": monotone_vector(features),
              "monotone_constraints_method": "intermediate"}
    seeded = lambda seed: {**params, "seed": seed, "bagging_seed": seed,
                           "feature_fraction_seed": seed}
    probe = train_booster(seeded(seeds[0]), train_x, train_y, train_races, 4000,
                          (tune_x, tune_y, tune_races))
    best = int(probe.best_iteration or 300)
    s1_x, s1_y, s1_races = lgbm_matrix(s1, features)
    rounds = max(50, round(best * len(s1_races.starts) / len(train_races.starts)))
    boosters = [
        train_booster(seeded(seed), s1_x, s1_y, s1_races, rounds)
        for seed in seeds
    ]
    predictions = []
    for target in targets:
        target_x, _, target_races = lgbm_matrix(target, features)
        predictions.append(pd.DataFrame(
            {"p_lgbm": target_races.softmax(bag_utility(boosters, target_x))}, index=target.index
        ))
    return {"best_iter": best, "rounds": rounds, "seeds": seeds, "params": params}, predictions


def forward_lambda_scores(frame: pd.DataFrame, columns: list[str]) -> tuple[dict[float, float], bool]:
    months = np.sort(frame["race_date"].dt.to_period("M").unique())
    if len(months) < 3:
        return {0.03: float("nan")}, True
    scores = {}
    for lam in BLEND_LAMBDAS:
        losses = []
        for month in months[1:]:
            train = frame.loc[frame["race_date"].dt.to_period("M") < month]
            test = frame.loc[frame["race_date"].dt.to_period("M") == month]
            probability = blend_probability(test, columns, fit_blend(train, columns, lam))
            losses.extend(Races(test["race_id"]).loss_vec(
                probability, test["is_winner"].to_numpy(float)
            ))
        scores[lam] = float(np.mean(losses))
    return scores, False


def add_stage2_variants(
    s2: pd.DataFrame, valid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    columns = ["p_market", "p_cl", "p_lgbm"]
    lambda_scores, fallback = forward_lambda_scores(s2, columns)
    lam = 0.03 if fallback else min(lambda_scores, key=lambda_scores.get)
    variants = {
        "uniform": [],
        "market": ["p_market"],
        "market_calibrated": ["p_market"],
        "cl_only": ["p_cl"],
        "lgbm_only": ["p_lgbm"],
        "mkt+cl": ["p_market", "p_cl"],
        "mkt+lgbm": ["p_market", "p_lgbm"],
        "mkt+cl+lgbm": columns,
    }
    outputs = [s2.copy(), valid.copy()]
    theta = {}
    for name, inputs in variants.items():
        if name == "uniform":
            for output in outputs:
                output[f"p_{name}"] = 1.0 / output["n_runners"]
        elif name == "market":
            for output in outputs:
                output[f"p_{name}"] = output["p_market"]
        elif name == "market_calibrated":
            value = fit_blend(s2, inputs, 0.0)
            for output in outputs:
                output[f"p_{name}"] = blend_probability(output, inputs, value)
            theta[name] = value.tolist()
        else:
            value = fit_blend(s2, inputs, lam)
            for output in outputs:
                output[f"p_{name}"] = blend_probability(output, inputs, value)
            theta[name] = value.tolist()
    return outputs[0], outputs[1], {
        "lambda": lam, "lambda_scores": lambda_scores, "fallback": fallback, "theta": theta
    }


def run_fold(
    fold: pd.Series, data: pd.DataFrame, config_name: str, seeds: list[int],
    inner_tune_months: int, minimum_gain: float, output: Path, provenance: dict,
) -> dict:
    started = time.monotonic()
    config = get_config(config_name)
    lgbm_features = get_features(
        "lgbm", exclude_groups=config.exclude_groups, exclude_names=config.exclude_names,
        available_columns=data.columns,
    )
    lgbm_features = [feature for feature in lgbm_features if not get_spec(feature).optional]
    lgbm_features = add_context_features(
        data, lgbm_features, include_trainer_rates="jt_trainer_win365_rel" in lgbm_features
    )
    s1 = valid_races(select_period(data, fold.s1_start, fold.s1_end))
    s2, s2_dropped = market_complete(select_period(data, fold.s2_start, fold.s2_end))
    valid, valid_dropped = market_complete(select_period(data, fold.valid_start, fold.valid_end))
    validate_periods({"s1": s1, "s2": s2, "valid": valid})
    inner_train, inner_tune = inner_split(s1, inner_tune_months)
    if inner_train.empty or inner_tune.empty:
        raise ValueError(f"{fold.fold_id}: inner train/tune split is empty")
    inner_tune_races = inner_tune["race_id"].nunique()
    if inner_tune_races < 150:
        print(f"WARNING {fold.fold_id}: inner tune has only {inner_tune_races} races", file=sys.stderr)

    cl_features = get_features(
        "conditional_logit", exclude_groups=config.exclude_groups,
        exclude_names=config.exclude_names, available_columns=data.columns,
    )
    base = get_features(
        "conditional_logit", role="base", exclude_groups=config.exclude_groups,
        exclude_names=config.exclude_names, available_columns=data.columns,
    )
    forbidden = [feature for feature in cl_features if feature in FORBIDDEN_INPUTS or feature.startswith("mkt_")]
    if forbidden:
        raise ValueError(f"Stage-1 CL receives forbidden features: {forbidden}")
    cl, cl_predictions = fit_conditional_logit(
        inner_train, inner_tune, s1, [s2, valid], cl_features, base, minimum_gain
    )
    s2 = s2.join(cl_predictions[0])
    valid = valid.join(cl_predictions[1])

    lgbm, lgbm_predictions = fit_lgbm(inner_train, inner_tune, s1, [s2, valid], lgbm_features, seeds)
    s2 = s2.join(lgbm_predictions[0])
    valid = valid.join(lgbm_predictions[1])
    s2["p_market"] = s2["mkt_p"]
    valid["p_market"] = valid["mkt_p"]
    s2, valid, blend = add_stage2_variants(s2, valid)
    probability_sum_check(s2, [column for column in s2 if column.startswith("p_")])
    probability_sum_check(valid, [column for column in valid if column.startswith("p_")])

    output.mkdir(parents=True, exist_ok=True)
    output_frame = pd.concat(
        [s2.assign(role="s2"), valid.assign(role="valid")], ignore_index=True
    )
    keep = ["race_id", "horse_id", "race_date", "role", "is_winner", "p_market", "p_cl", "p_lgbm"]
    keep += [column for column in output_frame if column.startswith("p_") and column not in keep]
    output_frame.loc[:, keep].to_parquet(output / "predictions.parquet", index=False)
    (output / "cl_selected_features.json").write_text(json.dumps(cl["selected"], indent=2) + "\n")
    cl["trace"].to_csv(output / "cl_selection_trace.csv", index=False)
    cl["coefficients"].to_csv(output / "cl_coefficients.csv", index=False)
    (output / "lgbm_meta.json").write_text(json.dumps(lgbm, indent=2, default=str) + "\n")
    (output / "blend.json").write_text(json.dumps(blend, indent=2) + "\n")
    meta = {
        **provenance, "fold": fold.to_dict(), "config": asdict(config), "config_hash": config.fingerprint(),
        "race_counts": {"s1": int(s1.race_id.nunique()), "s2": int(s2.race_id.nunique()),
                        "valid": int(valid.race_id.nunique()), "s2_market_dropped": s2_dropped,
                        "valid_market_dropped": valid_dropped, "inner_tune": inner_tune_races},
        "runtime_seconds": time.monotonic() - started,
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return {"fold_id": fold.fold_id, **meta["race_counts"], "selected_features": len(cl["selected"]),
            "best_iter": lgbm["best_iter"], "blend_lambda": blend["lambda"],
            "fallback": blend["fallback"], "runtime_seconds": meta["runtime_seconds"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--config", default="primary")
    parser.add_argument("--data", type=Path, default=Path("data/clean data/runners_merged.parquet"))
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("artifacts/folds"))
    parser.add_argument("--inner-tune-months", type=int, default=4)
    parser.add_argument("--min-gain", type=float, default=0.001)
    parser.add_argument("--fold-ids", nargs="*")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    folds = load_folds(args.folds)
    if args.fold_ids:
        folds = folds.loc[folds.fold_id.isin(args.fold_ids)]
    else:
        folds = folds.iloc[:2]
    data_hash = sha256_file(args.data)
    cache = args.out / ".cache" / f"{registry_hash()}-{data_hash}.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        data = pd.read_parquet(cache)
    else:
        data = build_features(pd.read_parquet(args.data))
        data.to_parquet(cache, index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    provenance = {
        "git_commit": commit, "registry_version": REGISTRY_VERSION, "registry_hash": registry_hash(),
        "config_name": args.config, "data_hash": data_hash,
        "packages": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
        "primary": get_config(args.config).primary,
    }
    seeds = [17 + 101 * item for item in range(args.seeds)]
    reports = []
    for _, fold in folds.iterrows():
        target = args.out / args.config / fold.fold_id
        if args.resume and (target / "meta.json").exists():
            prior = json.loads((target / "meta.json").read_text())
            if all(prior.get(key) == provenance[key] for key in ("config_name", "registry_hash", "data_hash")):
                print(f"Skipping {fold.fold_id}: matching completed output")
                continue
        reports.append(run_fold(
            fold, data, args.config, seeds, args.inner_tune_months, args.min_gain, target, provenance
        ))
    if reports:
        print(pd.DataFrame(reports).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
