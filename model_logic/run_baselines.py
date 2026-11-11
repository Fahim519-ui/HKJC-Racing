"""Fit frozen odds-free baseline rungs once for each chronological fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.provenance import require_evidence_run, smoke_record
from model_logic.baselines import (
    BASELINE_FEATURES,
    DRAW_ONLY_FEATURES,
    ORIGINAL_CL_BASE,
    ORIGINAL_CL_CANDIDATES,
    fit_predict_draw_only,
    fit_predict_original_cl,
    validate_baseline_features,
)
from model_logic.feature_builder import build_features
from model_logic.feature_registry import REGISTRY_VERSION, registry_hash
from model_logic.run_folds import REPO
from model_logic.run_folds import (
    inner_split,
    load_folds,
    probability_sum_check,
    select_period,
    sha256_file,
    valid_races,
    validate_periods,
)


DRAW_DEVIATION_NOTE = (
    "Current rc_draw_pct ranks draw values among actual starters; the historical definition "
    "was (draw - 1) / (field_size - 1)."
)


def baseline_feature_hash() -> str:
    encoded = json.dumps(
        {
            "draw_only": DRAW_ONLY_FEATURES,
            "original_cl_base": ORIGINAL_CL_BASE,
            "original_cl_candidates": ORIGINAL_CL_CANDIDATES,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def dirty_worktree() -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    return bool(status.stdout.strip())


def draw_deviation(frame: pd.DataFrame) -> dict[str, float | int]:
    historical = (frame["draw"].astype(float) - 1) / (frame["n_runners"].astype(float) - 1)
    differs = ~np.isclose(frame["rc_draw_pct"].to_numpy(float), historical.to_numpy(float), equal_nan=True)
    count = int(differs.sum())
    runners = int(len(frame))
    return {
        "runners_different": count,
        "runner_count": runners,
        "fraction_different": count / runners if runners else 0.0,
    }


def load_featured_data(data_path: Path, output: Path, data_hash: str) -> pd.DataFrame:
    cache = output / ".cache" / f"baselines-{registry_hash()}-{data_hash}.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_parquet(cache)
    raw = pd.read_parquet(data_path)
    market_columns = [
        column for column in raw
        if column in {"market_prob", "win_odds", "implied_prob"}
        or column.startswith(("mkt_", "market_", "odds_"))
    ]
    data = build_features(raw.drop(columns=market_columns), include_market=False)
    data.to_parquet(cache, index=False)
    return data


def run_baseline_fold(
    fold: pd.Series,
    data: pd.DataFrame,
    output: Path,
    inner_tune_months: int,
    minimum_gain: float,
    provenance: dict,
) -> dict:
    started = time.monotonic()
    s1 = valid_races(select_period(data, fold.s1_start, fold.s1_end))
    s2 = valid_races(select_period(data, fold.s2_start, fold.s2_end))
    valid = valid_races(select_period(data, fold.valid_start, fold.valid_end))
    validate_periods({"s1": s1, "s2": s2, "valid": valid})
    inner_train, inner_tune = inner_split(s1, inner_tune_months)
    if inner_train.empty or inner_tune.empty:
        raise ValueError(f"{fold.fold_id}: inner train/tune split is empty")

    draw_only, draw_predictions = fit_predict_draw_only(inner_train, inner_tune, s1, s2, valid)
    original_cl, original_predictions = fit_predict_original_cl(
        inner_train, inner_tune, s1, s2, valid, minimum_gain
    )
    s2 = s2.join(draw_predictions[0]).join(original_predictions[0])
    valid = valid.join(draw_predictions[1]).join(original_predictions[1])
    for frame in (s2, valid):
        frame["p_uniform"] = 1.0 / frame["n_runners"]
    probability_sum_check(s2, ["p_uniform", "p_draw_only", "p_original_cl"])
    probability_sum_check(valid, ["p_uniform", "p_draw_only", "p_original_cl"])

    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(
        [s2.assign(role="s2"), valid.assign(role="valid")], ignore_index=True
    )
    keep = [
        "race_id", "horse_id", "role", "is_winner",
        "p_uniform", "p_draw_only", "p_original_cl",
    ]
    predictions.loc[:, keep].to_parquet(output / "baseline_predictions.parquet", index=False)
    (output / "original_cl_selected_features.json").write_text(
        json.dumps(original_cl["selected"], indent=2) + "\n"
    )
    original_cl["trace"].to_csv(output / "original_cl_selection_trace.csv", index=False)
    draw_only["coefficients"].to_csv(output / "draw_only_coefficients.csv", index=False)
    original_cl["coefficients"].to_csv(output / "original_cl_coefficients.csv", index=False)
    periods = {"s1": s1, "s2": s2, "valid": valid}
    meta = {
        **provenance,
        "fold": fold.to_dict(),
        "draw_definition_deviation": {
            "note": DRAW_DEVIATION_NOTE,
            "per_period": {name: draw_deviation(frame) for name, frame in periods.items()},
        },
        "race_counts": {name: int(frame["race_id"].nunique()) for name, frame in periods.items()},
        "inner_tune_races": int(inner_tune["race_id"].nunique()),
        "runtime_seconds": time.monotonic() - started,
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return {
        "fold_id": fold.fold_id,
        **meta["race_counts"],
        "selected_features": len(original_cl["selected"]),
        "runtime_seconds": meta["runtime_seconds"],
    }


def run_baselines(
    folds_path: Path,
    data_path: Path = Path("data/clean data/runners_merged.parquet"),
    output: Path = Path("artifacts/folds"),
    inner_tune_months: int = 4,
    minimum_gain: float = 0.001,
    fold_ids: list[str] | None = None,
    resume: bool = False,
    run_record: dict | None = None,
) -> list[dict]:
    """run_record: provenance from the command-line run-gate. Direct calls without it record
    evidence=false."""
    validate_baseline_features()
    folds = load_folds(folds_path)
    if fold_ids:
        folds = folds.loc[folds.fold_id.isin(fold_ids)]
    data_hash = sha256_file(data_path)
    data = load_featured_data(data_path, output, data_hash)
    if run_record is None:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = dirty_worktree()
        run_record = {"git_commit": commit, "git_dirty": dirty, "evidence": False, "gate": "none"}
    dirty = bool(run_record["git_dirty"])
    if dirty:
        print("WARNING: worktree is dirty; baseline metadata records this run as dirty.", file=sys.stderr)
    provenance = {
        "git_commit": run_record["git_commit"],
        "dirty_worktree": dirty,
        "git_dirty": dirty,
        "evidence": run_record["evidence"],
        "gate": run_record.get("gate"),
        "registry_version": REGISTRY_VERSION,
        "registry_hash": registry_hash(),
        "data_hash": data_hash,
        "baseline_feature_hash": baseline_feature_hash(),
        "packages": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    reports = []
    for _, fold in folds.iterrows():
        target = output / "_baselines" / fold.fold_id
        if resume and (target / "meta.json").exists():
            prior = json.loads((target / "meta.json").read_text())
            keys = ("registry_hash", "data_hash", "baseline_feature_hash", "git_commit")
            if all(prior.get(key) == provenance[key] for key in keys):
                print(f"Skipping baselines/{fold.fold_id}: matching completed output")
                continue
        reports.append(
            run_baseline_fold(
                fold, data, target, inner_tune_months, minimum_gain, provenance
            )
        )
    return reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/clean data/runners_merged.parquet"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/folds"))
    parser.add_argument("--inner-tune-months", type=int, default=4)
    parser.add_argument("--min-gain", type=float, default=0.001)
    parser.add_argument("--fold-ids", nargs="*")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="non-evidence run: skips the run-gate, records evidence=false, "
                             "and may not write to artifacts/folds")
    args = parser.parse_args()
    run_record = (smoke_record(REPO, args.out) if args.smoke
                  else require_evidence_run(REPO, args.data))
    reports = run_baselines(
        folds_path=args.folds,
        data_path=args.data,
        output=args.out,
        inner_tune_months=args.inner_tune_months,
        minimum_gain=args.min_gain,
        fold_ids=args.fold_ids,
        resume=args.resume,
        run_record=run_record,
    )
    if reports:
        print(pd.DataFrame(reports).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
