"""History depth (starts per horse) and absolute missingness by source and window.

Task (status sheet of 8 Nov, Tier 1, Nayeon): starts-per-horse by source for career, track,
distance band and track x distance, plus absolute missingness by source and window, handed to
Inns for the stability report.

Reads only pre-race information: start counts come from earlier race dates, and missingness
is whether a feature value exists. No outcome of the current race, no metric, no model.
Windows are the stability audit's own (analysis/feature_stability_audit.windows) plus the
fold periods in model_logic/validation/folds.csv. h_test appears for information only, as in
the stability audit; nothing here may be used to choose or drop a feature on h_test.

Outputs (analysis/history_depth_results/):
  history_depth_by_window.csv   earlier-date starts at race time, per runner row:
                                source x window x level -> n, mean, quantiles, share 0, share < K
  starts_per_horse.csv          per horse, total starts and distinct tracks/bands/combos in each source
  missingness_by_window.csv     every model input x source x window -> n, n_missing, missing_pct
  history_depth_report.md       the tables Inns needs, with notes
Usage:
    python analysis/history_depth.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.v3_build import DEFAULT_DATA, load_raw, sha256_file
from model_logic.feature_builder import build_features
from model_logic.feature_registry import REGISTRY_VERSION, get_features, get_spec, registry_hash
from model_logic.lightgbm_independent_probability_v2 import add_context_features
from model_logic.track_distance_features import K, prior_day_sums

OUT = REPO / "analysis" / "history_depth_results"
LEVELS = {
    "career": "td_career_starts",
    "track": "td_track_starts",
    "distance_band": "td_distance_band_starts",
    "track_distance": "td_track_distance_starts",
}
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
INFO_ONLY = {"h_test"}


def audit_windows(data: pd.DataFrame) -> dict[str, pd.Series]:
    """The stability audit's windows, then the fold periods (s1 / s2 / valid of k1, k2, h1)."""
    t = data["race_date"]
    windows = {
        "k_train": data["split"].eq("k_train"),
        "k_early": data["source"].eq("kaggle") & (t >= "2017-11-01") & (t < "2018-04-01"),
        "k_tune": data["split"].eq("k_tune"),
        "h_s2": data["split"].eq("h_s2"),
        "h_test": data["split"].eq("h_test"),
    }
    folds = pd.read_csv(REPO / "model_logic" / "validation" / "folds.csv")
    for fold in folds.itertuples():
        for period in ("s1", "s2", "valid"):
            start, end = getattr(fold, f"{period}_start"), getattr(fold, f"{period}_end")
            windows[f"{fold.fold_id}_{period}"] = (t >= start) & (t <= end)
    return windows


def model_inputs(data: pd.DataFrame) -> list[str]:
    """Every column any active model can receive: the v3 CL and LightGBM sets plus the
    LightGBM rank-context columns."""
    cl = get_features("conditional_logit")
    lgbm = add_context_features(data, get_features("lgbm"))
    return list(dict.fromkeys(cl + lgbm))


def depth_table(data: pd.DataFrame, windows: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, mask in windows.items():
        for source in ("kaggle", "hkjc"):
            part = data.loc[mask & data["source"].eq(source)]
            if part.empty:
                continue
            for level, column in LEVELS.items():
                starts = part[column].astype(float)
                weight = K["career" if level == "career" else level]
                row = {"window": name, "source": source, "level": level, "n_rows": len(part),
                       "mean": starts.mean()}
                row.update({f"p{int(q * 100)}": starts.quantile(q) for q in QUANTILES})
                row["share_zero"] = starts.eq(0).mean()
                row[f"share_below_k{int(weight)}"] = starts.lt(weight).mean()
                row["info_only"] = name in INFO_ONLY
                rows.append(row)
    return pd.DataFrame(rows)


def horse_table(data: pd.DataFrame) -> pd.DataFrame:
    """Per horse within each source: total starts and how many distinct contexts it ran in."""
    grouped = data.groupby(["source", "horse_id"])
    per_horse = pd.DataFrame({
        "starts": grouped.size(),
        "tracks": grouped["track_code"].nunique(),
        "distance_bands": grouped["td_distance_band"].nunique(),
        "track_distance": grouped[["track_code", "td_distance_band"]].apply(lambda g: len(g.drop_duplicates())),
    }).reset_index()
    rows = []
    for source, part in per_horse.groupby("source"):
        row = {"source": source, "horses": len(part), "starts_total": int(part["starts"].sum())}
        for column in ("starts", "tracks", "distance_bands", "track_distance"):
            row[f"{column}_mean"] = part[column].mean()
            row[f"{column}_median"] = part[column].median()
        row["starts_p90"] = part["starts"].quantile(0.9)
        row["pct_horses_1_start"] = 100 * part["starts"].eq(1).mean()
        row["pct_horses_ge5_starts"] = 100 * part["starts"].ge(5).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def missing_table(data: pd.DataFrame, windows: dict[str, pd.Series], features: list[str]) -> pd.DataFrame:
    rows = []
    for name, mask in windows.items():
        for source in ("kaggle", "hkjc"):
            part = data.loc[mask & data["source"].eq(source), features]
            if part.empty:
                continue
            missing = part.isna().sum()
            for feature in features:
                rows.append({"feature": feature, "window": name, "source": source, "n": len(part),
                             "n_missing": int(missing[feature]),
                             "missing_pct": 100.0 * missing[feature] / len(part),
                             "info_only": name in INFO_ONLY})
    return pd.DataFrame(rows)


def _md(frame: pd.DataFrame, digits: int = 2) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = []
    for row in frame.itertuples(index=False):
        cells = [f"{value:.{digits}f}" if isinstance(value, float) else str(value) for value in row]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])


def report(depth: pd.DataFrame, horses: pd.DataFrame, missing: pd.DataFrame, meta: dict) -> str:
    main_windows = ["k_train", "k_early", "k_tune", "h_s2", "h_test"]
    depth_view = depth.loc[depth["window"].isin(main_windows)].copy()
    depth_view["window"] = pd.Categorical(depth_view["window"], main_windows, ordered=True)
    depth_view = depth_view.sort_values(["level", "window", "source"])
    depth_view = depth_view[["level", "window", "source", "n_rows", "mean", "p25", "p50", "p75", "share_zero"]]
    depth_view["share_zero"] = 100 * depth_view["share_zero"]
    depth_view = depth_view.rename(columns={"share_zero": "zero_%"})

    pivot = (missing.loc[missing["window"].isin(main_windows)]
             .pivot_table(index="feature", columns="window", values="missing_pct", aggfunc="first")
             .reindex(columns=[w for w in main_windows if w in set(missing["window"])]))
    pivot = pivot.loc[pivot.max(axis=1) > 0].sort_values("h_s2", ascending=False).reset_index()
    pivot.columns.name = None
    complete = sorted(set(missing["feature"]) - set(pivot["feature"]))

    lines = [
        "# History depth and missingness (generated)",
        "",
        f"Generated {meta['generated_utc']} from `{meta['data']}` (SHA-256 `{meta['data_sha256'][:12]}...`), "
        f"registry `{meta['registry_version']}` (hash `{meta['registry_hash'][:12]}...`), commit `{meta['commit'][:7]}`.",
        "",
        "Owner: Nayeon. For Inns's stability report (status sheet of 8 Nov, Tier 1). Only pre-race "
        "information is read: start counts come from earlier race dates, and missingness is whether a "
        "value exists. **h_test is information only** and must not be used to keep or drop a feature.",
        "",
        "## 1. Earlier-date starts at race time (per runner row)",
        "",
        "How much history each runner had when its features were built. `career` is every earlier start "
        "in the source; the other levels count only earlier starts at today's track, distance band, or both. "
        f"With pseudo-count K = {int(K['track'])}, a level with fewer than {int(K['track'])} starts is still "
        "at least half its parent (see `docs/features_v3_track_distance.md`).",
        "",
        _md(depth_view, 1),
        "",
        "Full table with p10/p90, share below K and the fold windows: `history_depth_by_window.csv`.",
        "",
        "## 2. Starts per horse within each source",
        "",
        _md(horses.rename(columns=lambda c: c.replace("_", " ")), 2),
        "",
        "## 3. Absolute missingness of model inputs (% of rows)",
        "",
        f"Inputs with any missing value in the main windows; the other {len(complete)} model inputs are "
        "never missing. Counts and the fold windows: `missingness_by_window.csv`.",
        "",
        _md(pivot, 2),
        "",
        "Never missing: " + ", ".join(f"`{name}`" for name in complete) + ".",
        "",
        "## Notes for the stability report",
        "",
        "- Missing values come from short histories: a first start has no previous finish, and "
        "`tr_fin_pct_slope3_rel` needs two earlier finishes. Their rates follow the zero-start shares in table 1.",
        "- The `td_*_rel` rates are never missing by construction: a runner without history at a level "
        "inherits the parent level (track x distance -> track -> career -> prior-day source rate). Their "
        "shift between sources is therefore driven by history depth, not by blanks.",
        "- HKJC history starts on " + meta["hkjc_first_date"] + " and Kaggle and HKJC histories are never "
        "mixed, so h_s2 runners have much shorter histories than k_train or k_tune at every level, close to "
        "k_early (the history-matched Kaggle window). Compare h_s2 with k_early before reading a Kaggle->HKJC "
        "shift as a change in racing. By h_test, HKJC depth is back near k_train.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    raw_path = args.data.resolve()
    data = build_features(load_raw(raw_path))
    # Career depth uses the same prior-day convention as the td_* levels.
    data["td_career_starts"] = prior_day_sums(data, ["source", "horse_id"], {"n": np.ones(len(data))})["n"]
    windows = audit_windows(data)
    features = model_inputs(data)
    for name in features:
        get_spec(name)  # every model input must be registered (raises KeyError otherwise)

    depth = depth_table(data, windows)
    horses = horse_table(data)
    missing = missing_table(data, windows, features)

    import subprocess
    from datetime import datetime, timezone
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    meta = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data": raw_path.relative_to(REPO).as_posix() if raw_path.is_relative_to(REPO) else str(raw_path),
            "data_sha256": sha256_file(raw_path), "registry_version": REGISTRY_VERSION,
            "registry_hash": registry_hash(), "commit": commit, "windows": list(windows),
            "info_only_windows": sorted(INFO_ONLY), "model_inputs": features,
            "hkjc_first_date": str(data.loc[data["source"].eq("hkjc"), "race_date"].min().date())}

    args.out.mkdir(parents=True, exist_ok=True)
    depth.to_csv(args.out / "history_depth_by_window.csv", index=False, float_format="%.6g")
    horses.to_csv(args.out / "starts_per_horse.csv", index=False, float_format="%.6g")
    missing.to_csv(args.out / "missingness_by_window.csv", index=False, float_format="%.6g")
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (args.out / "history_depth_report.md").write_text(report(depth, horses, missing, meta), encoding="utf-8")
    print(f"Wrote {args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out}: "
          f"{len(depth)} depth rows, {len(horses)} source rows, {len(missing)} missingness rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
