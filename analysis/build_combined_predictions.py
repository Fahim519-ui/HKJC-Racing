"""Combine every model's predictions into one file for Week 5 scoring (plan 6.8).

Takes the richest existing predictions artifact (market, conditional logit, LightGBM, and
their Stage-2 joint blend) and adds the odds baselines (p_power, p_shin, p_market_refit) so
every benchmark lives in one place. Keeps both the h_s2 and h_test windows as PREDICTIONS
only - it computes no metric and never scores h_test.

Not yet present, so not in the file: Track B columns (p_b_base / p_b_rating / p_b_running /
p_b_all). They do not exist in any artifact yet; add them here once Track B is built.

The market-only refit beta is fitted on h_s2 only (the Stage-2 fitting window) and applied
to both windows, so no test-window outcome is used. It minimises eval_utils.per_race_log_loss,
so it follows the same dead-heat rule as the scoring (since 6 Nov: -log mean winner p).
race_date is carried so the scoreboard can bootstrap over whole race days.

    python analysis/build_combined_predictions.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import eval_utils as ev  # noqa: E402

KEYS = ["race_id", "horse_id", "split", "is_winner"]
MODEL_COLS = ["p_market", "p_cl", "p_lgbm", "p_joint"]
TRACK_B = ["p_b_base", "p_b_rating", "p_b_running", "p_b_all"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path,
                        default=REPO / "artifacts/lightgbm_independent_probability_v2/joint_predictions.parquet")
    parser.add_argument("--clean", type=Path, default=REPO / "data/clean data/runners_merged.parquet")
    parser.add_argument("--output", type=Path, default=REPO / "artifacts/combined/combined_predictions.parquet")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(args.predictions)
    # win_odds for the odds baselines; race_date so scoring can resample whole race days
    odds = pd.read_parquet(args.clean)[["race_id", "horse_id", "race_date", "win_odds"]]
    if "race_date" in preds.columns:
        odds = odds.drop(columns="race_date")
    df = preds.merge(odds, on=["race_id", "horse_id"], how="left", validate="one_to_one")

    # odds baselines over every window (these use only the current race's odds, no outcome);
    # fit the refit beta on h_s2 only, then apply it to all windows
    h_s2 = df.loc[df["split"].eq("h_s2")].copy()
    beta = ev.fit_market_refit_beta(h_s2["p_market"], h_s2)
    df["p_power"] = ev._per_race_apply(df, "win_odds", ev.power_probs)
    df["p_shin"] = ev._per_race_apply(df, "win_odds", ev.shin_probs)
    df["p_market_refit"] = ev.market_refit_prob(df["p_market"], df, beta)
    df["p_uniform"] = ev.uniform_prob(df)

    present = [c for c in ["p_uniform", "p_market", "p_power", "p_shin", "p_market_refit",
                           *MODEL_COLS] if c in df.columns]
    missing_track_b = [c for c in TRACK_B if c not in df.columns]

    keep = [k for k in KEYS if k in df.columns] + ["race_date", "win_odds", "market_complete"] + \
           [c for c in present if c not in KEYS]
    keep = list(dict.fromkeys(keep))
    out = df[keep].copy()
    out.attrs.clear()
    out.to_parquet(args.output, index=False)

    # sanity: every probability column sums to 1 per race (within each window)
    checks = {}
    for c in present:
        s = out.groupby("race_id")[c].sum()
        checks[c] = bool(np.allclose(s, 1, atol=1e-6)) and int(out[c].isna().sum()) == 0

    report = [
        "# Combined predictions", "",
        f"Source predictions: `{args.predictions.relative_to(REPO)}`.",
        f"Rows: {len(out)} ({out['split'].value_counts().to_dict()}).",
        f"Market-only refit beta (fitted on h_s2): {beta:.4f}.", "",
        "## Columns", "",
        "| Column | Present | Sums to 1 per race, no NaN |", "|---|---|---|",
    ]
    for c in dict.fromkeys(["p_uniform", "p_market", "p_power", "p_shin", "p_market_refit", *MODEL_COLS]):
        report.append(f"| `{c}` | {'yes' if c in present else 'NO'} | "
                      f"{checks.get(c, '-') if c in present else '-'} |")
    for c in TRACK_B:
        report.append(f"| `{c}` | NO (Track B not built yet) | - |")
    report += ["",
               "Track B columns are absent because Track B has not been built. Add them to "
               "this file once it exists. The file holds predictions for h_s2 and h_test; "
               "h_test is for Week 5 one-shot scoring only and is not scored here."]
    (args.output.parent / "combined_predictions_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"wrote {args.output.relative_to(REPO)}  rows={len(out)}  cols={list(out.columns)}")
    print("sums-to-1 / no-NaN:", checks)
    if missing_track_b:
        print("missing (Track B, expected):", missing_track_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
