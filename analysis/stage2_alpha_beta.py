"""Stage 2 alpha/beta table: how much weight does the market-anchored blend give each model?

Stage 2 (Benter) scores each runner by  beta * log p_market + alpha * log p_model  and
softmaxes within the race. beta near 1 and alpha near 0 means the model adds nothing the odds
do not already know; a clearly positive alpha means the model carries extra information.

This script refits that blend WITHOUT a penalty (plain statsmodels ConditionalLogit), so every
coefficient has a standard error, on each Stage-2 fitting window we have:
  * h_s2   - HKJC Stage-2 window, Stage-1 predictions from
             artifacts/combined/combined_predictions.parquet (split == "h_s2" only)
  * k1_s2, k2_s2, h1_s2 - the s2 rows of Fahim's fold runner
             (artifacts/folds/primary/<fold>/predictions.parquet, role == "s2")
For each window it fits: market only (beta), market+cl, market+lgbm, market+cl+lgbm, and
reports the likelihood-ratio test of each blend against market only.

Read the numbers with care:
  * Every fit and log loss here is IN-SAMPLE on its fitting window (plan: Stage 2 is fitted on
    h_s2, so its log loss there is in-sample). Out-of-sample evidence is the fold validation.
  * Standard errors treat races as independent. Races on one day are correlated, so the true
    uncertainty is somewhat larger; treat |z| near 2 as weak.
  * The team's production blends use a small ridge penalty (lambda); the penalised weights
    are listed next to the plain fit for reference.
  * h_test is never read: the combined file is filtered to h_s2 and every frame goes through
    eval_utils.check_not_test.

Provenance (7 Nov): fold predictions are read through provenance.load_fold_predictions,
which refuses a run whose meta.json does not record the live registry version and a commit
on origin/main. The 6 Nov fold runs (registry v3.0, unpublished commit fc759fe) fail that
check. --allow-unverified reproduces the table from them anyway and stamps the report
UNVERIFIED with the reasons. The h_s2 rows of the combined file are checked the same way
(analysis/provenance.py check_predictions_file).

    python analysis/stage2_alpha_beta.py                     # refuses stale fold runs
    python analysis/stage2_alpha_beta.py --allow-unverified  # labelled reproduction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.discrete.conditional_models import ConditionalLogit

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import eval_utils as ev  # noqa: E402
import provenance as prov  # noqa: E402

EPSILON = 1e-12
SPECS = {
    "market only": ["p_market"],
    "mkt+cl": ["p_market", "p_cl"],
    "mkt+lgbm": ["p_market", "p_lgbm"],
    "mkt+cl+lgbm": ["p_market", "p_cl", "p_lgbm"],
}
NAMES = {"p_market": "beta (market)", "p_cl": "alpha_cl", "p_lgbm": "alpha_lgbm"}
FOLD_THETA_KEYS = {"market only": "market_calibrated", "mkt+cl": "mkt+cl",
                   "mkt+lgbm": "mkt+lgbm", "mkt+cl+lgbm": "mkt+cl+lgbm"}
JOINT_KEYS = {"mkt+cl": "mkt+cl", "mkt+lgbm": "mkt+lgbm", "mkt+cl+lgbm": "mkt+lgbm+cl"}


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Races where every probability is present for every runner and someone won."""
    cols = ["p_market", "p_cl", "p_lgbm"]
    ok = df[cols].notna().all(axis=1)
    if "market_complete" in df.columns:
        ok &= df["market_complete"].astype(bool)
    ok = ok.groupby(df["race_id"]).transform("all")
    out = df.loc[ok].copy()
    out = out[out.groupby("race_id")["is_winner"].transform("max") > 0]
    return out.reset_index(drop=True)


def fit(df: pd.DataFrame, cols: list[str]):
    X = np.log(df[cols].astype(float).clip(lower=EPSILON)).to_numpy()
    y = df["is_winner"].astype(float).to_numpy()
    res = ConditionalLogit(y, X, groups=df["race_id"].to_numpy()).fit(disp=0)
    score = pd.Series(X @ res.params, index=df.index)
    e = np.exp(score - score.groupby(df["race_id"]).transform("max"))
    p = e / e.groupby(df["race_id"]).transform("sum")
    return res, ev.per_race_log_loss(p, df)


def load_windows(combined: Path, folds_root: Path, allow_unverified: bool = False,
                 checks: "prov.Report | None" = None) -> dict[str, pd.DataFrame]:
    """checks (if given) collects every provenance problem/warning for the report."""
    checks = checks if checks is not None else prov.Report()
    combined_check = prov.check_predictions_file(combined)
    checks.extend(combined_check)
    if not combined_check.ok and not allow_unverified:
        prov.require(combined_check, f"{combined.name} (pass --allow-unverified to reproduce)")
    windows = {}
    h = ev.load_predictions(combined, split="h_s2")          # filters to h_s2 + guard
    windows["h_s2"] = h
    for fold in ["k1", "k2", "h1"]:
        path = folds_root / fold / "predictions.parquet"
        if not path.exists():
            continue
        f, checked = prov.load_fold_predictions(folds_root / fold, role="s2",
                                                allow_unverified=allow_unverified)
        checks.extend(checked)
        ev.check_not_test(f)
        windows[f"{fold}_s2"] = f
    return windows


def penalised_reference(folds_root: Path, joint_csv: Path) -> dict[tuple[str, str], list[float]]:
    """Weights the team's scripts actually used (ridge-penalised), for side-by-side reading."""
    ref = {}
    if joint_csv.exists():
        j = pd.read_csv(joint_csv).set_index("variant")
        for spec, key in JOINT_KEYS.items():
            if key in j.index:
                ref[("h_s2", spec)] = [j.loc[key, c] for c in SPECS[spec]]
    for fold in ["k1", "k2", "h1"]:
        path = folds_root / fold / "blend.json"
        if path.exists():
            theta = json.loads(path.read_text())["theta"]
            for spec, key in FOLD_THETA_KEYS.items():
                if key in theta:
                    ref[(f"{fold}_s2", spec)] = list(theta[key])
    return ref


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, default=REPO / "artifacts/combined/combined_predictions.parquet")
    parser.add_argument("--folds-root", type=Path, default=REPO / "artifacts/folds/primary")
    parser.add_argument("--joint-coefficients", type=Path,
                        default=REPO / "artifacts/lightgbm_independent_probability_v2/joint_coefficients.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO / "analysis/stage2_results")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="read runs that fail the provenance check; the report is stamped UNVERIFIED")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checks = prov.Report()
    windows = load_windows(args.combined, args.folds_root, args.allow_unverified, checks)
    ref = penalised_reference(args.folds_root, args.joint_coefficients)
    coef_rows, fit_rows = [], []
    for wname, raw in windows.items():
        df = usable(raw)
        n_races, n_runners = df["race_id"].nunique(), len(df)
        base_res, base_ll = fit(df, SPECS["market only"])
        for spec, cols in SPECS.items():
            res, ll = fit(df, cols) if spec != "market only" else (base_res, base_ll)
            lr = 2 * (res.llf - base_res.llf)
            dof = len(cols) - 1
            pen = ref.get((wname, spec))
            for i, c in enumerate(cols):
                coef_rows.append({
                    "window": wname, "spec": spec, "term": NAMES[c],
                    "coef": res.params[i], "se": res.bse[i], "z": res.tvalues[i], "p": res.pvalues[i],
                    "penalised_coef": pen[i] if pen is not None and i < len(pen) else np.nan,
                })
            fit_rows.append({
                "window": wname, "spec": spec, "races": n_races, "runners": n_runners,
                "loglik": res.llf, "in_sample_log_loss": ll,
                "gain_vs_market_only": base_ll - ll,
                "lr_stat": lr if dof else np.nan, "lr_df": dof if dof else np.nan,
                "lr_p": chi2.sf(lr, dof) if dof else np.nan,
            })
    coefs, fits = pd.DataFrame(coef_rows), pd.DataFrame(fit_rows)
    coefs.to_csv(args.output_dir / "alpha_beta.csv", index=False)
    fits.to_csv(args.output_dir / "fit_summary.csv", index=False)

    show = coefs.copy()
    for c in ["coef", "se", "z", "penalised_coef"]:
        show[c] = show[c].round(3)
    show["p"] = show["p"].map(lambda v: f"{v:.3g}")
    fshow = fits.copy()
    for c in ["loglik", "lr_stat"]:
        fshow[c] = fshow[c].round(2)
    for c in ["in_sample_log_loss", "gain_vs_market_only"]:
        fshow[c] = fshow[c].round(5)
    fshow["lr_p"] = fshow["lr_p"].map(lambda v: "" if pd.isna(v) else f"{v:.3g}")
    stamp = [] if checks.ok else [
        "**UNVERIFIED.** Built with --allow-unverified from runs that fail the provenance check "
        "(analysis/provenance.py). Do not quote these numbers as current results:", "",
        *[f"- {line}" for line in checks.lines()], ""]
    report = [
        "# Stage 2 alpha/beta table", "", *stamp,
        "Blend: score = beta * log p_market + alpha * log p_model, softmax within race. Plain "
        "(unpenalised) conditional logit, so each weight has a standard error. All fits are "
        "in-sample on their Stage-2 window; out-of-sample evidence is in the fold validation. "
        "SEs treat races as independent (same-day races are correlated, so true SEs are larger). "
        "`penalised_coef` is the ridge-penalised weight the team's scripts used (h_s2: "
        "lightgbm_independent_probability_v2/joint_coefficients.csv; folds: blend.json; for the "
        "folds' 'market only' row it is the calibrated-market beta).", "",
        "Windows: h_s2 = HKJC 2025-11..2026-03 Stage-2 rows of the combined file; k1_s2, k2_s2, "
        "h1_s2 = s2 rows of the fold runner (see model_logic/validation/folds.csv). h1_s2 lies "
        "inside h_s2. h_test is not read.", "",
        "## Coefficients", "", "```", show.to_string(index=False), "```", "",
        "## Fit and likelihood-ratio test against market only", "", "```",
        fshow.to_string(index=False), "```", "",
        "Read: alpha with |z| < 2 is not distinguishable from 0, i.e. the blend does not lean on "
        "the model beyond the odds. A negative alpha means the model, given the odds, points the "
        "wrong way on that window. beta > 1 is the favourite-longshot correction (favourites "
        "win more often than the odds imply).",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(show.to_string(index=False))
    print()
    print(fshow.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
