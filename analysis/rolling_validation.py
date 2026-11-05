"""Chronological rolling validation and ablation for the market-plus-model blend.

Why this exists (WEEK5_PLAN.md Job B; result.md section 3): a single train/tune split, or a
month-CV that fits on later months to score earlier ones, flatters the model. Here every
fold trains only on races BEFORE its validation window, the LightGBM tree count is chosen by
an inner split INSIDE the training window (never on the validation window), and the blend is
fitted on the training window and scored once on the next period. A feature or model is
trusted only if it helps across folds.

What it reports:
  * market, market+logit, market+lgbm, market+both, per fold, race-weighted, plus the mean
    gain over the market-only refit with a paired bootstrap across all pooled fold races.
  * a trainer-feature on/off comparison (drop jt_trainer_* and refit), to settle whether the
    weaker-in-HKJC trainer rate should stay (stability audit flagged it).

Scope: Kaggle only, so it never touches HKJC h_s2/h_test. This answers "does the approach
generalise over time" on the training era; the one-shot HKJC test is scored separately in
Week 5. Run:  python analysis/rolling_validation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.discrete.conditional_models import ConditionalLogit

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "model_logic"), str(REPO / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_utils as ev  # noqa: E402

EPSILON = 1e-12
SEED = 17
# Validation windows over the Kaggle era. Each trains on everything strictly before start.
FOLDS = [
    ("2019-02-01", "2019-08-01"),
    ("2019-08-01", "2020-02-01"),
    ("2020-02-01", "2020-08-01"),
]
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                  random_state=SEED, deterministic=True, force_row_wise=True, verbose=-1)


# --------------------------------------------------------------------------- helpers
def load_builder():
    import conditional_logit_tuning_v2 as cl
    import lightgbm_independent_probability_v2 as lgb2
    return cl, lgb2


def race_softmax(score: pd.Series, race_id: pd.Series) -> pd.Series:
    e = np.exp(score - score.groupby(race_id).transform("max"))
    return e / e.groupby(race_id).transform("sum")


def normalise(p: pd.Series, race_id: pd.Series) -> pd.Series:
    p = pd.Series(np.asarray(p, float), index=race_id.index).clip(lower=EPSILON)
    return p / p.groupby(race_id).transform("sum")


def inner_split(train: pd.DataFrame, frac: float = 0.75):
    """Split the training window by race date: earlier `frac` to fit, the rest to early-stop.
    This keeps model selection strictly inside training (never on the validation window)."""
    days = np.sort(train["race_date"].unique())
    cut = days[int(len(days) * frac)]
    return train[train["race_date"] < cut], train[train["race_date"] >= cut]


def fit_lgbm(train: pd.DataFrame, feats: list[str]) -> lgb.LGBMClassifier:
    inner_tr, inner_va = inner_split(train)
    probe = lgb.LGBMClassifier(n_estimators=3000, **LGB_PARAMS)
    probe.fit(inner_tr[feats].astype(float), inner_tr["is_winner"].astype(int),
              eval_set=[(inner_va[feats].astype(float), inner_va["is_winner"].astype(int))],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    n = int(probe.best_iteration_ or 300)
    return lgb.LGBMClassifier(n_estimators=n, **LGB_PARAMS).fit(
        train[feats].astype(float), train["is_winner"].astype(int)), n


def fit_cl(train: pd.DataFrame, feats: list[str]):
    mu = train[feats].astype(float).mean()
    sd = train[feats].astype(float).std().replace(0, 1)
    X = ((train[feats].astype(float) - mu) / sd).fillna(0.0).to_numpy()
    res = ConditionalLogit(train["is_winner"].astype(float).to_numpy(), X,
                           groups=train["race_id"].to_numpy()).fit(disp=0)
    return (mu, sd, res.params)


def predict_cl(model, df: pd.DataFrame, feats: list[str]) -> pd.Series:
    mu, sd, params = model
    X = ((df[feats].astype(float) - mu) / sd).fillna(0.0).to_numpy()
    return race_softmax(pd.Series(X @ params, index=df.index), df["race_id"])


def fit_blend(train: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Conditional-logit blend on log probabilities; market-anchored prior (first col = 1)."""
    Z = np.log(train[columns].clip(lower=EPSILON).to_numpy(float))
    res = ConditionalLogit(train["is_winner"].astype(float).to_numpy(), Z,
                           groups=train["race_id"].to_numpy()).fit(disp=0)
    return res.params


def blend_prob(df: pd.DataFrame, columns: list[str], theta: np.ndarray) -> pd.Series:
    Z = np.log(df[columns].clip(lower=EPSILON).to_numpy(float))
    return race_softmax(pd.Series(Z @ theta, index=df.index), df["race_id"])


# ------------------------------------------------------------------------- the run
def run(data: pd.DataFrame, feats_cl: list[str], feats_lgb: list[str], label: str) -> dict:
    """One rolling-validation pass. Returns per-fold rows and pooled per-race losses."""
    fold_rows = []
    pooled = {k: [] for k in ["market", "refit", "cl", "lgbm", "mkt_cl", "mkt_lgbm", "mkt_both"]}
    for start, end in FOLDS:
        train = data[data["race_date"] < start]
        valid = data[(data["race_date"] >= start) & (data["race_date"] < end)]
        if len(train) == 0 or len(valid) == 0:
            continue
        v = valid.copy()
        v["p_market"] = normalise(v["mkt_p"], v["race_id"])
        # refit beta fitted on the TRAIN window only, then applied to valid
        tr = train.copy()
        tr["p_market"] = normalise(tr["mkt_p"], tr["race_id"])
        beta = ev.fit_market_refit_beta(tr["p_market"], tr)
        v["p_refit"] = ev.market_refit_prob(v["p_market"], v, beta)

        cl_model = fit_cl(train, feats_cl)
        v["p_cl"] = predict_cl(cl_model, v, feats_cl)
        lgb_model, n_trees = fit_lgbm(train, feats_lgb)
        v["p_lgbm"] = normalise(lgb_model.predict_proba(v[feats_lgb].astype(float))[:, 1], v["race_id"])

        # Blend weights must be fitted on OUT-OF-FOLD Stage-1 predictions: a model scoring
        # its own training races looks over-informative, and the blend then over-weights it
        # and does worse out of sample. So fit Stage-1 on the inner-train part only, predict
        # the held-out inner part, and fit the blend there.
        inner_tr, inner_va = inner_split(train)
        cl_inner = fit_cl(inner_tr, feats_cl)
        lgb_inner, _ = fit_lgbm(inner_tr, feats_lgb)
        oof = inner_va.copy()
        oof["p_market"] = normalise(oof["mkt_p"], oof["race_id"])
        oof["p_cl"] = predict_cl(cl_inner, oof, feats_cl)
        oof["p_lgbm"] = normalise(lgb_inner.predict_proba(oof[feats_lgb].astype(float))[:, 1], oof["race_id"])
        th_cl = fit_blend(oof, ["p_market", "p_cl"])
        th_lgb = fit_blend(oof, ["p_market", "p_lgbm"])
        th_both = fit_blend(oof, ["p_market", "p_cl", "p_lgbm"])
        v["p_mkt_cl"] = blend_prob(v, ["p_market", "p_cl"], th_cl)
        v["p_mkt_lgbm"] = blend_prob(v, ["p_market", "p_lgbm"], th_lgb)
        v["p_mkt_both"] = blend_prob(v, ["p_market", "p_cl", "p_lgbm"], th_both)

        names = {"market": "p_market", "refit": "p_refit", "cl": "p_cl", "lgbm": "p_lgbm",
                 "mkt_cl": "p_mkt_cl", "mkt_lgbm": "p_mkt_lgbm", "mkt_both": "p_mkt_both"}
        ll = {k: ev.per_race_log_loss(v[c], v) for k, c in names.items()}
        for k, c in names.items():
            pooled[k].append(ev.race_losses(v[c], v))
        fold_rows.append({"fold": f"{start}..{end}", "races": v["race_id"].nunique(),
                          "trees": n_trees, "beta": round(beta, 3), **{k: round(x, 4) for k, x in ll.items()}})
    return {"label": label, "folds": pd.DataFrame(fold_rows),
            "pooled": {k: pd.concat(v) for k, v in pooled.items() if v}}


def pooled_summary(result: dict) -> pd.DataFrame:
    pooled = result["pooled"]
    base = pooled["refit"]
    rows = []
    for k, loss in pooled.items():
        mean, lo, hi = ev.paired_bootstrap(base, loss)   # >0: k beats the refit
        rows.append({"model": k, "pooled_log_loss": round(loss.mean(), 4),
                     "gain_vs_refit": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "beats_refit": bool(lo > 0)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=REPO / "data/clean data/runners_merged.parquet")
    parser.add_argument("--output-dir", type=Path, default=REPO / "analysis/rolling_validation_results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cl, lgb2 = load_builder()
    data = cl.build_features(pd.read_parquet(args.data))
    data = data[data["source"].eq("kaggle") & data["mkt_ok"]].copy()
    data = data[data.groupby("race_id")["is_winner"].transform("max") > 0].copy()
    feats_cl = list(cl.ALL_FEATURES)
    feats_lgb = lgb2.add_context_features(data, list(cl.ALL_FEATURES))

    full = run(data, feats_cl, feats_lgb, "all features")
    no_tr = [f for f in feats_cl if not f.startswith("jt_trainer")]
    no_tr_lgb = [f for f in feats_lgb if not f.startswith("jt_trainer")]
    ablate = run(data, no_tr, no_tr_lgb, "no trainer features")

    full_folds, full_sum = full["folds"], pooled_summary(full)
    abl_sum = pooled_summary(ablate)
    full_folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    full_sum.to_csv(args.output_dir / "pooled_summary.csv", index=False)
    abl_sum.to_csv(args.output_dir / "ablation_no_trainer.csv", index=False)

    report = [
        "# Rolling validation (Kaggle era, chronological)", "",
        "Each fold trains on races strictly before its window; the LightGBM tree count is chosen "
        "by an inner split inside training only; the blend weights are fitted on training and "
        "scored once on the next period. Gain is against the market-only refit (favourite-longshot "
        "corrected), with a paired race-level bootstrap over pooled fold races. Kaggle only; HKJC "
        "is never touched here.", "",
        "## Per-fold log loss", "", "```", full_folds.to_string(index=False), "```", "",
        "## Pooled, all features", "", "```", full_sum.to_string(index=False), "```", "",
        "## Pooled, trainer features removed", "", "```", abl_sum.to_string(index=False), "```", "",
        "Read: a model 'beats' the refit only if its whole 95% CI is above 0. In-era gains here "
        "do not prove an HKJC edge; the one-shot HKJC test is scored separately in Week 5.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(full_sum.to_string(index=False))
    print(f"\nOutputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
