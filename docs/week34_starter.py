"""Reference copy (restored 7 Nov 2026 from update_2026-10-31_fahim_cli/): the 31 Oct Week 3-4
starter. Kept because eval_utils.py and the evaluation docs cite its conventions (dead-heat rule,
odds baselines). Not part of the current pipeline; model_logic/ is.

Weeks 3-4 starter pipeline (Group 17): features -> baselines -> logit + LightGBM -> Stage 2.

Run from the project folder (SageMaker terminal or laptop venv):
    python week34_starter.py

Reads   data/processed/runners_merged.parquet
Writes  data/features/features_v1.parquet
        data/predictions/predictions_v1.parquet   (Stage 2 + test windows; test is NOT scored here)
        models/cl_coefficients.csv, models/lgbm_v1.txt
        reports/week4_models.md

Version: 31 Oct 2026 (rc_weight_x_dist now uses distance centred at 1,400 m; 28 Oct was uncentred).

This is a working starting point with a small feature set. Add features from
docs/WEEK3-4_PLAN.md section 5.3 inside build_features(), then re-run.
Test-window metrics are computed only in Week 5.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import brentq
from statsmodels.discrete.conditional_models import ConditionalLogit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/processed/runners_merged.parquet"
OUT_F = ROOT / "data/features"
OUT_P = ROOT / "data/predictions"
OUT_M = ROOT / "models"
OUT_R = ROOT / "reports"
SEED = 17


# ---------------------------------------------------------------- splits (plan section 3)
def assign_split(d):
    s = pd.Series("warmup", index=d.index)
    k, h, t = d["source"] == "kaggle", d["source"] == "hkjc", d["race_date"]
    s[k & (t >= "2017-11-01") & (t < "2019-08-01")] = "k_train"
    s[k & (t >= "2019-08-01")] = "k_tune"
    s[h & (t >= "2025-11-01") & (t < "2026-04-01")] = "h_s2"
    s[h & (t >= "2026-04-01")] = "h_test"
    return s


# ---------------------------------------------------------------- features (plan section 5)
def smoothed_rate(m, key, target, prior, a=30.0, days="365D"):
    """Rate for jockey/trainer over the window BEFORE race day, smoothed toward the prior."""
    daily = (m.groupby(["source", key, "race_date"], as_index=False)
               .agg(hits=(target, "sum"), n=(target, "size"))
               .sort_values(["source", key, "race_date"]))
    roll = (daily.set_index("race_date").groupby(["source", key])[["hits", "n"]]
                 .rolling(days, closed="left").sum().fillna(0).reset_index())
    roll["v"] = (roll["hits"] + a * prior) / (roll["n"] + a)
    return m.merge(roll[["source", key, "race_date", "v"]], on=["source", key, "race_date"], how="left")["v"].to_numpy()


def build_features(df):
    m = df[df["race_valid"] & df["started"]].copy()
    m = m.sort_values(["source", "race_date", "race_no"]).reset_index(drop=True)
    m["is_winner"] = m["is_winner"].astype(int)
    m["is_top3"] = m["is_top3"].astype(int)
    m["split"] = assign_split(m)

    # race conditions (today, known before the race)
    fs = m["field_size"].astype(float)
    m["rc_draw_pct"] = (m["draw"].astype(float) - 1) / (fs - 1)
    m["rc_draw_x_hv"] = m["rc_draw_pct"] * (m["track_code"] == "HV")
    m["rc_draw_x_sprint"] = m["rc_draw_pct"] * (m["distance_m"].astype(float) <= 1200)
    w = m["carried_weight_lb"].astype(float)
    m["rc_weight_rel"] = w - w.groupby(m["race_id"]).transform("mean")
    # distance centred at 1,400 m (the median): uncentred, this was 0.99 correlated with rc_weight_rel
    m["rc_weight_x_dist"] = m["rc_weight_rel"] * (m["distance_m"].astype(float) - 1400) / 1000
    m["rc_prize_level"] = np.log(m["prize_hkd"].astype(float)
                                 / m.groupby(["source", "season"])["prize_hkd"].transform("median").astype(float))

    # horse form: earlier starts, same source (a horse runs at most once a day)
    m["fin_pct"] = (m["finish_pos"].astype(float) - 1) / (fs - 1)
    g = m.groupby(["source", "horse_id"], sort=False)
    m["hf_days_since"] = np.log1p((m["race_date"] - g["race_date"].shift(1)).dt.days.clip(upper=365))
    m["hf_first_start"] = g.cumcount().eq(0).astype(float)
    m["hf_last_fin_pct"] = g["fin_pct"].shift(1)
    m["hf_avg3_fin_pct"] = g["fin_pct"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    m["hf_win5_s"] = g["is_winner"].transform(lambda s: (s.shift(1).rolling(5, min_periods=1).sum().fillna(0) + 5 / 12)
                                               / (s.shift(1).rolling(5, min_periods=1).count() + 5))
    m["hf_weight_change"] = w - g["carried_weight_lb"].shift(1).astype(float)
    m["hf_class_change"] = np.log(m["prize_hkd"].astype(float) / g["prize_hkd"].shift(1).astype(float))
    for c in ["hf_last_fin_pct", "hf_avg3_fin_pct"]:
        m[c + "_rel"] = m[c] - m.groupby("race_id")[c].transform("mean")

    # jockey / trainer: earlier race DAYS only, last 365 days, smoothed
    m["jt_jockey_win365"] = smoothed_rate(m, "jockey_key", "is_winner", 1 / 12)
    m["jt_trainer_win365"] = smoothed_rate(m, "trainer_key", "is_winner", 1 / 12)
    m["jt_jockey_top3_365"] = smoothed_rate(m, "jockey_key", "is_top3", 3 / 12)
    for c in ["jt_jockey_win365", "jt_trainer_win365"]:
        m[c + "_rel"] = m[c] - m.groupby("race_id")[c].transform("mean")

    m = m.drop(columns=["fin_pct"])
    return m


FEATS_CL = ["rc_draw_pct", "rc_draw_x_hv", "rc_weight_rel", "rc_weight_x_dist", "hf_days_since",
            "hf_first_start", "hf_avg3_fin_pct_rel", "hf_last_fin_pct_rel", "hf_win5_s",
            "hf_class_change", "jt_jockey_win365_rel", "jt_trainer_win365_rel"]
FEATS_LGB = FEATS_CL + ["rc_draw_x_sprint", "rc_prize_level", "hf_weight_change", "jt_jockey_top3_365",
                        "hf_avg3_fin_pct", "jt_jockey_win365", "jt_trainer_win365"]


# ---------------------------------------------------------------- helpers (plan 6.2)
def race_softmax(score, rid):
    s = pd.Series(np.asarray(score, dtype=float), index=rid.index)
    e = np.exp(s - s.groupby(rid).transform("max"))
    return e / e.groupby(rid).transform("sum")


def normalise(p, rid):
    p = pd.Series(np.asarray(p, dtype=float), index=rid.index)
    return p / p.groupby(rid).transform("sum")


def per_race_log_loss(p, d):
    win = d["is_winner"] == 1
    return float(-np.log(p[win].groupby(d.loc[win, "race_id"]).mean()).mean())


def power_probs(odds):
    q = 1.0 / np.clip(np.asarray(odds, dtype=float), 1.01, None)
    k = brentq(lambda k: (q ** k).sum() - 1.0, 0.1, 10.0)
    return q ** k


def shin_probs(odds, tol=1e-12, max_iter=1000):
    q = 1.0 / np.clip(np.asarray(odds, dtype=float), 1.01, None)
    n, B, z = len(q), q.sum(), 0.0
    for _ in range(max_iter):
        z_new = (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / B).sum() - 2) / (n - 2)
        if abs(z_new - z) < tol:
            z = z_new
            break
        z = z_new
    return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / B) - z) / (2 * (1 - z))


def per_race(d, fn):
    out = pd.Series(np.nan, index=d.index)
    for _, idx in d.groupby("race_id").groups.items():
        out.loc[idx] = fn(d.loc[idx, "win_odds"].to_numpy())
    return out


# ---------------------------------------------------------------- Stage 1a: conditional logit
class CL:
    def __init__(self, feats):
        self.feats = feats

    def _X(self, d):
        X = (d[self.feats].astype(float) - self.mu) / self.sd
        return X.fillna(0.0).to_numpy()            # 0 = training mean after standardising

    def fit(self, d):
        self.mu = d[self.feats].astype(float).mean()
        self.sd = d[self.feats].astype(float).std().replace(0, 1)
        self.res = ConditionalLogit(d["is_winner"].astype(float).to_numpy(), self._X(d),
                                    groups=d["race_id"].to_numpy()).fit(disp=0)
        return self

    def predict(self, d):
        return race_softmax(self._X(d) @ self.res.params, d["race_id"])

    def table(self):
        return pd.DataFrame({"coef": self.res.params, "se": self.res.bse}, index=self.feats).round(4)


# ---------------------------------------------------------------- Stage 1b: LightGBM
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                  random_state=SEED, deterministic=True, force_row_wise=True, verbose=-1)


def fit_lgb(train, tune):
    probe = lgb.LGBMClassifier(n_estimators=2000, **LGB_PARAMS)
    probe.fit(train[FEATS_LGB].astype(float), train["is_winner"],
              eval_set=[(tune[FEATS_LGB].astype(float), tune["is_winner"])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    n_best = probe.best_iteration_ or 300
    p_tune = normalise(probe.predict_proba(tune[FEATS_LGB].astype(float))[:, 1], tune["race_id"])
    return n_best, p_tune


# ---------------------------------------------------------------- Stage 2 (plan 6.6)
def stage2(s2, p_model_s2, p_mkt_s2):
    Z = np.column_stack([np.log(p_model_s2), np.log(p_mkt_s2)]).astype(float)
    return ConditionalLogit(s2["is_winner"].astype(float).to_numpy(), Z,
                            groups=s2["race_id"].to_numpy()).fit(disp=0)


def main():
    for p in (OUT_F, OUT_P, OUT_M, OUT_R):
        p.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    m = build_features(df)
    keep = ["source", "race_id", "race_date", "season", "horse_id", "split", "is_winner",
            "field_size", "win_odds", "market_prob"] + sorted(set(FEATS_LGB))
    m[keep].to_parquet(OUT_F / "features_v1.parquet", index=False)
    print(f"features: {len(m):,} rows, {len(set(FEATS_LGB))} features -> {OUT_F / 'features_v1.parquet'}")
    print(m["split"].value_counts().to_string())

    tr, tu = m[m.split == "k_train"], m[m.split == "k_tune"]
    full = m[m.split.isin(["k_train", "k_tune"])]
    s2, te = m[m.split == "h_s2"].copy(), m[m.split == "h_test"].copy()
    rep = ["# Week 4 models (starter run)\n"]

    # Stage 1a: choose on k_train -> k_tune, then refit on all Kaggle
    cl_tune = CL(FEATS_CL).fit(tr)
    ll_cl_tune = per_race_log_loss(cl_tune.predict(tu), tu)
    cl = CL(FEATS_CL).fit(full)

    # Stage 1b
    n_best, p_lgb_tune = fit_lgb(tr, tu)
    ll_lgb_tune = per_race_log_loss(p_lgb_tune, tu)
    lg = lgb.LGBMClassifier(n_estimators=n_best, **LGB_PARAMS).fit(full[FEATS_LGB].astype(float), full["is_winner"])
    lg.booster_.save_model(str(OUT_M / "lgbm_v1.txt"))
    cl.table().to_csv(OUT_M / "cl_coefficients.csv")

    preds = []
    for d in (s2, te):
        out = d[["race_id", "horse_id", "split", "is_winner"]].copy()
        out["p_uniform"] = 1.0 / d.groupby("race_id")["race_id"].transform("size")
        out["p_market"] = d["market_prob"].astype(float)
        out["p_power"] = normalise(per_race(d, power_probs), d["race_id"])
        out["p_shin"] = normalise(per_race(d, shin_probs), d["race_id"])
        out["p_cl"] = cl.predict(d)
        out["p_lgbm"] = normalise(lg.predict_proba(d[FEATS_LGB].astype(float))[:, 1], d["race_id"])
        preds.append(out)
    ps2, pte = preds

    # Stage 2 + market-only refit, fitted on h_s2 only
    mo = ConditionalLogit(s2["is_winner"].astype(float).to_numpy(), np.log(ps2[["p_market"]].to_numpy()),
                          groups=s2["race_id"].to_numpy()).fit(disp=0)
    rows = []
    for name in ["p_cl", "p_lgbm"]:
        st = stage2(s2, ps2[name], ps2["p_market"])
        a, b = st.params
        rows.append({"stage1": name, "alpha": a, "alpha_se": st.bse[0], "beta": b, "beta_se": st.bse[1]})
        for out in (ps2, pte):
            out["p_s2_" + name[2:]] = race_softmax(np.column_stack([np.log(out[name]), np.log(out["p_market"])]) @ st.params,
                                                   out["race_id"])
    for out in (ps2, pte):
        out["p_market_refit"] = race_softmax(np.log(out[["p_market"]].to_numpy()) @ mo.params, out["race_id"])

    pred = pd.concat([ps2, pte], ignore_index=True)
    pred.to_parquet(OUT_P / "predictions_v1.parquet", index=False)

    # report: tuning + Stage 2 window only (test stays unscored until Week 5)
    s2_ll = {c: per_race_log_loss(ps2[c], s2) for c in ps2.columns if c.startswith("p_")}
    rep += ["## Stage 1 on the Kaggle tuning window (2019/20)\n",
            f"- conditional logit per-race log loss: {ll_cl_tune:.4f}",
            f"- LightGBM per-race log loss: {ll_lgb_tune:.4f} (trees: {n_best})",
            f"- market on the same races: {per_race_log_loss(tu['market_prob'].astype(float), tu):.4f}\n",
            "## Conditional logit coefficients (standardised, all Kaggle)\n",
            "```\n" + cl.table().to_string() + "\n```", "\n",
            "## Stage 2 on HKJC Nov 2025 - Mar 2026\n",
            "```\n" + pd.DataFrame(rows).round(4).to_string(index=False) + "\n```",
            f"\nMarket-only refit beta: {mo.params[0]:.4f} (se {mo.bse[0]:.4f})\n",
            "## Per-race log loss on the Stage 2 window (in-sample for Stage 2 / refit)\n",
            "```\n" + pd.Series(s2_ll).round(4).to_frame("log_loss").to_string() + "\n```", "\n",
            "Test window (HKJC Apr - Oct 2026): predictions saved, **not scored** (Week 5)."]
    (OUT_R / "week4_models.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"\npredictions: {len(pred):,} rows -> {OUT_P / 'predictions_v1.parquet'}")


if __name__ == "__main__":
    main()
