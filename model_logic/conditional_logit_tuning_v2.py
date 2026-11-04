"""Leakage-safe conditional logit (McFadden) with analytic gradients, tuned on the
fixed Kaggle validation window.

Design notes
------------
* Choice set = race. All softmax / log-sum-exp / group sums are segment operations
  (np.*.reduceat) over rows sorted contiguously by race, so no statistic can cross races.
  The choice-set size J_r is recomputed from actual starters (scratchings handled).
* Likelihood: fractional-target cross-entropy. A k-way dead heat for position s puts
  weight 1/k on each tied runner. Optional Plackett-Luce / Harville "exploded" stages
  (win, 2nd, 3rd ...) give a multi-market objective from one utility.
* Regularisation (all quadratic forms except L1, so everything is closed-form):
    lambda * [ a*||b||_1 + (1-a) * b' S b ]   S = pooled WITHIN-RACE utility covariance
    lambda_mkt * b' M b                       M = mean_r a_r a_r',  a_r = Cov_r(x, market logit)
  b'Sb is the mean within-race variance of the utility (race-level ridge: shrinks the
  dispersion of race utilities, not raw coefficients). b'Mb is the mean squared
  within-race covariance between model utility and market logit, i.e. an orthogonality
  penalty that pushes the utility toward information the market has not priced. Odds enter
  ONLY this training-time regulariser; the fitted utility never takes odds as an input.
* Exact L1 through the b = b+ - b- split under L-BFGS-B bounds.
* Log-sum-exp is max-shifted per race and per stage; fully masked sets are guarded.
* Tune-window diagnostics include a 2-parameter Benter blend of (market logit, model
  utility): the incremental log-likelihood over the market is the alpha test.
* The Apr-Oct 2026 HKJC window is predicted but never scored.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from model_logic.jt_partnership_features import add_lagged_jockey_trainer_features
except ModuleNotFoundError:
    from jt_partnership_features import add_lagged_jockey_trainer_features

BASE_FEATURES = [
    "jt_jockey_win365_rel",
    "hf_avg3_fin_pct_rel",
    "rc_draw_pct",
    "rc_draw_x_hv",
    "rc_weight_rel",
    "hf_days_since",
]
CANDIDATE_FEATURES = [
    "hf_last_fin_pct_rel",
    "jt_trainer_win365_rel",
    "hf_win5_s",
    "hf_first_start",
    "rc_draw_x_sprint",
    "rc_weight_x_dist",
    "hf_class_change",
    "w3_horse_past_win_rate",
    "w3_horse_past_top3_rate",
    "jtp_pair_effect365_rel",
    "jtp_pair_effect_last_50_rel",
]
ALL_FEATURES = BASE_FEATURES + CANDIDATE_FEATURES
SPEARMAN_LIMIT = 0.9
MIN_PROB = 1e-12


# --------------------------------------------------------------------------- features
def assign_split(data: pd.DataFrame) -> pd.Series:
    split = pd.Series("warmup", index=data.index)
    kaggle = data["source"].eq("kaggle")
    hkjc = data["source"].eq("hkjc")
    date = data["race_date"]
    split[kaggle & (date >= "2017-11-01") & (date < "2019-08-01")] = "k_train"
    split[kaggle & (date >= "2019-08-01")] = "k_tune"
    split[hkjc & (date >= "2025-11-01") & (date < "2026-04-01")] = "h_s2"
    split[hkjc & (date >= "2026-04-01")] = "h_test"
    return split


def rolling_rate(
    data: pd.DataFrame, key: str, target: str, prior: float, weight: float = 30.0,
    window: int = 365,
) -> np.ndarray:
    """Smoothed hit rate over [d-365, d) per (source, key); same-day rides excluded.

    Vectorised: one global cumsum + searchsorted. Keys are packed as gid*BIG + day so a
    window can never reach into another group (BIG > max_day + window).
    """
    gid = data.groupby(["source", key], sort=False, dropna=False).ngroup().to_numpy(np.int64)
    day = (data["race_date"] - pd.Timestamp("1990-01-01")).dt.days.to_numpy(np.int64)
    big = np.int64(1 << 20)
    row_t = gid * big + day
    uniq_t, inv = np.unique(row_t, return_inverse=True)
    hits = np.bincount(inv, weights=data[target].to_numpy(float))
    n = np.bincount(inv).astype(float)
    c_hits = np.concatenate([[0.0], np.cumsum(hits)])
    c_n = np.concatenate([[0.0], np.cumsum(n)])
    lo = np.searchsorted(uniq_t, uniq_t - window, side="left")
    hi = np.arange(len(uniq_t))  # first index with t >= d is itself -> excludes today
    value = ((c_hits[hi] - c_hits[lo]) + weight * prior) / ((c_n[hi] - c_n[lo]) + weight)
    return value[inv]


def _nanmean_rows(a: np.ndarray) -> np.ndarray:
    cnt = (~np.isnan(a)).sum(1)
    return np.where(cnt > 0, np.nansum(a, 1) / np.maximum(cnt, 1), np.nan)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.loc[raw["race_valid"].astype(bool) & raw["started"].astype(bool)].copy()
    d["race_date"] = pd.to_datetime(d["race_date"])
    d = d.sort_values(["source", "race_date", "race_no"], kind="stable").reset_index(drop=True)
    d["is_winner"] = d["is_winner"].astype(float)
    d["is_top3"] = d["is_top3"].astype(float)

    # Choice-set size from ACTUAL starters (robust to late scratchings / stale field_size).
    d["n_runners"] = d.groupby("race_id")["race_id"].transform("size").astype(float)
    d = d.loc[d["n_runners"] >= 2].reset_index(drop=True)
    d["split"] = assign_split(d)
    n = d["n_runners"]
    race = d.groupby("race_id")

    # Draw rank among starters (a scratched stall no longer leaves a gap / pct > 1).
    draw_rank = race["draw"].rank(method="first")
    d["rc_draw_pct"] = (draw_rank - 1) / (n - 1)
    is_hv = d["track_code"].eq("HV").astype(float)
    is_sprint = (d["distance_m"].astype(float) <= 1200).astype(float)
    d["rc_draw_x_hv"] = d["rc_draw_pct"] * is_hv
    d["rc_draw_x_sprint"] = d["rc_draw_pct"] * is_sprint
    w = d["carried_weight_lb"].astype(float)
    d["rc_weight_rel"] = w - w.groupby(d["race_id"]).transform("mean")
    d["rc_weight_x_dist"] = d["rc_weight_rel"] * (d["distance_m"].astype(float) - 1400) / 1000

    fp = d["finish_pos"].astype(float).where(lambda s: s > 0)
    d["finish_percentile"] = (fp - 1) / (n - 1)

    horse = d.groupby(["source", "horse_id"], sort=False)

    def lags(col: str, k: int) -> np.ndarray:
        return np.column_stack([horse[col].shift(i).to_numpy(float) for i in range(1, k + 1)])

    d["hf_days_since"] = np.log1p(
        (d["race_date"] - horse["race_date"].shift(1)).dt.days.clip(upper=365)
    )
    d["hf_first_start"] = horse.cumcount().eq(0).astype(float)
    fin_lags = lags("finish_percentile", 3)
    d["hf_last_fin_pct"] = fin_lags[:, 0]
    d["hf_avg3_fin_pct"] = _nanmean_rows(fin_lags)
    prior = 1 / 12
    win5 = lags("is_winner", 5)
    d["hf_win5_s"] = (np.nansum(win5, 1) + 5 * prior) / ((~np.isnan(win5)).sum(1) + 5)
    d["w3_horse_past_win_rate"] = _nanmean_rows(win5[:, :3])
    d["w3_horse_past_top3_rate"] = _nanmean_rows(lags("is_top3", 3))
    ratio = d["prize_hkd"].astype(float) / horse["prize_hkd"].shift(1).astype(float)
    d["hf_class_change"] = np.log(ratio.where((ratio > 0) & np.isfinite(ratio)))

    def rel(col: str) -> pd.Series:
        return d[col] - d.groupby("race_id")[col].transform("mean")

    d["hf_last_fin_pct_rel"] = rel("hf_last_fin_pct")
    d["hf_avg3_fin_pct_rel"] = rel("hf_avg3_fin_pct")
    d["jt_jockey_win365"] = rolling_rate(d, "jockey_key", "is_winner", prior)
    d["jt_trainer_win365"] = rolling_rate(d, "trainer_key", "is_winner", prior)
    d["jt_jockey_win365_rel"] = rel("jt_jockey_win365")
    d["jt_trainer_win365_rel"] = rel("jt_trainer_win365")
    d = add_lagged_jockey_trainer_features(d)

    # Market: overround removed per race over actual starters; races with any missing /
    # non-positive price fall back to uniform and are flagged (excluded from comparisons).
    mp = d["market_prob"].astype(float)
    ok = (mp > 0).groupby(d["race_id"]).transform("all")
    mp_norm = mp.where(mp > 0) / mp.where(mp > 0).groupby(d["race_id"]).transform("sum")
    d["mkt_ok"] = ok.astype(bool)
    d["mkt_p"] = np.where(ok, mp_norm, 1.0 / n)
    d["mkt_logit"] = np.log(np.clip(d["mkt_p"].to_numpy(float), 1e-6, None))

    # Contiguous races are required by every reduceat below.
    return d.drop(columns=["finish_percentile"]).sort_values(
        "race_id", kind="stable"
    ).reset_index(drop=True)


# ------------------------------------------------------------------ choice-set engine
@dataclass
class Stage:
    mask: np.ndarray | None  # remaining choice set (None = everyone)
    y: np.ndarray            # fractional target (dead heat -> 1/k each)
    valid_r: np.ndarray      # race-level: stage has a target and >= 2 candidates
    valid_row: np.ndarray    # valid_r expanded to rows (float)
    w: float


@dataclass
class RaceSet:
    X: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    seg: np.ndarray
    mkt_logit: np.ndarray
    mkt_ok: np.ndarray
    stages: list[Stage] = field(default_factory=list)
    S: np.ndarray | None = None
    M: np.ndarray | None = None

    @property
    def n_valid(self) -> int:
        return int(self.stages[0].valid_r.sum())

    def select(self, idx: list[int]) -> "RaceSet":
        ix = np.ix_(idx, idx)
        return replace(
            self,
            X=np.ascontiguousarray(self.X[:, idx]),
            S=None if self.S is None else self.S[ix],
            M=None if self.M is None else self.M[ix],
        )


def seg_lse(u: np.ndarray, rs: RaceSet, mask: np.ndarray | None = None):
    """Per-race log-sum-exp and softmax, max-shifted. Masked-out rows get p = 0.

    Returns (lse[R], p[n]). No exp() ever sees a positive argument -> no overflow;
    the largest term per race is exp(0) = 1 -> denominator >= 1 -> no log(0).
    """
    um = u if mask is None else np.where(mask, u, -np.inf)
    m = np.maximum.reduceat(um, rs.starts)
    m = np.where(np.isfinite(m), m, 0.0)  # fully masked race guard
    e = np.exp(um - m[rs.seg])
    den = np.add.reduceat(e, rs.starts)
    den = np.where(den > 0, den, 1.0)
    return m + np.log(den), e / den[rs.seg]


def make_stages(finish_pos: np.ndarray, rs: RaceSet, weights: list[float]) -> list[Stage]:
    """Exploded-logit stages. Stage s: choose the runner(s) with finish_pos == s from
    {finish_pos >= s}. Dead heat for 1st (pos 1,1,3): stage 2 has no target and is
    skipped, stage 3 removes both tied winners. Non-finishers stay in every remaining set.
    """
    fp = np.where(np.isfinite(finish_pos) & (finish_pos > 0), finish_pos, np.inf)
    stages = []
    for s, w in enumerate(weights, start=1):
        tgt = (fp == s).astype(float)
        nt = np.add.reduceat(tgt, rs.starts)
        mask = fp >= s
        n_rem = np.add.reduceat(mask.astype(float), rs.starts)
        valid_r = (nt > 0) & (n_rem >= 2)
        y = tgt / np.where(nt > 0, nt, 1.0)[rs.seg]
        stages.append(
            Stage(
                mask=None if mask.all() else mask,
                y=y,
                valid_r=valid_r,
                valid_row=valid_r[rs.seg].astype(float),
                w=float(w),
            )
        )
    return stages


def penalty_mats(rs: RaceSet) -> tuple[np.ndarray, np.ndarray]:
    """S: pooled within-race utility covariance; M: mean squared race-level covariance
    between each feature and the (globally standardised) within-race market logit."""
    rw = rs.stages[0].valid_r.astype(float)
    n_r = rw.sum()
    cnt = rs.counts[:, None].astype(float)
    xc = rs.X - (np.add.reduceat(rs.X, rs.starts, axis=0) / cnt)[rs.seg]
    wrow = (rw / rs.counts)[rs.seg]
    S = xc.T @ (xc * wrow[:, None]) / n_r
    lm = rs.mkt_logit - (np.add.reduceat(rs.mkt_logit, rs.starts) / rs.counts)[rs.seg]
    v = rw[rs.seg] > 0
    lm = lm / max(np.sqrt(np.mean(lm[v] ** 2)), 1e-9)
    A = np.add.reduceat(lm[:, None] * rs.X, rs.starts, axis=0) / cnt  # (R, d)
    M = (A * rw[:, None]).T @ A / n_r
    return S, M


def prepare(
    df: pd.DataFrame, feats: list[str], mean: pd.Series, std: pd.Series,
    weights: list[float], labels: bool = True, penalties: bool = False,
) -> RaceSet:
    codes = pd.factorize(df["race_id"])[0]
    change = np.r_[True, codes[1:] != codes[:-1]]
    starts = np.flatnonzero(change)
    assert len(starts) == codes.max() + 1, "rows of a race must be contiguous"
    seg = np.cumsum(change) - 1
    counts = np.diff(np.r_[starts, len(df)])
    X = ((df[feats].astype(float) - mean[feats]) / std[feats]).fillna(0.0).to_numpy()
    ok_row = df["mkt_ok"].to_numpy(bool).astype(int)
    rs = RaceSet(
        X=np.ascontiguousarray(X, dtype=np.float64),
        starts=starts, counts=counts, seg=seg,
        mkt_logit=df["mkt_logit"].to_numpy(float),
        mkt_ok=np.minimum.reduceat(ok_row, starts) > 0,
    )
    if labels:
        rs.stages = make_stages(df["finish_pos"].to_numpy(float), rs, weights)
        if penalties:
            rs.S, rs.M = penalty_mats(rs)
    return rs


# --------------------------------------------------------------- objective and solver
def nll_grad(beta: np.ndarray, rs: RaceSet) -> tuple[float, np.ndarray]:
    """Sum over races/stages of w_s * [lse_s - sum_j y_j u_j] and its exact gradient
    X'(p - y) -- one matvec in, one matvec out, no Python loop over races."""
    u = rs.X @ beta
    total = 0.0
    gu = np.zeros_like(u)
    for st in rs.stages:
        lse, p = seg_lse(u, rs, st.mask)
        yu = np.add.reduceat(st.y * u, rs.starts)
        total += st.w * float(np.sum(np.where(st.valid_r, lse - yu, 0.0)))
        gu += st.w * st.valid_row * (p - st.y)
    return total, rs.X.T @ gu


def penalty_matrix(rs: RaceSet, lam: float, l1_ratio: float, lam_mkt: float) -> np.ndarray:
    d = rs.X.shape[1]
    Q = 1e-8 * np.eye(d)
    if lam > 0:
        Q = Q + lam * (1 - l1_ratio) * rs.S
    if lam_mkt > 0:
        Q = Q + lam_mkt * rs.M
    return Q


def fit_beta(
    rs: RaceSet, lam: float = 0.0, l1_ratio: float = 0.5, lam_mkt: float = 0.0,
    beta0: np.ndarray | None = None,
) -> np.ndarray:
    """min  (1/R) NLL(b) + b'Qb + lam*l1_ratio*||b||_1,  Q = lam(1-a)S + lam_mkt*M."""
    d = rs.X.shape[1]
    Q = penalty_matrix(rs, lam, l1_ratio, lam_mkt)
    l1 = lam * l1_ratio
    norm = float(rs.n_valid)
    b0 = np.zeros(d) if beta0 is None else beta0
    opts = {"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-9}

    def smooth(b: np.ndarray) -> tuple[float, np.ndarray]:
        nll, g = nll_grad(b, rs)
        Qb = Q @ b
        return nll / norm + b @ Qb, g / norm + 2 * Qb

    if l1 > 0:
        def obj(z: np.ndarray) -> tuple[float, np.ndarray]:
            f, g = smooth(z[:d] - z[d:])
            return f + l1 * z.sum(), np.concatenate([g + l1, -g + l1])

        z0 = np.concatenate([np.maximum(b0, 0), np.maximum(-b0, 0)])
        res = minimize(obj, z0, jac=True, method="L-BFGS-B",
                       bounds=[(0, None)] * (2 * d), options=opts)
        beta = res.x[:d] - res.x[d:]
        beta[np.abs(beta) < 1e-6] = 0.0
        return beta
    return minimize(smooth, b0, jac=True, method="L-BFGS-B", options=opts).x


def standard_errors(beta: np.ndarray, rs: RaceSet, lam: float, l1_ratio: float,
                    lam_mkt: float) -> np.ndarray:
    """Sandwich-free model-based SE: inv(I(beta) + R*2Q). Exact for lam = lam_mkt = 0 and a
    single stage; otherwise approximate (use a race-cluster bootstrap for final inference)."""
    u = rs.X @ beta
    H = np.zeros((len(beta), len(beta)))
    for st in rs.stages:
        _, p = seg_lse(u, rs, st.mask)
        pw = p * st.valid_row
        xp = rs.X * pw[:, None]
        mr = np.add.reduceat(xp, rs.starts, axis=0)
        H += st.w * (rs.X.T @ xp - mr.T @ mr)
    H += rs.n_valid * 2 * penalty_matrix(rs, lam, l1_ratio, lam_mkt)
    return np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 0, None))


# ------------------------------------------------------------------------ evaluation
def per_race_nll(p: np.ndarray, rs: RaceSet) -> np.ndarray:
    """Win NLL per race with dead-heat-fractional target; NaN for unlabeled races."""
    st = rs.stages[0]
    v = np.add.reduceat(-st.y * np.log(np.clip(p, MIN_PROB, None)), rs.starts)
    return np.where(st.valid_r, v, np.nan)


def tune_loss(beta: np.ndarray, rs: RaceSet) -> float:
    _, p = seg_lse(rs.X @ beta, rs)
    return float(np.nanmean(per_race_nll(p, rs)))


def within_race_corr(u: np.ndarray, v: np.ndarray, rs: RaceSet, races: np.ndarray) -> float:
    def centre(a: np.ndarray) -> np.ndarray:
        return a - (np.add.reduceat(a, rs.starts) / rs.counts)[rs.seg]

    uc, vc, rows = centre(u), centre(v), races[rs.seg]
    return float((uc * vc)[rows].sum() / np.sqrt((uc ** 2)[rows].sum() * (vc ** 2)[rows].sum()))


def blend_with_market(u: np.ndarray, rs: RaceSet) -> dict:
    """Benter combination softmax(a*market_logit + b*u) fit on the tune window (2 params,
    in-sample). Gain over the market alone = incremental information in u."""
    st = rs.stages[0]
    ok_r = rs.mkt_ok & st.valid_r
    st2 = replace(st, valid_r=ok_r, valid_row=ok_r[rs.seg].astype(float))
    rs2 = replace(rs, X=np.column_stack([rs.mkt_logit, u]), stages=[st2], S=None, M=None)
    ab = fit_beta(rs2, beta0=np.array([1.0, 0.0]))
    _, p_blend = seg_lse(rs2.X @ ab, rs2)
    _, p_mkt = seg_lse(rs.mkt_logit, rs)
    _, p_mod = seg_lse(u, rs)
    nll = lambda p: float(np.nanmean(per_race_nll(p, rs2)[ok_r]))  # noqa: E731
    return {
        "market_nll": nll(p_mkt), "model_nll": nll(p_mod), "blend_nll": nll(p_blend),
        "blend_a_mkt": float(ab[0]), "blend_b_model": float(ab[1]),
        "corr_mkt": within_race_corr(u, rs.mkt_logit, rs, ok_r),
    }


# ------------------------------------------------------------------ feature selection
def select_features(rs_tr: RaceSet, rs_tu: RaceSet, corr: pd.DataFrame, min_gain: float):
    idx = [ALL_FEATURES.index(f) for f in BASE_FEATURES]

    def score(cols: list[int]) -> float:
        beta = fit_beta(rs_tr.select(cols))
        return tune_loss(beta, rs_tu.select(cols))

    current = score(idx)
    rows = [{"step": 0, "candidate": "(base)", "log_loss": current, "improvement": np.nan,
             "accepted": True, "note": "", "selected_features": ", ".join(BASE_FEATURES)}]
    remaining = list(CANDIDATE_FEATURES)
    while remaining:
        scored, blocked = [], []
        names = [ALL_FEATURES[i] for i in idx]
        for c in remaining:
            if corr.loc[c, names].abs().max() > SPEARMAN_LIMIT:
                blocked.append(c)
                continue
            scored.append((score(idx + [ALL_FEATURES.index(c)]), c))
        for c in blocked:
            remaining.remove(c)
            rows.append({"step": len(rows), "candidate": c, "log_loss": np.nan,
                         "improvement": np.nan, "accepted": False,
                         "note": f"|spearman|>{SPEARMAN_LIMIT}", "selected_features": ""})
        if not scored:
            break
        loss, cand = min(scored)
        gain = current - loss
        ok = gain >= min_gain
        rows.append({"step": len(rows), "candidate": cand, "log_loss": loss,
                     "improvement": gain, "accepted": ok, "note": "",
                     "selected_features": ", ".join(names + [cand])})
        remaining.remove(cand)
        if not ok:
            break
        idx.append(ALL_FEATURES.index(cand))
        current = loss
    return idx, pd.DataFrame(rows)


def run_grid(rs_tr: RaceSet, rs_tu: RaceSet, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for lam in args.lambdas:
        for lam_mkt in args.mkt_lambdas:
            beta = fit_beta(rs_tr, lam, args.l1_ratio, lam_mkt)
            u = rs_tu.X @ beta
            r = blend_with_market(u, rs_tu)
            rows.append({"lam": lam, "lam_mkt": lam_mkt, **r,
                         "blend_gain": r["market_nll"] - r["blend_nll"],
                         "model_gain": r["market_nll"] - r["model_nll"],
                         "nnz": int((beta != 0).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ Win / Place / Quinella
def harville_markets(p: np.ndarray, rs: RaceSet):
    """Per-runner P(top2), P(top3) and all-pairs quinella from stage-1 win probabilities
    (Harville == Plackett-Luce under the fitted utility). Per race O(J^3), J <= 14."""
    top2, top3 = np.empty_like(p), np.empty_like(p)
    pi, pj, pq = [], [], []
    for a, J in zip(rs.starts, rs.counts):
        q = p[a:a + J]
        den1 = np.maximum(1.0 - q, MIN_PROB)
        P2 = q[:, None] * q[None, :] / den1[:, None]          # P(i 1st, j 2nd)
        np.fill_diagonal(P2, 0.0)
        top2[a:a + J] = q + P2.sum(0)
        den2 = np.maximum(1.0 - q[:, None] - q[None, :], MIN_PROB)
        P3 = P2[:, :, None] * q[None, None, :] / den2[:, :, None]
        i, j, k = np.ogrid[:J, :J, :J]
        P3 = P3 * ((i != j) & (i != k) & (j != k))
        top3[a:a + J] = top2[a:a + J] + P3.sum((0, 1))
        iu, ju = np.triu_indices(J, 1)
        pi.append(a + iu); pj.append(a + ju); pq.append(P2[iu, ju] + P2[ju, iu])
    return top2, top3, np.concatenate(pi), np.concatenate(pj), np.concatenate(pq)


# ----------------------------------------------------------------------------- report
def write_report(out: Path, summary: dict, selected: list[str], trace: pd.DataFrame,
                 grid: pd.DataFrame, corr: pd.DataFrame, coefs: pd.DataFrame) -> None:
    beats = summary["model_nll"] < summary["market_nll"]
    adds = summary["blend_gain"] > 0
    text = [
        "# Conditional-logit tuning report",
        "",
        f"## Status: SELECTED - standalone {'BEATS' if beats else 'DOES NOT BEAT'} market; "
        f"blend {'ADDS' if adds else 'DOES NOT ADD'} information over market",
        "",
        "Validation window: Kaggle 2019/20 (`k_tune`), races with complete market prices.",
        f"- Market per-race log loss: **{summary['market_nll']:.4f}**",
        f"- Selected conditional logit: **{summary['model_nll']:.4f}** "
        f"(gain {summary['model_gain']:+.4f})",
        f"- Benter blend softmax(a*mkt + b*u): **{summary['blend_nll']:.4f}** "
        f"(gain {summary['blend_gain']:+.4f}; a={summary['blend_a_mkt']:.3f}, "
        f"b={summary['blend_b_model']:.3f}; 2 in-sample parameters)",
        f"- Within-race corr(utility, market logit): {summary['corr_mkt']:.3f}",
        "",
        f"Hyper-parameters: lambda={summary['lam']}, l1_ratio={summary['l1_ratio']}, "
        f"lambda_mkt={summary['lam_mkt']}; stage weights={summary['stage_weights']}.",
        "",
        "Odds are used only inside the training-time orthogonality penalty; no odds or "
        "market-derived value is a model input. The Apr-Oct 2026 HKJC window is predicted, "
        "never scored. Feature selection and penalty choice both use `k_tune`, so its "
        "gains are mildly optimistic.",
        "",
        "## Selected features", "", *[f"- `{f}`" for f in selected], "",
        "## Selection trace", "", "```", trace.round(4).to_string(index=False), "```", "",
        "## Penalty grid (k_tune)", "", "```", grid.round(4).to_string(index=False), "```", "",
        "## Coefficients (train+tune refit, standardised features)", "", "```",
        coefs.round(4).to_string(index=False), "```", "",
        "SEs are model-based and approximate whenever a penalty or extra stage is active.",
    ]
    (out / "tuning_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    trace.to_csv(out / "selection_trace.csv", index=False)
    grid.to_csv(out / "penalty_grid.csv", index=False)
    corr.to_csv(out / "feature_correlations.csv")


def moments(df: pd.DataFrame, feats: list[str]) -> tuple[pd.Series, pd.Series]:
    x = df[feats].astype(float)
    return x.mean(), x.std().replace(0, 1).fillna(1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    local_feat = Path("data/clean data/runners_merged_with_features.parquet")
    local = Path("data/clean data/runners_merged.parquet")
    sage = Path("data/processed/runners_merged.parquet")
    parser.add_argument("--data", type=Path,
                        default=local_feat if local_feat.exists()
                        else local if local.exists() else sage)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/conditional_logit_tuning"))
    parser.add_argument("--minimum-gain", type=float, default=0.001)
    parser.add_argument("--stage-weights", type=float, nargs="+", default=[1.0],
                        help="Exploded-logit weights for win/2nd/3rd...; e.g. 1 0.4 0.2.")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.003, 0.01])
    parser.add_argument("--mkt-lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.5, 2.0])
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = build_features(pd.read_parquet(args.data))
    train = data.loc[data["split"].eq("k_train")]
    tune = data.loc[data["split"].eq("k_tune")]
    mean, std = moments(train, ALL_FEATURES)
    rs_tr = prepare(train, ALL_FEATURES, mean, std, args.stage_weights, penalties=True)
    rs_tu = prepare(tune, ALL_FEATURES, mean, std, [1.0])
    corr_all = train[ALL_FEATURES].corr(method="spearman")
    print(f"features+sets built in {time.time() - t0:.1f}s "
          f"(train races {rs_tr.n_valid}, tune races {rs_tu.n_valid})")

    idx, trace = select_features(rs_tr, rs_tu, corr_all, args.minimum_gain)
    selected = [ALL_FEATURES[i] for i in idx]
    tr_s, tu_s = rs_tr.select(idx), rs_tu.select(idx)
    grid = run_grid(tr_s, tu_s, args)
    best = grid.sort_values("blend_gain", ascending=False).iloc[0]
    hp = {"lam": float(best["lam"]), "l1_ratio": args.l1_ratio,
          "lam_mkt": float(best["lam_mkt"])}
    summary = {**best.to_dict(), **hp, "stage_weights": args.stage_weights}

    beta_t = fit_beta(tr_s, **hp)
    pd.DataFrame({"feature": selected, "coefficient": beta_t,
                  "standard_error": standard_errors(beta_t, tr_s, **hp)}
                 ).to_csv(args.output_dir / "tuning_coefficients.csv", index=False)

    # Final refit on train + tune with the chosen penalties; predict HKJC windows (unscored).
    final = data.loc[data["split"].isin(["k_train", "k_tune"])]
    mean_f, std_f = moments(final, selected)
    rs_f = prepare(final, selected, mean_f, std_f, args.stage_weights, penalties=True)
    beta = fit_beta(rs_f, **hp)
    coefs = pd.DataFrame({"feature": selected, "coefficient": beta,
                          "standard_error": standard_errors(beta, rs_f, **hp)})
    coefs.to_csv(args.output_dir / "coefficients.csv", index=False)

    out = data.loc[data["split"].isin(["h_s2", "h_test"])].reset_index(drop=True)
    rs_o = prepare(out, selected, mean_f, std_f, [], labels=False)
    _, p_win = seg_lse(rs_o.X @ beta, rs_o)
    top2, top3, qi, qj, qp = harville_markets(p_win, rs_o)
    pred = out[["race_id", "horse_id", "split", "is_winner", "market_prob"]].copy()
    pred["p_cl"], pred["p_top2"], pred["p_top3"] = p_win, top2, top3
    pred.to_parquet(args.output_dir / "predictions.parquet", index=False)
    pd.DataFrame({"race_id": out["race_id"].to_numpy()[qi],
                  "horse_id_1": out["horse_id"].to_numpy()[qi],
                  "horse_id_2": out["horse_id"].to_numpy()[qj],
                  "p_quinella": qp}).to_parquet(args.output_dir / "quinella_pairs.parquet",
                                                index=False)
    (args.output_dir / "selected_features.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "hyperparameters.json").write_text(
        json.dumps({**hp, "stage_weights": args.stage_weights}, indent=2) + "\n",
        encoding="utf-8")
    write_report(args.output_dir, summary, selected, trace, grid,
                 train[selected].astype(float).corr(method="spearman"), coefs)

    verb = "beats" if summary["model_nll"] < summary["market_nll"] else "does not beat"
    print(f"SELECTED: CL {summary['model_nll']:.4f} {verb} market "
          f"{summary['market_nll']:.4f} on k_tune; blend {summary['blend_nll']:.4f} "
          f"(gain {summary['blend_gain']:+.4f}). hp={hp}. "
          f"{time.time() - t0:.1f}s. Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())