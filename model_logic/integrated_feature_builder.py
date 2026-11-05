"""Shared leakage-safe feature construction and model feature registry."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from model_logic.feature_registry import REGISTRY, get_features
    from model_logic.jt_partnership_features import add_lagged_jockey_trainer_features
except ModuleNotFoundError:
    from feature_registry import REGISTRY, get_features
    from jt_partnership_features import add_lagged_jockey_trainer_features


def __getattr__(name: str):
    """Provide read-only compatibility aliases while callers move to the registry."""
    roles = {
        "BASE_FEATURES": "base",
        "CANDIDATE_FEATURES": "candidate",
        "ALL_FEATURES": None,
    }
    if name not in roles:
        raise AttributeError(name)
    warnings.warn(
        f"feature_builder.{name} is deprecated; use feature_registry.get_features()",
        DeprecationWarning,
        stacklevel=2,
    )
    return tuple(get_features("conditional_logit", role=roles[name]))


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
    data: pd.DataFrame,
    key: str,
    target: str,
    prior: float,
    weight: float = 30.0,
    window: int = 365,
) -> np.ndarray:
    """Smoothed hit rate over [d-365, d) per (source, key); same-day rides excluded."""
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
    hi = np.arange(len(uniq_t))
    value = ((c_hits[hi] - c_hits[lo]) + weight * prior) / ((c_n[hi] - c_n[lo]) + weight)
    return value[inv]


def _nanmean_rows(values: np.ndarray) -> np.ndarray:
    count = (~np.isnan(values)).sum(1)
    return np.where(count > 0, np.nansum(values, 1) / np.maximum(count, 1), np.nan)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build all pre-race features used by the active conditional-logit and LightGBM models."""
    data = raw.loc[raw["race_valid"].astype(bool) & raw["started"].astype(bool)].copy()
    data["race_date"] = pd.to_datetime(data["race_date"])
    data = data.sort_values(["source", "race_date", "race_no"], kind="stable").reset_index(drop=True)
    data["is_winner"] = data["is_winner"].astype(float)
    data["is_top3"] = data["is_top3"].astype(float)

    data["n_runners"] = data.groupby("race_id")["race_id"].transform("size").astype(float)
    data = data.loc[data["n_runners"] >= 2].reset_index(drop=True)
    data["split"] = assign_split(data)
    n_runners = data["n_runners"]
    race = data.groupby("race_id")

    draw_rank = race["draw"].rank(method="first")
    data["rc_draw_pct"] = (draw_rank - 1) / (n_runners - 1)
    is_hv = data["track_code"].eq("HV").astype(float)
    is_sprint = (data["distance_m"].astype(float) <= 1200).astype(float)
    data["rc_draw_x_hv"] = data["rc_draw_pct"] * is_hv
    data["rc_draw_x_sprint"] = data["rc_draw_pct"] * is_sprint
    weight = data["carried_weight_lb"].astype(float)
    data["rc_weight_rel"] = weight - weight.groupby(data["race_id"]).transform("mean")
    data["rc_weight_x_dist"] = (
        data["rc_weight_rel"] * (data["distance_m"].astype(float) - 1400) / 1000
    )

    finish_position = data["finish_pos"].astype(float).where(lambda value: value > 0)
    data["finish_percentile"] = (finish_position - 1) / (n_runners - 1)

    # =========================================================
    # Nayeon's Work 1: Track, Distance, and Combo Form Features
    # =========================================================
    
    def get_distance_band(d):
        if d <= 1200: return 'sprint'
        elif d <= 1650: return 'mile'
        elif d <= 1800: return 'middle'
        else: return 'stay'

    data['distance_band'] = data['distance_m'].apply(get_distance_band)
    data['track_dist_combo'] = data['track_code'] + '_' + data['distance_band']
    
    # Global averages for shrinkage
    source_global_win = data.groupby('source')['is_winner'].transform('mean')
    source_global_top3 = data.groupby('source')['is_top3'].transform('mean')
    source_global_pct = data.groupby('source')['finish_percentile'].transform('mean')

    def calc_smoothed(data, group_cols, prefix):
        win_s = data.groupby(group_cols)['is_winner'].shift(1)
        top3_s = data.groupby(group_cols)['is_top3'].shift(1)
        pct_s = data.groupby(group_cols)['finish_percentile'].shift(1)
        
        data['_tmp_win'] = win_s
        data['_tmp_top3'] = top3_s
        data['_tmp_pct'] = pct_s
        
        runs = data.groupby(group_cols)['_tmp_win'].transform(lambda x: x.expanding().count())
        wins = data.groupby(group_cols)['_tmp_win'].transform(lambda x: x.expanding().sum())
        top3s = data.groupby(group_cols)['_tmp_top3'].transform(lambda x: x.expanding().sum())
        pcts = data.groupby(group_cols)['_tmp_pct'].transform(lambda x: x.expanding().sum())
        
        runs = runs.fillna(0); wins = wins.fillna(0); top3s = top3s.fillna(0); pcts = pcts.fillna(0)
        k = 5
        
        data[f'{prefix}_win'] = (wins + k * source_global_win) / (runs + k)
        data[f'{prefix}_top3'] = (top3s + k * source_global_top3) / (runs + k)
        data[f'{prefix}_pct'] = (pcts + k * source_global_pct) / (runs + k)
        
        data.drop(columns=['_tmp_win', '_tmp_top3', '_tmp_pct'], inplace=True)
        return data

    data = calc_smoothed(data, ['source', 'horse_id', 'track_code'], 'td_track')
    data = calc_smoothed(data, ['source', 'horse_id', 'distance_band'], 'td_distance_band')
    data = calc_smoothed(data, ['source', 'horse_id', 'track_dist_combo'], 'td_track_distance')

    # Convert to relative measures
    for prefix in ['td_track', 'td_distance_band', 'td_track_distance']:
        for suffix in ['win', 'top3', 'pct']:
            col = f'{prefix}_{suffix}'
            data[f'{col}_rel'] = data[col] - data.groupby('race_id')[col].transform('mean')


    # =========================================================

    
    horse = data.groupby(["source", "horse_id"], sort=False)

    def lags(column: str, count: int) -> np.ndarray:
        return np.column_stack(
            [horse[column].shift(offset).to_numpy(float) for offset in range(1, count + 1)]
        )

    data["hf_days_since"] = np.log1p(
        (data["race_date"] - horse["race_date"].shift(1)).dt.days.clip(upper=365)
    )
    data["hf_first_start"] = horse.cumcount().eq(0).astype(float)
    finish_lags = lags("finish_percentile", 3)
    data["hf_last_fin_pct"] = finish_lags[:, 0]
    data["hf_avg3_fin_pct"] = _nanmean_rows(finish_lags)
    prior = 1 / 12
    win_lags = lags("is_winner", 5)
    data["hf_win5_s"] = (
        np.nansum(win_lags, 1) + 5 * prior
    ) / ((~np.isnan(win_lags)).sum(1) + 5)
    data["w3_horse_past_win_rate"] = _nanmean_rows(win_lags[:, :3])
    data["w3_horse_past_top3_rate"] = _nanmean_rows(lags("is_top3", 3))
    prize_ratio = data["prize_hkd"].astype(float) / horse["prize_hkd"].shift(1).astype(float)
    data["hf_class_change"] = np.log(prize_ratio.where((prize_ratio > 0) & np.isfinite(prize_ratio)))

    def relative(column: str) -> pd.Series:
        return data[column] - data.groupby("race_id")[column].transform("mean")

    data["hf_last_fin_pct_rel"] = relative("hf_last_fin_pct")
    data["hf_avg3_fin_pct_rel"] = relative("hf_avg3_fin_pct")
    data["jt_jockey_win365"] = rolling_rate(data, "jockey_key", "is_winner", prior)
    data["jt_trainer_win365"] = rolling_rate(data, "trainer_key", "is_winner", prior)
    data["jt_jockey_win365_rel"] = relative("jt_jockey_win365")
    data["jt_trainer_win365_rel"] = relative("jt_trainer_win365")
    data = add_lagged_jockey_trainer_features(data)

    market_probability = data["market_prob"].astype(float)
    market_complete = (market_probability > 0).groupby(data["race_id"]).transform("all")
    normalized_market = market_probability.where(market_probability > 0)
    normalized_market /= normalized_market.groupby(data["race_id"]).transform("sum")
    data["mkt_ok"] = market_complete.astype(bool)
    data["mkt_p"] = np.where(market_complete, normalized_market, 1.0 / n_runners)
    data["mkt_logit"] = np.log(np.clip(data["mkt_p"].to_numpy(float), 1e-6, None))

    data = data.drop(columns=["finish_percentile"]).sort_values(
        "race_id", kind="stable"
    ).reset_index(drop=True)
    required = [
        spec.name for spec in REGISTRY if spec.implemented and not spec.optional
    ]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise RuntimeError(f"Builder did not create registered features: {missing}")
    data.attrs["built_features"] = [
        spec.name for spec in REGISTRY if spec.implemented and spec.name in data.columns
    ]
    return data
