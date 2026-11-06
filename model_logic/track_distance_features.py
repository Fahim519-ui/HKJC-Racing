"""Source-separated, prior-day, hierarchically smoothed track/distance form (roadmap Work 1).

For every runner we estimate how the horse has performed at today's track, today's
distance band and today's track x distance-band combination, using only races run on
strictly earlier dates in the same source (Kaggle 2017-20 or HKJC 2025-26).

Three outcome measures are used: win rate, top-three rate and mean finishing
percentile (0 = winner, 1 = last). Each is smoothed down a hierarchy, every level
shrinking toward the level above it with a fixed pseudo-count:

    source prior  (prior-day running rate of the whole source)
      -> career            (this horse, any track/distance)       weight K["career"]
           -> track         (this horse at today's track)          weight K["track"]
                -> track x distance  (this horse, track + band)    weight K["track_distance"]
           -> distance band (this horse in today's band)            weight K["distance_band"]

    smoothed = (sum of outcomes at this level + K * parent) / (starts at this level + K)

So a horse with no history at a level simply inherits its parent, and the estimate
moves toward the horse's own record as starts accumulate. Nothing is fitted to the
data: every prior is either a constant or a running total of earlier dates, so the
feature on day d is unchanged if any race after d is altered or removed.

The nine model features are the race-relative versions (value minus the race mean).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DISTANCE_BANDS = (
    # (label, inclusive upper bound in metres)
    ("sprint", 1200),   # 1000, 1200
    ("mile", 1650),     # 1400, 1600, 1650
    ("middle", 2000),   # 1800, 2000
    ("staying", np.inf),  # 2200, 2400
)
# Pseudo-counts (in starts). Fixed a priori, not tuned on any split.
K = {"career": 5.0, "track": 5.0, "distance_band": 5.0, "track_distance": 5.0}
# Used only before a source has any completed race day.
SOURCE_DEFAULTS = {"win": 1 / 12, "top3": 3 / 12, "fin_pct": 0.5}
MEASURES = ("win", "top3", "fin_pct")
LEVELS = ("track", "distance_band", "track_distance")

TD_FEATURES = tuple(f"td_{level}_{measure}_rel" for level in LEVELS for measure in MEASURES)
TD_HISTORY_COLUMNS = tuple(f"td_{level}_starts" for level in LEVELS)

_DAY0 = pd.Timestamp("1990-01-01")
_BIG = np.int64(1 << 20)


def distance_band(distance_m: pd.Series) -> pd.Series:
    distance = distance_m.astype(float).to_numpy()
    labels = np.array([label for label, _ in DISTANCE_BANDS], dtype=object)
    bounds = np.array([upper for _, upper in DISTANCE_BANDS], dtype=float)
    return pd.Series(labels[np.searchsorted(bounds, distance, side="left")], index=distance_m.index)


def prior_day_sums(data: pd.DataFrame, keys: list[str], values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per row, the totals of `values` over the same key group on strictly earlier dates.

    All rows of the group on the current date are excluded, so same-day results never
    leak (this matters for the source-level prior, where many races share a date).
    """
    gid = data.groupby(keys, sort=False, dropna=False).ngroup().to_numpy(np.int64)
    day = (data["race_date"] - _DAY0).dt.days.to_numpy(np.int64)
    uniq, inv = np.unique(gid * _BIG + day, return_inverse=True)
    group = uniq // _BIG
    result = {}
    for name, value in values.items():
        per_day = pd.Series(np.bincount(inv, weights=value, minlength=len(uniq)))
        # Cumulate within each group only, so a value depends on nothing but that group's
        # own earlier days (bit-for-bit, not just up to floating-point rounding).
        through_day = per_day.groupby(group, sort=False).cumsum()
        before_day = through_day.groupby(group, sort=False).shift(1, fill_value=0.0)
        result[name] = before_day.to_numpy()[inv]
    return result


def _outcomes(data: pd.DataFrame) -> dict[str, np.ndarray]:
    pct = data["finish_percentile"].to_numpy(float)
    has_pct = np.isfinite(pct)
    return {
        "n": np.ones(len(data)),
        "win": data["is_winner"].to_numpy(float),
        "top3": data["is_top3"].to_numpy(float),
        "fin_pct": np.where(has_pct, pct, 0.0),
        "n_pct": has_pct.astype(float),
    }


def _shrink(sums: dict[str, np.ndarray], parent: dict[str, np.ndarray], weight: float) -> dict[str, np.ndarray]:
    return {
        "win": (sums["win"] + weight * parent["win"]) / (sums["n"] + weight),
        "top3": (sums["top3"] + weight * parent["top3"]) / (sums["n"] + weight),
        "fin_pct": (sums["fin_pct"] + weight * parent["fin_pct"]) / (sums["n_pct"] + weight),
    }


def add_track_distance_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add td_* columns. Expects the builder's frame: started runners of valid races with
    `finish_percentile`, `is_winner`, `is_top3`, `track_code`, `distance_m` and `race_id`."""
    data["td_distance_band"] = distance_band(data["distance_m"])
    outcomes = _outcomes(data)

    source_sums = prior_day_sums(data, ["source"], outcomes)
    source_prior = {
        "win": (source_sums["win"] + SOURCE_DEFAULTS["win"]) / (source_sums["n"] + 1.0),
        "top3": (source_sums["top3"] + SOURCE_DEFAULTS["top3"]) / (source_sums["n"] + 1.0),
        "fin_pct": (source_sums["fin_pct"] + SOURCE_DEFAULTS["fin_pct"]) / (source_sums["n_pct"] + 1.0),
    }
    horse = ["source", "horse_id"]
    career = _shrink(prior_day_sums(data, horse, outcomes), source_prior, K["career"])

    track_sums = prior_day_sums(data, horse + ["track_code"], outcomes)
    band_sums = prior_day_sums(data, horse + ["td_distance_band"], outcomes)
    combo_sums = prior_day_sums(data, horse + ["track_code", "td_distance_band"], outcomes)

    smoothed = {"track": _shrink(track_sums, career, K["track"])}
    smoothed["distance_band"] = _shrink(band_sums, career, K["distance_band"])
    smoothed["track_distance"] = _shrink(combo_sums, smoothed["track"], K["track_distance"])
    starts = {"track": track_sums["n"], "distance_band": band_sums["n"], "track_distance": combo_sums["n"]}

    race = data["race_id"]
    for level in LEVELS:
        data[f"td_{level}_starts"] = starts[level]
        for measure in MEASURES:
            value = pd.Series(smoothed[level][measure], index=data.index)
            data[f"td_{level}_{measure}_rel"] = value - value.groupby(race).transform("mean")
    return data
