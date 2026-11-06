"""Form-trajectory features and two targeted interactions (roadmap Works 3-4).

Trajectory features describe the direction of a horse's form, not its level:

* tr_last_minus_avg3_fin_pct_rel - last finishing percentile minus the mean of the last
  three, centred on the race. Negative = the latest run was better than recent average.
* tr_fin_pct_slope3_rel - least-squares slope of the last (up to) three finishing
  percentiles, per start, centred on the race. Needs at least two earlier finishes.
  Negative = improving.
* tr_days_since_bucket - fixed layoff buckets on days since the previous start
  (0: <=14, 1: 15-28, 2: 29-60, 3: 61-180, 4: >180). NaN on a first start, which
  hf_first_start already flags. Bucket edges are fixed constants, so no quantiles are
  estimated from the data (and none from later periods).
* tr_class_change_direction - +1 up in class, -1 down, 0 same or unknown, from the log
  prize-money ratio to the previous start. Changes smaller than CLASS_DEADBAND are
  seasonal prize adjustments, not class moves, and count as 0.

Interactions:

* int_recent_form_x_class_change - three-start relative form x class-change direction.
  Recent form is less informative when the horse moves class.
* int_jt_strength_x_field_size - mean of the race-relative jockey and trainer 365-day
  win rates x (field size - 12) / 12. Lets connection strength matter more or less in
  big fields.

All inputs are earlier-date horse history or pre-race declarations, so each feature
inherits the leakage safety of hf_* / jt_* columns it is built from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DAYS_SINCE_EDGES = (14, 28, 60, 180)
CLASS_DEADBAND = 0.10
FIELD_SIZE_CENTRE = 12.0

TRAJECTORY_FEATURES = (
    "tr_last_minus_avg3_fin_pct_rel",
    "tr_fin_pct_slope3_rel",
    "tr_days_since_bucket",
    "tr_class_change_direction",
)
INTERACTION_FEATURES = ("int_recent_form_x_class_change", "int_jt_strength_x_field_size")


def _race_relative(data: pd.DataFrame, values: np.ndarray | pd.Series) -> pd.Series:
    series = pd.Series(np.asarray(values, dtype=float), index=data.index)
    return series - series.groupby(data["race_id"]).transform("mean")


def _slope(lags: np.ndarray) -> np.ndarray:
    """OLS slope per row of `lags` (column 0 = most recent), at least two finite points."""
    x = -np.arange(lags.shape[1], dtype=float)
    mask = np.isfinite(lags)
    count = mask.sum(1)
    safe = np.maximum(count, 1)
    x_mean = (mask * x).sum(1) / safe
    y_mean = np.where(mask, lags, 0.0).sum(1) / safe
    dx = np.where(mask, x - x_mean[:, None], 0.0)
    dy = np.where(mask, lags - y_mean[:, None], 0.0)
    denominator = (dx * dx).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (dx * dy).sum(1) / denominator
    return np.where(count >= 2, slope, np.nan)


def add_trajectory_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add tr_* and int_* columns. Requires the builder's hf_* and jt_* columns and
    `finish_percentile`, so call it after those are built."""
    horse = data.groupby(["source", "horse_id"], sort=False)
    finish_lags = np.column_stack(
        [horse["finish_percentile"].shift(offset).to_numpy(float) for offset in (1, 2, 3)]
    )

    data["tr_last_minus_avg3_fin_pct_rel"] = _race_relative(
        data, data["hf_last_fin_pct"] - data["hf_avg3_fin_pct"]
    )
    data["tr_fin_pct_slope3_rel"] = _race_relative(data, _slope(finish_lags))

    days = (data["race_date"] - horse["race_date"].shift(1)).dt.days.to_numpy(float)
    bucket = np.searchsorted(np.asarray(DAYS_SINCE_EDGES, dtype=float), days, side="left").astype(float)
    data["tr_days_since_bucket"] = np.where(np.isfinite(days), bucket, np.nan)

    change = data["hf_class_change"].to_numpy(float)
    direction = np.where(np.abs(change) > CLASS_DEADBAND, np.sign(change), 0.0)
    data["tr_class_change_direction"] = np.where(np.isfinite(change), direction, 0.0)

    recent_form = data["hf_avg3_fin_pct_rel"].fillna(0.0).to_numpy(float)
    data["int_recent_form_x_class_change"] = recent_form * data["tr_class_change_direction"]

    strength = (data["jt_jockey_win365_rel"] + data["jt_trainer_win365_rel"]) / 2.0
    field = (data["n_runners"] - FIELD_SIZE_CENTRE) / FIELD_SIZE_CENTRE
    data["int_jt_strength_x_field_size"] = strength * field
    return data
