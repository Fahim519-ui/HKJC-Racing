"""Leakage-safe jockey-trainer partnership history features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _lagged_group_rates(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    date_col: str,
    target: pd.Series,
    window_days: int,
    last_n: int,
    prior: float,
    smoothing_k: float,
) -> pd.DataFrame:
    """Return pre-day rolling rates for one grouping without row-wise loops."""
    work = frame[group_cols + [date_col]].copy()
    work["_row"] = np.arange(len(work))
    work["_day"] = (
        pd.to_datetime(work[date_col], errors="raise").dt.normalize()
        - pd.Timestamp("1970-01-01")
    ).dt.days

    outcome = pd.to_numeric(target, errors="coerce").to_numpy(float)
    observed = np.isfinite(outcome)
    work["_hit"] = np.where(observed, outcome, 0.0)
    work["_count"] = observed.astype(float)
    work["_group"] = pd.factorize(
        pd.MultiIndex.from_frame(work[group_cols]), sort=False
    )[0]
    work = work.sort_values(["_group", "_day", "_row"], kind="stable").reset_index(drop=True)

    group = work["_group"].to_numpy()
    day = work["_day"].to_numpy(np.int64)
    hit = work["_hit"].to_numpy(float)
    count = work["_count"].to_numpy(float)
    positions = np.arange(len(work))

    group_start = np.r_[True, group[1:] != group[:-1]]
    day_start = np.r_[True, group_start[1:] | (day[1:] != day[:-1])]
    group_first = np.maximum.accumulate(np.where(group_start, positions, 0))
    day_first = np.maximum.accumulate(np.where(day_start, positions, 0))

    # Prefix i contains rows [0, i). Every row on a date uses day_first as
    # its end position, so all same-day rides are excluded from its history.
    hit_prefix = np.r_[0.0, np.cumsum(hit)]
    count_prefix = np.r_[0.0, np.cumsum(count)]

    boundary_rows = np.flatnonzero(day_start)
    boundary_group = group[boundary_rows]
    boundary_day = day[boundary_rows]
    spacing = int(boundary_day.max() - boundary_day.min()) + window_days + 2
    packed_day = boundary_group.astype(np.int64) * spacing + boundary_day
    lower_boundary = np.searchsorted(
        packed_day, packed_day - window_days, side="left"
    )
    day_id = np.cumsum(day_start) - 1
    window_begin = boundary_rows[lower_boundary[day_id]]

    hit_365 = hit_prefix[day_first] - hit_prefix[window_begin]
    n_365 = count_prefix[day_first] - count_prefix[window_begin]

    # The last-N window ends at the current date's first row for the same
    # reason: a person's other rides that day are unavailable pre-race.
    group_count_start = count_prefix[group_first]
    desired_start = np.maximum(count_prefix[day_first] - last_n, group_count_start)
    last_n_begin = np.searchsorted(count_prefix, desired_start, side="left")
    last_n_begin = np.maximum(last_n_begin, group_first)
    hit_n = hit_prefix[day_first] - hit_prefix[last_n_begin]
    n_n = count_prefix[day_first] - count_prefix[last_n_begin]

    denominator_365 = n_365 + smoothing_k
    denominator_n = n_n + smoothing_k
    rate_365 = np.divide(
        hit_365 + smoothing_k * prior,
        denominator_365,
        out=np.full(len(work), prior, dtype=float),
        where=denominator_365 > 0,
    )
    rate_n = np.divide(
        hit_n + smoothing_k * prior,
        denominator_n,
        out=np.full(len(work), prior, dtype=float),
        where=denominator_n > 0,
    )
    result = pd.DataFrame(
        {
            "_row": work["_row"].to_numpy(),
            "rate_365": rate_365,
            "n_365": n_365,
            f"rate_last_{last_n}": rate_n,
            f"n_last_{last_n}": n_n,
        }
    ).sort_values("_row", kind="stable")
    result.index = frame.index
    return result.drop(columns="_row")


def add_lagged_jockey_trainer_features(
    frame: pd.DataFrame,
    *,
    date_col: str = "race_date",
    source_col: str = "source",
    jockey_col: str = "jockey_key",
    trainer_col: str = "trainer_key",
    win_col: str = "is_winner",
    race_col: str = "race_id",
    window_days: int = 365,
    last_n: int = 50,
    prior_win_rate: float = 1 / 12,
    individual_k: float = 30.0,
    partnership_k: float = 50.0,
    prefix: str = "jtp",
) -> pd.DataFrame:
    """Add smoothed, strictly pre-day jockey-trainer partnership effects.

    Partnership rates shrink toward the count-weighted individual jockey and
    trainer baseline. The effect columns are residual team signal beyond that
    baseline and are the intended model inputs.
    """
    required = {date_col, source_col, jockey_col, trainer_col, win_col, race_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = frame.copy()
    target = pd.to_numeric(out[win_col], errors="coerce")
    jockey = _lagged_group_rates(
        out,
        group_cols=[source_col, jockey_col],
        date_col=date_col,
        target=target,
        window_days=window_days,
        last_n=last_n,
        prior=prior_win_rate,
        smoothing_k=individual_k,
    )
    trainer = _lagged_group_rates(
        out,
        group_cols=[source_col, trainer_col],
        date_col=date_col,
        target=target,
        window_days=window_days,
        last_n=last_n,
        prior=prior_win_rate,
        smoothing_k=individual_k,
    )
    pair = _lagged_group_rates(
        out,
        group_cols=[source_col, jockey_col, trainer_col],
        date_col=date_col,
        target=target,
        window_days=window_days,
        last_n=last_n,
        prior=prior_win_rate,
        smoothing_k=0.0,
    )

    def combined_baseline(
        jockey_rate: pd.Series,
        jockey_count: pd.Series,
        trainer_rate: pd.Series,
        trainer_count: pd.Series,
    ) -> pd.Series:
        total_count = jockey_count + trainer_count
        return pd.Series(
            np.where(
                total_count > 0,
                (jockey_rate * jockey_count + trainer_rate * trainer_count)
                / total_count,
                prior_win_rate,
            ),
            index=out.index,
        )

    baseline_365 = combined_baseline(
        jockey["rate_365"], jockey["n_365"], trainer["rate_365"], trainer["n_365"]
    )
    baseline_n = combined_baseline(
        jockey[f"rate_last_{last_n}"],
        jockey[f"n_last_{last_n}"],
        trainer[f"rate_last_{last_n}"],
        trainer[f"n_last_{last_n}"],
    )
    pair_365 = (
        pair["rate_365"] * pair["n_365"] + partnership_k * baseline_365
    ) / (pair["n_365"] + partnership_k)
    pair_n = (
        pair[f"rate_last_{last_n}"] * pair[f"n_last_{last_n}"]
        + partnership_k * baseline_n
    ) / (pair[f"n_last_{last_n}"] + partnership_k)

    out[f"{prefix}_pair_rides365"] = pair["n_365"]
    out[f"{prefix}_pair_baseline365"] = baseline_365
    out[f"{prefix}_pair_win365"] = pair_365
    out[f"{prefix}_pair_effect365"] = pair_365 - baseline_365
    out[f"{prefix}_pair_rides_last_{last_n}"] = pair[f"n_last_{last_n}"]
    out[f"{prefix}_pair_baseline_last_{last_n}"] = baseline_n
    out[f"{prefix}_pair_win_last_{last_n}"] = pair_n
    out[f"{prefix}_pair_effect_last_{last_n}"] = pair_n - baseline_n

    for column in (
        f"{prefix}_pair_win365",
        f"{prefix}_pair_effect365",
        f"{prefix}_pair_win_last_{last_n}",
        f"{prefix}_pair_effect_last_{last_n}",
    ):
        out[f"{column}_rel"] = out[column] - out.groupby(race_col)[column].transform("mean")

    return out
