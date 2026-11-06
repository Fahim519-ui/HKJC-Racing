"""Race-level evaluation helpers for forward-fold outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def race_losses(frame: pd.DataFrame, probability: str) -> pd.Series:
    winners = frame["is_winner"].to_numpy(float)
    weighted = frame[probability].to_numpy(float) * winners
    races = frame["race_id"].to_numpy()
    sums = pd.Series(weighted).groupby(races, sort=False).sum()
    counts = pd.Series(winners).groupby(races, sort=False).sum()
    return -np.log(np.clip(sums / counts, 1e-12, None))


def paired_bootstrap_gain(
    loss_a: pd.Series, loss_b: pd.Series, day_ids: pd.Series, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% race-day-cluster bootstrap interval for loss_b minus loss_a."""
    data = pd.DataFrame({"gain": loss_a - loss_b, "day": day_ids.to_numpy()})
    by_day = data.groupby("day", sort=False)["gain"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(by_day), size=(n_boot, len(by_day)))
    sums = by_day["sum"].to_numpy()[index].sum(axis=1)
    counts = by_day["count"].to_numpy()[index].sum(axis=1)
    values = sums / counts
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
