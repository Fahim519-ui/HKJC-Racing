"""Shared, leakage-safe evaluation for every model (Week 5 "scoreboard").

Design rules (from WEEK5_PLAN.md and DATA_RULES.md):
  * The race is the unit. Every metric is a mean over races. The bootstrap resamples whole
    race DAYS (races on one day are correlated), keeping both models' predictions paired.
  * h_test (HKJC Apr-Oct 2026) is scored ONCE, in Week 5. Every function that touches
    predictions refuses h_test rows unless final_run=True is passed explicitly, so a
    stray call cannot peek at the test window.
  * Models are compared only on races where every compared column has a probability and
    the market is complete. Report the race count.
  * The honest baseline for "did the model beat the odds" is the market-only refit
    (p proportional to market^beta, beta fitted on the SAME data being scored only when
    that data is a fitting window; on a held-out window beta must come from training).

This module computes metrics and makes baselines. It does NOT fit models and does NOT
choose features. Keep those in the model scripts.

Quick use:
    import eval_utils as ev
    df = ev.load_predictions("artifacts/.../joint_predictions.parquet", split="h_s2")
    df = ev.add_odds_baselines(df)                 # p_power, p_shin, p_market_refit
    table = ev.scoreboard(df, ["p_market", "p_cl", "p_lgbm", "p_joint"],
                          baseline="p_market_refit")
    print(table)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

EPSILON = 1e-12
BINS = [0, 0.02, 0.05, 0.10, 0.20, 0.35, 1.0]
SEED = 17


# --------------------------------------------------------------------- loading / guard
H_TEST_START = pd.Timestamp("2026-04-01")   # HKJC test window (Kaggle data ends in 2020)


def is_test_row(df: pd.DataFrame) -> pd.Series:
    """True for rows in the HKJC test window, found by split label OR by race date, so a file
    without a split column (e.g. fold predictions) is still caught."""
    test = pd.Series(False, index=df.index)
    if "split" in df.columns:
        test |= df["split"].eq("h_test")
    if "race_date" in df.columns:
        test |= pd.to_datetime(df["race_date"]) >= H_TEST_START
    return test


def check_not_test(df: pd.DataFrame, final_run: bool = False) -> None:
    """Refuse h_test rows unless this is the sanctioned one-shot final run."""
    if is_test_row(df).any() and not final_run:
        raise RuntimeError(
            "h_test rows present (split == 'h_test' or race_date >= 2026-04-01). The test "
            "window is scored once, in Week 5. Pass final_run=True only for that sanctioned run."
        )


def load_predictions(path: str | Path, split: str | None = "h_s2",
                     final_run: bool = False) -> pd.DataFrame:
    """Load a predictions parquet. By default keep only h_s2 (safe to use any time).
    Pass split=None with final_run=True to keep every row for the one-shot test scoring."""
    df = pd.read_parquet(path)
    if split is not None:
        df = df.loc[df["split"].eq(split)].copy()
    check_not_test(df, final_run=final_run)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------- per-race metrics
def _winner_mask(df: pd.DataFrame) -> pd.Series:
    return df["is_winner"].astype(float).eq(1)


def race_losses(p: pd.Series, df: pd.DataFrame) -> pd.Series:
    """One cross-entropy loss per race: -log p(winner).

    Dead heat (two or more winners): -log of the MEAN winner probability. This matches the
    project convention in week34_starter.py and Fahim's model_logic/run_folds.py (checked 6 Nov:
    identical on every fold). Before 6 Nov this used the mean of the -log values, which differs
    only in dead-heat races (1 race in fold k1, 2 in h1).
    Returns a Series indexed by race_id, so two models' losses align for pairing."""
    p = pd.Series(np.asarray(p, dtype=float), index=df.index).clip(lower=EPSILON)
    win = _winner_mask(df)
    return -np.log(p[win].groupby(df.loc[win, "race_id"]).mean())


def per_race_log_loss(p: pd.Series, df: pd.DataFrame) -> float:
    return float(race_losses(p, df).mean())


def brier(p: pd.Series, df: pd.DataFrame) -> float:
    """Per race: sum over runners of (p - won)^2, averaged over races."""
    p = pd.Series(np.asarray(p, dtype=float), index=df.index)
    sq = (p - df["is_winner"].astype(float)) ** 2
    return float(sq.groupby(df["race_id"]).sum().mean())


def top1(p: pd.Series, df: pd.DataFrame) -> float:
    """Share of races whose highest-p horse won. Tied-top rows share the credit, so the
    row order (winners are listed first in the data) cannot inflate this."""
    p = pd.Series(np.asarray(p, dtype=float), index=df.index)
    won = df["is_winner"].astype(float)
    is_top = p.eq(p.groupby(df["race_id"]).transform("max"))
    return float(won[is_top].groupby(df.loc[is_top, "race_id"]).mean().mean())


def uniform_prob(df: pd.DataFrame) -> pd.Series:
    return 1.0 / df.groupby("race_id")["race_id"].transform("size")


def pseudo_r2(p: pd.Series, df: pd.DataFrame) -> float:
    """1 - model log loss / guessing (1/field) log loss. Benter's measure."""
    return 1.0 - per_race_log_loss(p, df) / per_race_log_loss(uniform_prob(df), df)


def calibration_table(p: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    p = pd.Series(np.asarray(p, dtype=float), index=df.index)
    bins = pd.cut(p, BINS, include_lowest=True)
    return (df.assign(_p=p, _bin=bins)
              .groupby("_bin", observed=True)
              .agg(n=("_p", "size"), mean_p=("_p", "mean"),
                   win_rate=("is_winner", lambda s: s.astype(float).mean())))


def race_days(df: pd.DataFrame) -> pd.Series | None:
    """race_id -> race day (date), used to resample whole race days. None if no race_date."""
    if "race_date" not in df.columns:
        return None
    return pd.to_datetime(df.groupby("race_id")["race_date"].first()).dt.normalize()


def paired_bootstrap(loss_a: pd.Series, loss_b: pd.Series, n: int = 2000,
                     seed: int = SEED, days: pd.Series | None = None) -> tuple[float, float, float]:
    """Mean of (a - b) per race and its 95% CI. Positive means b is better (lower loss).
    loss_a and loss_b are race-indexed; only shared races are used.

    days (race_id -> race day): resample whole race DAYS, the default in scoreboard() and
    rolling_validation since 6 Nov. Races on one day share conditions (going, track bias), so
    their results move together; resampling single races treats them as independent and gives
    intervals that are too narrow. This is the same method as Fahim's
    model_logic/validation/evaluation.paired_bootstrap_gain (race-weighted mean per resample).
    days=None: resample single races (the method used before 6 Nov)."""
    joined = pd.concat([loss_a.rename("a"), loss_b.rename("b")], axis=1, join="inner").dropna()
    d = joined["a"] - joined["b"]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if days is None:
        values = d.to_numpy()
        means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    else:
        day = days.reindex(d.index)
        if day.isna().any():
            raise ValueError("days is missing some races")
        by_day = d.groupby(day.to_numpy()).agg(["sum", "count"])
        pick = rng.integers(0, len(by_day), size=(n, len(by_day)))
        means = by_day["sum"].to_numpy()[pick].sum(axis=1) / by_day["count"].to_numpy()[pick].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


# ------------------------------------------------------------------- odds baselines
def _normalise(p: np.ndarray) -> np.ndarray:
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))


def power_probs(odds: np.ndarray) -> np.ndarray:
    """p proportional to (1/odds)^k, k solved so the race sums to 1 (favourite-longshot)."""
    q = 1.0 / np.clip(np.asarray(odds, dtype=float), 1.01, None)
    try:
        k = brentq(lambda k: (q ** k).sum() - 1.0, 0.1, 10.0)
    except ValueError:
        return _normalise(q)
    return q ** k


def shin_probs(odds: np.ndarray, tol: float = 1e-12, max_iter: int = 1000) -> np.ndarray:
    """Shin's method per race (insider-trading model of the overround)."""
    q = 1.0 / np.clip(np.asarray(odds, dtype=float), 1.01, None)
    n, B, z = len(q), q.sum(), 0.0
    if n <= 2:
        return _normalise(q)
    for _ in range(max_iter):
        z_new = (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / B).sum() - 2) / (n - 2)
        if abs(z_new - z) < tol:
            z = z_new
            break
        z = z_new
    p = (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / B) - z) / (2 * (1 - z))
    return _normalise(p)


def _per_race_apply(df: pd.DataFrame, odds_col: str, fn) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby("race_id").groups.items():
        out.loc[idx] = fn(df.loc[idx, odds_col].to_numpy())
    return out


def fit_market_refit_beta(p_market: pd.Series, df: pd.DataFrame) -> float:
    """beta for p proportional to market^beta, fitted by per-race log loss on THIS df.
    Only call on a fitting window (e.g. h_s2); on a held-out window pass beta from training."""
    lm = np.log(np.clip(np.asarray(p_market, float), EPSILON, None))
    lm = pd.Series(lm, index=df.index)

    def loss(beta: float) -> float:
        score = beta * lm
        e = np.exp(score - score.groupby(df["race_id"]).transform("max"))
        p = e / e.groupby(df["race_id"]).transform("sum")
        return per_race_log_loss(p, df)

    return float(minimize_scalar(loss, bounds=(0.3, 3.0), method="bounded").x)


def market_refit_prob(p_market: pd.Series, df: pd.DataFrame, beta: float) -> pd.Series:
    lm = pd.Series(np.log(np.clip(np.asarray(p_market, float), EPSILON, None)), index=df.index)
    score = beta * lm
    e = np.exp(score - score.groupby(df["race_id"]).transform("max"))
    return e / e.groupby(df["race_id"]).transform("sum")


def add_odds_baselines(df: pd.DataFrame, odds_col: str = "win_odds",
                       market_col: str = "p_market", beta: float | None = None) -> pd.DataFrame:
    """Add p_power, p_shin and p_market_refit. Needs win_odds for power/shin; if absent,
    those two are skipped with a note in df.attrs. beta: pass a training-fitted beta on a
    held-out window; leave None to fit on this df (only valid on a fitting window)."""
    out = df.copy()
    if odds_col in out.columns:
        out["p_power"] = _per_race_apply(out, odds_col, power_probs)
        out["p_shin"] = _per_race_apply(out, odds_col, shin_probs)
    else:
        out.attrs["baseline_note"] = f"{odds_col} missing; p_power and p_shin skipped"
    if market_col in out.columns:
        b = fit_market_refit_beta(out[market_col], out) if beta is None else beta
        out["p_market_refit"] = market_refit_prob(out[market_col], out, b)
        out.attrs["market_refit_beta"] = b
    return out


# ------------------------------------------------------------------------ scoreboard
def _complete_race_mask(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Races where every scored column is present for every runner and the market is ok."""
    ok = pd.Series(True, index=df.index)
    for col in cols:
        ok &= df[col].notna()
    if "market_complete" in df.columns:
        ok &= df["market_complete"].astype(bool)
    return ok.groupby(df["race_id"]).transform("all")


def scoreboard(df: pd.DataFrame, model_cols: list[str], baseline: str = "p_market_refit",
               raw_market: str = "p_market", final_run: bool = False,
               by_day: bool = True) -> pd.DataFrame:
    """One row per model: log loss, pseudo-R2, Brier, top-1, and the paired-bootstrap
    difference against the raw market and the refit baseline. All on the same races.
    by_day=True (default) resamples whole race days when race_date is present; the method
    used is recorded in table.attrs["bootstrap"]."""
    check_not_test(df, final_run=final_run)
    cols = [c for c in dict.fromkeys(model_cols + [baseline, raw_market]) if c in df.columns]
    scored = df.loc[_complete_race_mask(df, cols)].reset_index(drop=True)
    n_races = scored["race_id"].nunique()
    losses = {c: race_losses(scored[c], scored) for c in cols}
    days = race_days(scored) if by_day else None
    method = "race-day clusters" if days is not None else "single races"

    rows = []
    for col in model_cols:
        if col not in scored.columns:
            continue
        row = {
            "model": col,
            "races": n_races,
            "log_loss": per_race_log_loss(scored[col], scored),
            "pseudo_r2": pseudo_r2(scored[col], scored),
            "brier": brier(scored[col], scored),
            "top1": top1(scored[col], scored),
        }
        for name, base in [("vs_market", raw_market), ("vs_refit", baseline)]:
            if base in losses and base != col:
                mean, lo, hi = paired_bootstrap(losses[base], losses[col], days=days)
                row[f"{name}_gain"] = mean          # >0: this model beats the baseline
                row[f"{name}_lo"] = lo
                row[f"{name}_hi"] = hi
                row[f"{name}_beats"] = bool(lo > 0)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["n_races"] = n_races
    table.attrs["scored_columns"] = cols
    table.attrs["bootstrap"] = method
    table.attrs["n_days"] = int(days.nunique()) if days is not None else None
    return table


def verdict(row: pd.Series, name: str = "vs_refit") -> str:
    """Plain-words reading of a scoreboard row's CI against one baseline."""
    lo, hi = row.get(f"{name}_lo"), row.get(f"{name}_hi")
    if lo is None or not np.isfinite(lo):
        return "no comparison"
    g = row[f"{name}_gain"]
    if lo > 0:
        return f"better by {g:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})"
    if hi < 0:
        return f"worse by {g:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})"
    return f"no clear difference ({g:+.4f}, 95% CI {lo:+.4f} to {hi:+.4f})"
