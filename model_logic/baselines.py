"""Frozen odds-free historical baseline definitions.

These lists intentionally are not derived from the active registry sets: those sets include
later partnership, track/distance, trajectory, interaction, and rank features.  The historical
baselines must retain their original feature universe for a stable ladder comparison.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from model_logic.feature_registry import FORBIDDEN_INPUTS, REGISTRY


DRAW_ONLY_FEATURES = ("rc_draw_pct", "rc_draw_x_hv", "rc_draw_x_sprint")
ORIGINAL_CL_BASE = (
    "jt_jockey_win365_rel",
    "hf_avg3_fin_pct_rel",
    "rc_draw_pct",
    "rc_draw_x_hv",
    "rc_weight_rel",
    "hf_days_since",
)
ORIGINAL_CL_CANDIDATES = (
    "hf_last_fin_pct_rel",
    "jt_trainer_win365_rel",
    "hf_win5_s",
    "hf_first_start",
    "rc_draw_x_sprint",
    "rc_weight_x_dist",
    "hf_class_change",
    "w3_horse_past_win_rate",
    "w3_horse_past_top3_rate",
)
BASELINE_FEATURES = DRAW_ONLY_FEATURES + ORIGINAL_CL_BASE + ORIGINAL_CL_CANDIDATES


def validate_baseline_features(registry: Iterable = REGISTRY) -> None:
    """Ensure frozen baseline lists remain implemented and strictly odds-free."""
    specs = {spec.name: spec for spec in registry}
    missing = [feature for feature in BASELINE_FEATURES if feature not in specs]
    unimplemented = [
        feature for feature in BASELINE_FEATURES
        if feature in specs and not specs[feature].implemented
    ]
    forbidden = [
        feature for feature in BASELINE_FEATURES
        if feature in FORBIDDEN_INPUTS
        or feature.startswith(("mkt_", "market_", "odds_"))
        or "market" in feature
        or "odds" in feature
    ]
    if missing or unimplemented or forbidden:
        raise ValueError(
            "Invalid frozen baseline features: "
            f"missing={missing}, unimplemented={unimplemented}, forbidden={forbidden}"
        )


def fit_predict_draw_only(
    inner_train: pd.DataFrame,
    inner_tune: pd.DataFrame,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[dict, list[pd.DataFrame]]:
    """Fit the unpenalised draw-only historical conditional logit."""
    from model_logic.run_folds import fit_conditional_logit

    result, predictions = fit_conditional_logit(
        inner_train,
        inner_tune,
        s1,
        [s2, valid],
        list(DRAW_ONLY_FEATURES),
        list(DRAW_ONLY_FEATURES),
        minimum_gain=0.0,
        selected_features=list(DRAW_ONLY_FEATURES),
        fixed_penalty=0.0,
    )
    return result, [prediction.rename(columns={"p_cl": "p_draw_only"}) for prediction in predictions]


def fit_predict_original_cl(
    inner_train: pd.DataFrame,
    inner_tune: pd.DataFrame,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    valid: pd.DataFrame,
    minimum_gain: float,
) -> tuple[dict, list[pd.DataFrame]]:
    """Select on s1's inner split, then refit the frozen historical CL unpenalised."""
    from model_logic.run_folds import fit_conditional_logit

    features = list(ORIGINAL_CL_BASE + ORIGINAL_CL_CANDIDATES)
    result, predictions = fit_conditional_logit(
        inner_train,
        inner_tune,
        s1,
        [s2, valid],
        features,
        list(ORIGINAL_CL_BASE),
        minimum_gain,
        fixed_penalty=0.0,
    )
    return result, [prediction.rename(columns={"p_cl": "p_original_cl"}) for prediction in predictions]
