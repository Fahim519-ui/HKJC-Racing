"""Parity coverage for the existing LightGBM rank-context transformation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from model_logic.feature_builder import build_features
from model_logic.lightgbm_independent_probability_v2 import add_context_features


ROOT = Path(__file__).resolve().parents[1]


def test_lightgbm_context_matches_v2_baseline() -> None:
    raw = pd.read_parquet(ROOT / "tests" / "fixtures" / "feature_builder_input_v2.parquet")
    feature_lists = json.loads(
        (ROOT / "tests" / "golden" / "feature_lists_v2.json").read_text(encoding="utf-8")
    )
    actual = build_features(raw)
    add_context_features(actual, feature_lists["conditional_logit"])
    expected = pd.read_parquet(ROOT / "tests" / "golden" / "lgbm_features_baseline.parquet")
    assert_frame_equal(actual, expected, check_exact=True)
