from pathlib import Path

import pandas as pd

from model_logic.feature_builder import build_features
from model_logic.feature_registry import (
    TRAINER_RATE_FEATURES,
    get_features,
    without_trainer_rates,
)
from model_logic.lightgbm_independent_probability_v2 import add_context_features


ROOT = Path(__file__).resolve().parents[1]


def test_trainer_rate_ablation_removes_only_trainer_rate_features() -> None:
    features = get_features("lgbm", available_columns={"jt_trainer_win365"})

    ablated = without_trainer_rates(features)

    assert not TRAINER_RATE_FEATURES.intersection(ablated)
    assert set(ablated) == set(features).difference(TRAINER_RATE_FEATURES)


def test_lightgbm_trainer_ablation_skips_trainer_rank_feature() -> None:
    raw = pd.read_parquet(ROOT / "tests" / "fixtures" / "feature_builder_input_v2.parquet")
    data = build_features(raw)
    features = without_trainer_rates(get_features("lgbm"))

    contextual = add_context_features(data, features, include_trainer_rates=False)

    assert "jt_trainer_win365_rel" not in contextual
    assert "jt_trainer_win365_rk" not in contextual
    assert "jt_trainer_win365_rk" not in data.columns
