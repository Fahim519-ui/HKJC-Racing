"""Leakage and construction tests for the v3.1 track/distance, trajectory and interaction features.

Each test changes the raw runners and checks which built values may move:

* same-day perturbation - shuffling one day's results must not move any v3 feature on that
  day or earlier, and must move some later ones (history is really used);
* cut-off - deleting every race after a date must leave features on/before it identical
  (catches shrinkage toward whole-period means, which a within-race shuffle cannot);
* source isolation - rewriting all Kaggle results must not move any HKJC feature;
* market exclusion - shuffling odds must not move any feature.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_logic import feature_builder, track_distance_features, trajectory_features
from model_logic.feature_builder import build_features
from model_logic.feature_registry import FORBIDDEN_INPUTS, REGISTRY, get_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean data" / "runners_merged.parquet"
DOC = ROOT / "docs" / "features_v3_track_distance.md"
KEY = ["source", "race_id", "horse_id"]
RESULT_COLS = [
    "finish_pos", "is_winner", "is_top3", "finish_time_s", "running_positions", "finish_status",
    "finish_code_raw", "finished", "dead_heat", "kaggle_fg_figure", "kaggle_path",
]
MARKET_COLS = ["win_odds", "implied_prob", "market_prob"]
V3 = [spec.name for spec in REGISTRY if spec.added_in == "v3.1" and spec.name != "td_distance_band"]
V3_MODEL = [spec.name for spec in REGISTRY if spec.added_in == "v3.1" and spec.models]
HISTORY = [name for name in V3 if name.startswith(("td_", "tr_fin", "tr_last"))]


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    if not DATA.exists():
        pytest.skip("runners_merged.parquet not found")
    frame = pd.read_parquet(DATA)
    frame["race_date"] = pd.to_datetime(frame["race_date"])
    return frame


@pytest.fixture(scope="module")
def base(raw) -> pd.DataFrame:
    return build_features(raw).set_index(KEY).sort_index()


def build(raw: pd.DataFrame) -> pd.DataFrame:
    return build_features(raw).set_index(KEY).sort_index()


def shuffle_results(raw: pd.DataFrame, mask: pd.Series, seed: int = 17) -> pd.DataFrame:
    """Permute result columns between started runners inside each selected race."""
    out = raw.copy()
    rng = np.random.default_rng(seed)
    rows = out.index[mask & out["started"].astype(bool)]
    for _, index in out.loc[rows].groupby("race_id").groups.items():
        index = np.asarray(index)
        if len(index) > 1:
            order = rng.permutation(index)
            out.loc[index, RESULT_COLS] = out.loc[order, RESULT_COLS].to_numpy()
    return out


def moved(before: pd.DataFrame, after: pd.DataFrame, columns: list[str]) -> pd.Series:
    common = before.index.intersection(after.index)
    a = before.loc[common, columns].to_numpy(float)
    b = after.loc[common, columns].to_numpy(float)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return pd.Series(~same.all(1), index=common)


def middle_day(raw: pd.DataFrame, source: str) -> pd.Timestamp:
    days = np.sort(raw.loc[raw["source"].eq(source), "race_date"].unique())
    return pd.Timestamp(days[len(days) // 2])


@pytest.mark.parametrize("source", ["kaggle", "hkjc"])
def test_one_days_results_only_move_later_days(raw, base, source):
    day = middle_day(raw, source)
    after = build(shuffle_results(raw, raw["source"].eq(source) & raw["race_date"].eq(day)))
    change = moved(base, after, V3)
    dates = base.loc[change.index, "race_date"]
    in_source = change.index.get_level_values("source") == source
    early = change[(dates <= day).to_numpy() | ~in_source]
    assert not early.any(), f"{early.sum()} rows on/before {day.date()} or in the other source moved"
    later_history = moved(base, after, HISTORY)[(dates > day).to_numpy() & in_source]
    assert later_history.any(), "no later feature moved: history is not being used"


@pytest.mark.parametrize("source", ["kaggle", "hkjc"])
def test_cutoff_leaves_earlier_features_identical(raw, base, source):
    cutoff = middle_day(raw, source)
    after = build(raw.loc[raw["source"].ne(source) | (raw["race_date"] <= cutoff)])
    change = moved(base, after, V3)
    assert len(change) > 0
    assert not change.any(), f"{change.sum()} rows changed when later races were removed"


def test_one_source_never_moves_the_other(raw, base):
    after = build(shuffle_results(raw, raw["source"].eq("kaggle")))
    change = moved(base, after, V3)
    hkjc = change.index.get_level_values("source") == "hkjc"
    assert not change[hkjc].any()
    assert change[~hkjc].any()


def test_odds_never_move_a_v3_feature(raw, base):
    shuffled = raw.copy()
    rng = np.random.default_rng(5)
    permuted = shuffled[MARKET_COLS].iloc[rng.permutation(len(raw))]
    shuffled[MARKET_COLS] = permuted.set_axis(shuffled.index)
    assert not moved(base, build(shuffled), V3).any()


def test_relative_features_are_centred_within_each_race(base):
    relative = [spec.name for spec in REGISTRY if spec.added_in == "v3.1" and spec.relative_to_race
                and spec.name != "int_jt_strength_x_field_size"]
    means = base.groupby("race_id")[relative].mean()
    assert np.nanmax(np.abs(means.to_numpy())) < 1e-9


def test_no_history_inherits_the_parent_level(base):
    # In races where no runner has a track x distance start, the combo estimate equals the
    # track estimate for every runner, so the race-relative versions coincide.
    races = base.groupby("race_id")["td_track_distance_starts"].transform("max").eq(0)
    assert races.sum() > 0
    for measure in ("win", "top3", "fin_pct"):
        np.testing.assert_allclose(
            base.loc[races, f"td_track_distance_{measure}_rel"],
            base.loc[races, f"td_track_{measure}_rel"],
            atol=1e-12,
        )


def test_class_direction_and_buckets_use_fixed_rules(base):
    assert set(base["tr_class_change_direction"].unique()) <= {-1.0, 0.0, 1.0}
    assert set(base["tr_days_since_bucket"].dropna().unique()) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    assert base.loc[base["hf_first_start"].eq(1), "tr_days_since_bucket"].isna().all()


def test_registry_model_sets_and_documentation():
    v3_lgbm, v3_cl = get_features("lgbm"), get_features("conditional_logit")
    text = DOC.read_text(encoding="utf-8")
    for name in V3_MODEL:
        assert name in v3_lgbm and name in v3_cl
        assert name not in get_features("lgbm", version="v2")
        assert name not in FORBIDDEN_INPUTS and not name.startswith(("mkt_", "market_", "odds_"))
        assert f"`{name}`" in text, f"{name} is not documented"
    assert get_features("conditional_logit", version="v2") == list(feature_builder.ALL_FEATURES)


def test_builder_needs_only_raw_runners():
    for module in (feature_builder, track_distance_features, trajectory_features):
        assert "version2_runners_merged" not in inspect.getsource(module)
