"""Leakage tests for the shared feature builder (Week 4 roadmap, step 2).

What is tested (rules from DATA_RULES.md and WEEK3-4_PLAN.md section 5.2):
  1. Same-day exclusion: shuffling the results of one race day changes no feature on that
     day or before it. Jockey/trainer rates are equal for all of a person's rides on a day.
  2. Source separation: shuffling every result in one source (kaggle / hkjc) changes no
     feature in the other source. Each person's first day in a source gets exactly the prior.
  3. No post-race or market columns: no feature name is on the forbidden list, odds changes
     move no feature, and rows from invalid races or non-starters never feed history.
  4. Cut-off stability: dropping every race after a mid-season cut-off and rebuilding leaves
     every earlier-dated feature unchanged. This catches shrinkage toward a whole-period mean
     (e.g. a source average that includes h_test), which the shuffle tests in 1-3 cannot.

Checks 1-4 run against BOTH builders a live model could use: the one the models run today
(conditional_logit_tuning_v2.build_features, "cl_v2") and Fahim's feature_builder.py ("fb").
The feature list is taken from feature_registry joined with the scripts' own lists, so new
registered features are picked up automatically.

Each check also runs on a deliberately leaky feature (a "negative control") to prove the
check can fail. A check that cannot fail proves nothing.

The tests never compute a metric, and never read h_test results.

Run from the repo root (venv active, pytest 9.2.0):
    python -m pytest "leakage test/test_leakage.py" -v
Data: data/clean data/runners_merged.parquet, or set HKJC_DATA to another path.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
LOGIC = REPO / "model_logic"
ARTIFACTS = REPO / "artifacts"
KEY = ["source", "race_id", "horse_id"]
PRIOR_WIN = 1 / 12

# Never features (DATA_RULES.md). Matched exactly, or by prefix.
FORBIDDEN = {
    "is_winner", "is_top3", "running_positions", "race_winner_time_s", "kaggle_fg_figure",
    "kaggle_path", "dead_heat", "finished", "win_odds", "implied_prob", "market_prob",
    "odds_capped", "profile_missing", "early_position", "position_change",
}
FORBIDDEN_PREFIXES = ("finish_", "mkt_", "market_")

# Post-race columns shuffled between runners of the same race. Shuffling them together
# keeps each result internally consistent (winner, placings, times move as one).
RESULT_COLS = [
    "finish_pos", "is_winner", "is_top3", "finish_time_s", "running_positions",
    "finish_status", "finish_code_raw", "finished", "dead_heat", "kaggle_fg_figure",
    "kaggle_path",
]
MARKET_COLS = ["win_odds", "implied_prob", "market_prob"]


# --------------------------------------------------------------------------- loading
def _load(name: str):
    for folder in (str(REPO), str(LOGIC)):          # modules import each other
        if folder not in sys.path:
            sys.path.insert(0, folder)
    path = LOGIC / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _data_path() -> Path:
    candidates = [Path(os.environ["HKJC_DATA"])] if "HKJC_DATA" in os.environ else []
    candidates += [REPO / "data/clean data/runners_merged.parquet",
                   REPO / "data/processed/runners_merged.parquet"]
    for path in candidates:
        if path.exists():
            return path
    pytest.skip("runners_merged.parquet not found; set HKJC_DATA")


@lru_cache(maxsize=1)
def modules():
    return _load("conditional_logit_tuning_v2"), _load("lightgbm_independent_probability_v2")


# The two builders that the live models can run. "cl_v2" is what both models use today
# (conditional_logit_tuning_v2.build_features). "fb" is Fahim's newer feature_builder.py,
# which the parity test proves is numerically identical and which the models will switch to.
# Every check runs against both, so whichever the models use is always covered. When the
# models move fully to feature_builder.py, drop the "cl_v2" entry from BUILDERS.
BUILDER_NAMES = ("cl_v2", "fb")


def _add_context(data: pd.DataFrame) -> pd.DataFrame:
    """Append the v2 LightGBM rank-context features in place, as the models do."""
    _, lgb2 = modules()
    cl, _ = modules()
    lgb2.add_context_features(data, list(cl.ALL_FEATURES))
    return data


def make_builder(name: str):
    """Return a build function that produces every column a model could feed in."""
    if name == "cl_v2":
        cl, _ = modules()
        return lambda raw: _add_context(cl.build_features(raw))
    if name == "fb":
        fb = _load("feature_builder")
        return lambda raw: _add_context(fb.build_features(raw))
    raise ValueError(f"unknown builder {name}")


def fahim_builder(raw: pd.DataFrame) -> pd.DataFrame:
    """Back-compat alias used by the negative controls: the builder the models run today."""
    return make_builder("cl_v2")(raw)


@lru_cache(maxsize=1)
def model_features() -> tuple[str, ...]:
    """Every feature any model can use: the registry's model sets joined with what the
    scripts' own lists and context helper actually produce."""
    cl, lgb2 = modules()
    v1 = _load("lightgbm_independent_probability")
    names: set[str] = set(cl.ALL_FEATURES) | set(v1.FEATURES)
    try:
        reg = _load("feature_registry")
        names |= set(reg.get_features("all")) | set(reg.get_features("lgbm"))
    except Exception:
        pass
    # add the rank-context columns the lgbm helper creates
    raw = pd.read_parquet(_data_path())
    built = _add_context(cl.build_features(raw))
    names |= {c for c in built.columns if c.endswith("_rk")}
    # keep only columns at least one builder actually produces, so checks never compare NaN
    return tuple(sorted(n for n in names if n in built.columns))


# ---------------------------------------------------------------------- perturbation
def shuffle_results(raw: pd.DataFrame, mask: pd.Series, starters_only: bool = True,
                    seed: int = 17) -> pd.DataFrame:
    """Give each selected race a different result: permute the result columns between its
    runners. Pre-race columns (horse, jockey, draw, weight, date) are not touched."""
    out = raw.copy()
    keep = mask.to_numpy(bool)
    if starters_only:
        keep &= out["started"].fillna(False).astype(bool).to_numpy()
    pos = np.flatnonzero(keep)
    assert len(pos) > 0, "nothing selected to perturb"
    rng = np.random.default_rng(seed)
    race = out["race_id"].to_numpy()[pos]
    in_order = np.lexsort((np.arange(len(pos)), race))     # rows grouped by race
    shuffled = np.lexsort((rng.random(len(pos)), race))     # same races, random order
    take = np.arange(len(out))
    take[pos[in_order]] = pos[shuffled]                     # never crosses a race
    for col in RESULT_COLS:
        if col in out.columns:                              # take() keeps the dtype
            out[col] = out[col].take(take).set_axis(out.index)
    if "race_winner_time_s" in out.columns:
        rows = out.index[pos]
        out.loc[rows, "race_winner_time_s"] = out.loc[rows, "race_winner_time_s"] + 1.0
    return out


def shuffle_market(raw: pd.DataFrame, seed: int = 17) -> pd.DataFrame:
    out = raw.copy()
    factor = np.random.default_rng(seed).uniform(0.5, 2.0, len(out))
    for col in MARKET_COLS:
        out[col] = out[col].astype("float64") * factor
    out["odds_capped"] = ~out["odds_capped"].fillna(False).astype(bool)
    return out


def changed(base: pd.DataFrame, new: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Boolean frame (rows by KEY, columns = feats): True where the feature value moved."""
    a = base.set_index(KEY)[feats].astype("float64")
    b = new.set_index(KEY)[feats].astype("float64")
    assert len(a) == len(b) and a.index.is_unique and set(a.index) == set(b.index), \
        "perturbation changed which rows are modelled"
    b = b.reindex(a.index)
    moved = ~np.isclose(a.to_numpy(), b.to_numpy(), rtol=0.0, atol=1e-9, equal_nan=True)
    return pd.DataFrame(moved, index=a.index, columns=feats)


def offenders(moved: pd.DataFrame, rows: np.ndarray) -> dict[str, int]:
    counts = moved[rows].sum()
    return {feature: int(n) for feature, n in counts.items() if n}


def one_day_report(builder, feats, raw, source, day) -> dict:
    """Shuffle every result on one race day of one source, rebuild, and report what moved."""
    base = builder(raw)
    mask = raw["source"].eq(source) & pd.to_datetime(raw["race_date"]).eq(day)
    moved = changed(base, builder(shuffle_results(raw, mask)), feats)
    date = base.set_index(KEY)["race_date"].reindex(moved.index).to_numpy()
    src = moved.index.get_level_values("source").to_numpy()
    same = src == source
    return {
        "same_source_on_or_before_day": offenders(moved, same & (date <= day)),
        "other_source": offenders(moved, ~same),
        "later_rows_changed": int(moved[same & (date > day)].any(axis=1).sum()),
    }


def source_report(builder, feats, raw, perturbed_source) -> dict:
    """Shuffle every result in one source and report what moved in the other."""
    base = builder(raw)
    moved = changed(base, builder(shuffle_results(raw, raw["source"].eq(perturbed_source))), feats)
    src = moved.index.get_level_values("source").to_numpy()
    return {
        "other_source": offenders(moved, src != perturbed_source),
        "perturbed_source_rows_changed": int(moved[src == perturbed_source].any(axis=1).sum()),
    }


def with_extra(name: str, make):
    """A builder that adds one extra (leaky) feature - for negative controls."""
    def builder(raw):
        data = fahim_builder(raw)
        data[name] = make(data)
        return data
    return builder


def expanding_rate(data, keys, sort):
    ordered = data.sort_values(sort, kind="stable")
    return ordered.groupby(keys)["is_winner"].transform(lambda s: s.shift().expanding().mean())


LEAK_SAME_RACE = with_extra("leak_same_race", lambda d: d["is_winner"].astype(float))
LEAK_SAME_DAY = with_extra("leak_same_day", lambda d: expanding_rate(
    d, ["source", "jockey_key"], ["source", "race_date", "race_no"]))
LEAK_CROSS_SOURCE = with_extra("leak_cross_source", lambda d: expanding_rate(
    d, ["jockey_key"], ["race_date", "race_no"]))
LEAK_FUTURE = with_extra("leak_future", lambda d: d.sort_values(["source", "race_date"])
                         .groupby(["source", "horse_id"])["is_winner"].shift(-1))
LEAK_ODDS = with_extra("leak_odds", lambda d: d["mkt_logit"])


LEAK_WHOLE_PERIOD_MEAN = with_extra(
    "leak_whole_period_mean",
    # Shrink a horse's lagged mean toward the whole-source average (includes h_test),
    # exactly the pattern in Nayeon's td_* features. Same-day shuffles cannot catch it;
    # the cut-off test can, because dropping later races changes the source mean.
    lambda d: (
        d.sort_values(["source", "horse_id", "race_date"])
         .groupby(["source", "horse_id"])["is_winner"].shift(1).fillna(0.0)
        + 5 * d.groupby("source")["is_winner"].transform("mean")
    ) / 6.0,
)


def leak_excluded_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Control: a jockey win count that wrongly includes invalid races and non-starters."""
    data = fahim_builder(raw)
    bad = raw.loc[~(raw["race_valid"].fillna(False).astype(bool)
                    & raw["started"].fillna(False).astype(bool))].copy()
    bad["race_date"] = pd.to_datetime(bad["race_date"])
    bad["w"] = bad["is_winner"].fillna(False).astype(float)
    wins = bad.groupby(["source", "jockey_key", "race_date"])["w"].sum().rename("wins").reset_index()
    merged = data[["source", "jockey_key", "race_date"]].reset_index().merge(
        wins, on=["source", "jockey_key"], how="left")
    merged = merged[merged["race_date_y"] < merged["race_date_x"]]
    data["leak_excluded"] = merged.groupby("index")["wins"].sum().reindex(data.index).fillna(0.0)
    return data


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_parquet(_data_path())


@pytest.fixture(scope="module")
def base(raw) -> pd.DataFrame:
    return fahim_builder(raw)


@pytest.fixture(scope="module")
def feats() -> list[str]:
    return list(model_features())


def middle_day(base, source, split) -> pd.Timestamp:
    days = np.sort(base.loc[base["source"].eq(source) & base["split"].eq(split), "race_date"].unique())
    return pd.Timestamp(days[len(days) // 2])


# ------------------------------------------------------- 1. same-day exclusion
@pytest.mark.parametrize("builder_name", BUILDER_NAMES)
@pytest.mark.parametrize("source, split", [("hkjc", "h_s2"), ("kaggle", "k_train")])
def test_one_days_results_only_move_later_days(raw, base, feats, source, split, builder_name):
    builder = make_builder(builder_name)
    day = middle_day(base, source, split)
    report = one_day_report(builder, feats, raw, source, day)
    assert report["same_source_on_or_before_day"] == {}, f"{builder_name} {day.date()}: {report}"
    assert report["other_source"] == {}, f"{builder_name} {day.date()}: {report}"
    assert report["later_rows_changed"] > 0, "nothing moved: the perturbation was not seen"


@pytest.mark.parametrize("column, key", [("jt_jockey_win365", "jockey_key"),
                                         ("jt_trainer_win365", "trainer_key")])
def test_person_rate_is_equal_for_all_rides_on_a_day(base, column, key):
    per_day = base.groupby(["source", key, "race_date"])[column].nunique(dropna=False)
    assert (per_day <= 1).all(), per_day[per_day > 1].head().to_dict()


def test_horse_runs_at_most_once_a_day(base):
    # build_features uses groupby(horse).shift(1); that is only safe if this holds.
    assert not base.duplicated(["source", "horse_id", "race_date"]).any()


# --------------------------------------------------------- 2. source separation
@pytest.mark.parametrize("builder_name", BUILDER_NAMES)
@pytest.mark.parametrize("perturbed", ["kaggle", "hkjc"])
def test_one_source_never_moves_the_other(raw, feats, perturbed, builder_name):
    report = source_report(make_builder(builder_name), feats, raw, perturbed)
    assert report["other_source"] == {}, f"{builder_name}: {report}"
    assert report["perturbed_source_rows_changed"] > 0


@pytest.mark.parametrize("column, key", [("jt_jockey_win365", "jockey_key"),
                                         ("jt_trainer_win365", "trainer_key")])
def test_first_day_in_each_source_gets_the_prior(base, column, key):
    first_day = base.groupby(["source", key])["race_date"].transform("min")
    values = base.loc[base["race_date"].eq(first_day), column]
    assert np.allclose(values, PRIOR_WIN), values[~np.isclose(values, PRIOR_WIN)].head()


def test_people_shared_across_sources_exist(base):
    # Without shared people, the source-separation test above could not fail.
    shared = base.groupby("jockey_key")["source"].nunique()
    assert (shared > 1).sum() >= 10


# ------------------------------------------------ 3. no post-race or market columns
def _bad(name: str) -> bool:
    return name in FORBIDDEN or name.startswith(FORBIDDEN_PREFIXES)


def test_no_forbidden_column_is_a_model_feature(feats):
    assert [f for f in feats if _bad(f)] == []


@pytest.mark.parametrize("artifact", [
    "conditional_logit_tuning_v2/selected_features.json",
    "lightgbm_independent_probability_v2/stage1_features.json",
])
def test_saved_artifact_features_are_clean_and_covered(feats, artifact):
    path = ARTIFACTS / artifact
    if not path.exists():
        pytest.skip(f"{artifact} not found")
    used = json.loads(path.read_text(encoding="utf-8"))
    assert [f for f in used if _bad(f)] == [], "artifact used a forbidden column"
    assert set(used) <= set(feats), f"not covered by these tests: {set(used) - set(feats)}"


@pytest.mark.parametrize("builder_name", BUILDER_NAMES)
def test_odds_never_move_a_stage1_feature(raw, feats, builder_name):
    builder = make_builder(builder_name)
    moved = changed(builder(raw), builder(shuffle_market(raw)), feats)
    assert offenders(moved, np.ones(len(moved), bool)) == {}, builder_name


@pytest.mark.parametrize("builder_name", BUILDER_NAMES)
def test_invalid_races_and_non_starters_never_feed_history(raw, feats, builder_name):
    builder = make_builder(builder_name)
    excluded = ~(raw["race_valid"].fillna(False).astype(bool)
                 & raw["started"].fillna(False).astype(bool))
    moved = changed(builder(raw), builder(shuffle_results(raw, excluded, starters_only=False)), feats)
    assert offenders(moved, np.ones(len(moved), bool)) == {}, builder_name


# ----------------------------------- 4. cut-off test (catches whole-period shrinkage)
# The shuffle tests above cannot catch a feature that shrinks toward a mean taken over the
# whole source (including h_test), because shuffling results within races leaves each
# source's mean unchanged. This test instead drops every race after a mid-season cut-off
# and rebuilds: any feature dated on or before the cut-off that uses only earlier data must
# be unchanged. A feature that peeks at later races (e.g. Nayeon's td_* source-mean
# shrinkage) moves and is caught.
CUTOFFS = {"kaggle": "2019-02-01", "hkjc": "2026-01-15"}


def cutoff_report(builder, feats, raw, source) -> dict:
    cutoff = pd.Timestamp(CUTOFFS[source])
    full = builder(raw)
    rawd = raw.copy()
    rawd["race_date"] = pd.to_datetime(rawd["race_date"])
    # keep all of the OTHER source, and this source only up to the cut-off
    keep = rawd["source"].ne(source) | (rawd["race_date"] <= cutoff)
    truncated = builder(rawd.loc[keep].copy())
    f = full.set_index(KEY)
    t = truncated.set_index(KEY)
    common = f.index.intersection(t.index)
    cols = [c for c in feats if c in f.columns and c in t.columns]
    a = f.loc[common, cols].astype("float64")
    b = t.loc[common, cols].astype("float64")
    moved = pd.DataFrame(
        ~np.isclose(a.to_numpy(), b.to_numpy(), rtol=0.0, atol=1e-9, equal_nan=True),
        index=common, columns=cols,
    )
    date = pd.to_datetime(f.loc[common, "race_date"]).to_numpy()
    src = np.array([k[0] for k in common])
    on_or_before = (src == source) & (date <= cutoff.to_datetime64())
    return {
        "offenders_on_or_before_cutoff": offenders(moved, on_or_before),
        "rows_checked": int(on_or_before.sum()),
    }


@pytest.mark.parametrize("builder_name", BUILDER_NAMES)
@pytest.mark.parametrize("source", ["kaggle", "hkjc"])
def test_cutoff_earlier_features_are_stable(raw, feats, source, builder_name):
    report = cutoff_report(make_builder(builder_name), feats, raw, source)
    assert report["rows_checked"] > 0, "no rows on/before the cut-off; check the date"
    assert report["offenders_on_or_before_cutoff"] == {}, f"{builder_name}/{source}: {report}"


# ---------------------------------------------- negative controls (checks can fail)
def test_control_same_race_leak_is_caught(raw, base, feats):
    day = middle_day(base, "hkjc", "h_s2")
    report = one_day_report(LEAK_SAME_RACE, feats + ["leak_same_race"], raw, "hkjc", day)
    assert "leak_same_race" in report["same_source_on_or_before_day"]


def test_control_same_day_leak_is_caught(raw, base, feats):
    day = middle_day(base, "hkjc", "h_s2")
    report = one_day_report(LEAK_SAME_DAY, feats + ["leak_same_day"], raw, "hkjc", day)
    assert "leak_same_day" in report["same_source_on_or_before_day"]


def test_control_cross_source_leak_is_caught(raw, feats):
    report = source_report(LEAK_CROSS_SOURCE, feats + ["leak_cross_source"], raw, "kaggle")
    assert "leak_cross_source" in report["other_source"]


def test_control_look_ahead_leak_is_caught(raw, base, feats):
    day = middle_day(base, "hkjc", "h_s2")
    report = one_day_report(LEAK_FUTURE, feats + ["leak_future"], raw, "hkjc", day)
    assert "leak_future" in report["same_source_on_or_before_day"]


def test_control_odds_leak_is_caught(raw, feats):
    leaky = feats + ["leak_odds"]
    moved = changed(LEAK_ODDS(raw), LEAK_ODDS(shuffle_market(raw)), leaky)
    assert "leak_odds" in offenders(moved, np.ones(len(moved), bool))


def test_control_excluded_rows_leak_is_caught(raw, feats):
    excluded = ~(raw["race_valid"].fillna(False).astype(bool)
                 & raw["started"].fillna(False).astype(bool))
    leaky = feats + ["leak_excluded"]
    moved = changed(leak_excluded_rows(raw),
                    leak_excluded_rows(shuffle_results(raw, excluded, starters_only=False)), leaky)
    assert "leak_excluded" in offenders(moved, np.ones(len(moved), bool))


def test_control_whole_period_mean_leak_is_caught(raw, feats):
    report = cutoff_report(LEAK_WHOLE_PERIOD_MEAN, feats + ["leak_whole_period_mean"], raw, "kaggle")
    assert "leak_whole_period_mean" in report["offenders_on_or_before_cutoff"], report


def test_control_forbidden_name_is_caught():
    assert _bad("finish_pos") and _bad("mkt_logit") and _bad("position_change")
    assert not _bad("hf_last_fin_pct_rel")
