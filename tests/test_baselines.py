"""Synthetic contracts for frozen odds-free baseline rungs and their ladder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import model_logic.run_baselines as command
import model_logic.run_folds as runner
from model_logic.aggregate_folds import LADDER, aggregate_folds
from model_logic.baselines import (
    DRAW_ONLY_FEATURES,
    ORIGINAL_CL_BASE,
    ORIGINAL_CL_CANDIDATES,
    fit_predict_draw_only,
    fit_predict_original_cl,
)


ALL_FEATURES = DRAW_ONLY_FEATURES + ORIGINAL_CL_BASE + ORIGINAL_CL_CANDIDATES


def _frame(prefix: str, start: str, races: int, planted_draw: bool = True) -> pd.DataFrame:
    rows = []
    for race in range(races):
        for horse in range(3):
            draw = horse + 1
            winner = horse == 0 if planted_draw else horse == race % 3
            rows.append(
                {
                    "race_id": f"{prefix}{race}",
                    "horse_id": f"h{horse}",
                    "race_date": pd.Timestamp(start) + pd.Timedelta(days=7 * race),
                    "n_runners": 3.0,
                    "draw": float(draw),
                    "is_winner": float(winner),
                    "finish_pos": float(1 if winner else horse + 2),
                    "rc_draw_pct": horse / 2,
                    "rc_draw_x_hv": 0.0,
                    "rc_draw_x_sprint": horse / 2,
                    "jt_jockey_win365_rel": horse / 10,
                    "hf_avg3_fin_pct_rel": horse / 2,
                    "rc_weight_rel": float(horse - 1),
                    "hf_days_since": float(horse + 1),
                    "hf_last_fin_pct_rel": horse / 2,
                    "jt_trainer_win365_rel": horse / 3,
                    "hf_win5_s": horse / 3,
                    "hf_first_start": 0.0,
                    "rc_weight_x_dist": float(horse - 1),
                    "hf_class_change": 0.0,
                    "w3_horse_past_win_rate": horse / 3,
                    "w3_horse_past_top3_rate": horse / 3,
                    "market_prob": [0.2, 0.3, 0.5][horse],
                    "mkt_p": [0.2, 0.3, 0.5][horse],
                    "mkt_hidden": float(horse),
                    "td_signal": float(horse),
                    "tr_signal": float(horse),
                    "jtp_signal": float(horse),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def periods() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inner_train = _frame("train", "2020-01-01", 6)
    inner_tune = _frame("tune", "2020-03-01", 3)
    s1 = pd.concat([inner_train, inner_tune], ignore_index=True)
    return inner_train, inner_tune, s1, _frame("s2", "2020-04-01", 2), _frame("valid", "2020-05-01", 3)


def test_draw_only_ignores_every_non_draw_column(periods) -> None:
    inner_train, inner_tune, s1, s2, valid = periods
    _, first = fit_predict_draw_only(inner_train, inner_tune, s1, s2, valid)
    changed = [
        frame.assign(**{
            column: np.arange(len(frame), dtype=float) * 100
            for column in frame.columns
            if column not in {"race_id", "horse_id", "race_date", "n_runners", "is_winner", "finish_pos"}
            and column not in DRAW_ONLY_FEATURES
        })
        for frame in (inner_train, inner_tune, s1, s2, valid)
    ]
    _, second = fit_predict_draw_only(*changed)
    assert_frame_equal(first[0], second[0], check_exact=True)
    assert_frame_equal(first[1], second[1], check_exact=True)


def test_baselines_ignore_market_columns(periods) -> None:
    inner_train, inner_tune, s1, s2, valid = periods
    _, draw_first = fit_predict_draw_only(inner_train, inner_tune, s1, s2, valid)
    original_first, original_predictions = fit_predict_original_cl(
        inner_train, inner_tune, s1, s2, valid, 0.001
    )
    changed = [
        frame.assign(market_prob=1.0, mkt_p=1.0, mkt_hidden=-999.0)
        for frame in (inner_train, inner_tune, s1, s2, valid)
    ]
    _, draw_second = fit_predict_draw_only(*changed)
    original_second, changed_predictions = fit_predict_original_cl(*changed, 0.001)
    assert_frame_equal(draw_first[0], draw_second[0], check_exact=True)
    assert_frame_equal(draw_first[1], draw_second[1], check_exact=True)
    assert original_first["selected"] == original_second["selected"]
    assert_frame_equal(original_first["coefficients"], original_second["coefficients"], check_exact=True)
    assert_frame_equal(original_predictions[1], changed_predictions[1], check_exact=True)


def test_original_cl_cannot_select_later_feature_families(periods) -> None:
    inner_train, inner_tune, s1, s2, valid = periods
    result, _ = fit_predict_original_cl(inner_train, inner_tune, s1, s2, valid, 0.001)
    assert set(result["selected"]).issubset(set(ORIGINAL_CL_BASE + ORIGINAL_CL_CANDIDATES))
    assert not any(feature.startswith(("td_", "tr_", "jtp_")) for feature in result["selected"])


def test_baseline_selection_and_predictions_ignore_s2_and_valid_outcomes(periods) -> None:
    inner_train, inner_tune, s1, s2, valid = periods
    first, predictions = fit_predict_original_cl(inner_train, inner_tune, s1, s2, valid, 0.001)
    changed_s2 = s2.assign(is_winner=lambda frame: 1.0 - frame.is_winner, finish_pos=99.0)
    blank_valid = valid.assign(is_winner=np.nan, finish_pos=np.nan)
    second, changed_predictions = fit_predict_original_cl(
        inner_train, inner_tune, s1, changed_s2, blank_valid, 0.001
    )
    assert first["selected"] == second["selected"]
    assert_frame_equal(first["coefficients"], second["coefficients"], check_exact=True)
    assert_frame_equal(predictions[1], changed_predictions[1], check_exact=True)


def test_original_cl_selection_uses_only_inner_split(periods, monkeypatch) -> None:
    inner_train, inner_tune, s1, s2, valid = periods
    final_only = _frame("final", "2020-06-01", 2).assign(td_signal=999.0)
    monkeypatch.setattr(runner, "fit_beta", lambda races, **_: np.zeros(races.X.shape[1]))
    first, _ = fit_predict_original_cl(inner_train, inner_tune, pd.concat([s1, final_only]), s2, valid, 0.001)
    changed = final_only.copy()
    changed.loc[:, list(dict.fromkeys(ALL_FEATURES))] = -999.0
    second, _ = fit_predict_original_cl(
        inner_train, inner_tune, pd.concat([s1, changed]), s2, valid, 0.001
    )
    assert first["selected"] == second["selected"]


def test_baseline_probabilities_are_finite_and_sum_to_one(periods) -> None:
    inner_train, inner_tune, s1, s2, valid = periods
    _, draw_predictions = fit_predict_draw_only(inner_train, inner_tune, s1, s2, valid)
    _, original_predictions = fit_predict_original_cl(inner_train, inner_tune, s1, s2, valid, 0.001)
    for frame, prediction in zip((s2, valid), draw_predictions, strict=True):
        assert np.isfinite(prediction["p_draw_only"]).all()
        assert np.allclose(prediction.groupby(frame["race_id"])["p_draw_only"].sum(), 1.0)
    for frame, prediction in zip((s2, valid), original_predictions, strict=True):
        assert np.isfinite(prediction["p_original_cl"]).all()
        assert np.allclose(prediction.groupby(frame["race_id"])["p_original_cl"].sum(), 1.0)


def test_draw_signal_beats_uniform_on_valid(periods) -> None:
    inner_train, inner_tune, s1, s2, valid = periods
    _, predictions = fit_predict_draw_only(inner_train, inner_tune, s1, s2, valid)
    draw_loss = -np.log(
        predictions[1].loc[valid["is_winner"].eq(1), "p_draw_only"].to_numpy()
    ).mean()
    uniform_loss = np.log(3)
    assert draw_loss < uniform_loss


def _write_ladder_artifacts(root: Path, fold_id: str, config: str = "primary") -> None:
    config_directory = root / config / fold_id
    baseline_directory = root / "_baselines" / fold_id
    config_directory.mkdir(parents=True)
    baseline_directory.mkdir(parents=True)
    valid = _frame(fold_id, "2020-01-01", 2)
    probabilities = {
        "p_market": (0.6, 0.2),
        "p_market_calibrated": (0.6, 0.2),
        "p_cl": (0.55, 0.25),
        "p_lgbm": (0.54, 0.26),
        "p_cl_only": (0.55, 0.25),
        "p_lgbm_only": (0.54, 0.26),
        "p_mkt+cl": (0.56, 0.24),
        "p_mkt+lgbm": (0.57, 0.23),
        "p_mkt+cl+lgbm": (0.58, 0.22),
        "p_uniform": (1 / 3, 1 / 3),
    }
    prediction = valid[["race_id", "horse_id", "race_date", "is_winner"]].copy()
    prediction["role"] = "valid"
    for column, (winner, other) in probabilities.items():
        prediction[column] = np.where(prediction["is_winner"].eq(1), winner, other)
    prediction.to_parquet(config_directory / "predictions.parquet", index=False)
    baseline = prediction[["race_id", "horse_id", "role", "is_winner", "p_uniform"]].copy()
    baseline["p_draw_only"] = np.where(baseline["is_winner"].eq(1), 0.5, 0.25)
    baseline["p_original_cl"] = np.where(baseline["is_winner"].eq(1), 0.55, 0.225)
    baseline.to_parquet(baseline_directory / "baseline_predictions.parquet", index=False)
    provenance = {
        "data_hash": "data",
        "registry_hash": "registry",
        "git_commit": "commit",
        "primary": config == "primary",
        "unavailable_exclusions": [],
        "fold": {"era": "kaggle"},
        "race_counts": {"inner_tune": 200},
    }
    (config_directory / "meta.json").write_text(json.dumps(provenance))
    (baseline_directory / "meta.json").write_text(json.dumps(provenance))
    (config_directory / "cl_selected_features.json").write_text(json.dumps(["rc_draw_pct"]))


def test_ladder_has_every_rung_on_identical_races(tmp_path) -> None:
    _write_ladder_artifacts(tmp_path, "k1")
    aggregate_folds(tmp_path)
    ladder = pd.read_csv(tmp_path / "primary" / "ladder.csv")
    per_fold = ladder.loc[ladder["fold_id"].eq("k1")]
    assert per_fold["variant"].tolist() == [variant for _, variant, _ in LADDER]
    assert per_fold["rung_order"].tolist() == list(range(1, 11))
    assert per_fold["n_races"].nunique() == 1
    assert ladder["is_primary"].sum() == 1


def test_baseline_resume_key_requires_all_inputs(tmp_path, monkeypatch) -> None:
    folds = pd.DataFrame([{
        "fold_id": "f1", "era": "kaggle",
        "s1_start": pd.Timestamp("2020-01-01"), "s1_end": pd.Timestamp("2020-01-31"),
        "s2_start": pd.Timestamp("2020-02-01"), "s2_end": pd.Timestamp("2020-02-29"),
        "valid_start": pd.Timestamp("2020-03-01"), "valid_end": pd.Timestamp("2020-03-31"),
    }])
    data_path = tmp_path / "data.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(data_path)
    monkeypatch.setattr(command, "load_folds", lambda _: folds)
    monkeypatch.setattr(command, "load_featured_data", lambda *_: pd.DataFrame())
    monkeypatch.setattr(command, "sha256_file", lambda _: "data")
    monkeypatch.setattr(command, "registry_hash", lambda: "registry")
    monkeypatch.setattr(command, "baseline_feature_hash", lambda: "features")
    monkeypatch.setattr(command, "dirty_worktree", lambda: False)
    monkeypatch.setattr(command.subprocess, "run", lambda *args, **kwargs: type("R", (), {"stdout": "commit\n"})())
    calls = []

    def fake_fold(fold, data, target, *args):
        calls.append(fold.fold_id)
        target.mkdir(parents=True, exist_ok=True)
        (target / "meta.json").write_text(json.dumps({
            "registry_hash": "registry", "data_hash": "data", "baseline_feature_hash": "features",
            "git_commit": "commit",
        }))
        return {}

    monkeypatch.setattr(command, "run_baseline_fold", fake_fold)
    command.run_baselines(Path("folds.csv"), data_path, tmp_path, resume=True)
    command.run_baselines(Path("folds.csv"), data_path, tmp_path, resume=True)
    assert calls == ["f1"]
    # Same hashes but output from a different commit must not be reused.
    (tmp_path / "_baselines" / "f1" / "meta.json").write_text(json.dumps({
        "registry_hash": "registry", "data_hash": "data", "baseline_feature_hash": "features",
        "git_commit": "other",
    }))
    command.run_baselines(Path("folds.csv"), data_path, tmp_path, resume=True)
    assert calls == ["f1", "f1"]
    calls.pop()
    (tmp_path / "_baselines" / "f1" / "meta.json").write_text(json.dumps({
        "registry_hash": "old", "data_hash": "data", "baseline_feature_hash": "features",
        "git_commit": "commit",
    }))
    command.run_baselines(Path("folds.csv"), data_path, tmp_path, resume=True)
    assert calls == ["f1", "f1"]
    monkeypatch.setattr(command, "registry_hash", lambda: "new-registry")
    command.run_baselines(Path("folds.csv"), data_path, tmp_path, resume=True)
    monkeypatch.setattr(command, "sha256_file", lambda _: "new-data")
    command.run_baselines(Path("folds.csv"), data_path, tmp_path, resume=True)
    monkeypatch.setattr(command, "baseline_feature_hash", lambda: "new-features")
    command.run_baselines(Path("folds.csv"), data_path, tmp_path, resume=True)
    assert calls == ["f1", "f1", "f1", "f1", "f1"]


def test_baseline_feature_build_drops_market_columns(tmp_path, monkeypatch) -> None:
    raw = pd.DataFrame({
        "market_prob": [0.7],
        "mkt_p": [0.7],
        "win_odds": [2.0],
        "ordinary": [1.0],
    })
    observed = {}
    monkeypatch.setattr(command.pd, "read_parquet", lambda _: raw.copy())

    def fake_builder(frame, include_market=True):
        observed["columns"] = set(frame.columns)
        observed["include_market"] = include_market
        return frame.assign(built=1.0)

    monkeypatch.setattr(command, "build_features", fake_builder)
    command.load_featured_data(tmp_path / "raw.parquet", tmp_path, "hash")
    assert observed == {"columns": {"ordinary"}, "include_market": False}


def test_baseline_runner_rejects_test_period(tmp_path) -> None:
    folds = pd.DataFrame([{
        "fold_id": "bad", "era": "kaggle",
        "s1_start": "2020-01-01", "s1_end": "2020-01-31",
        "s2_start": "2020-02-01", "s2_end": "2020-02-29",
        "valid_start": "2020-03-01", "valid_end": "2026-04-01",
    }])
    path = tmp_path / "folds.csv"
    folds.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Invalid chronological fold rows"):
        command.run_baselines(path)
