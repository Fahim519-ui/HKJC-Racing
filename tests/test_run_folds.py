"""Synthetic contract tests for chronological fold evaluation."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import model_logic.run_folds as runner
from model_logic.feature_registry import FORBIDDEN_INPUTS, get_features


def _fold_row() -> dict[str, str]:
    return {
        "fold_id": "f1",
        "era": "kaggle",
        "s1_start": "2020-01-01",
        "s1_end": "2020-01-31",
        "s2_start": "2020-02-01",
        "s2_end": "2020-03-31",
        "valid_start": "2020-04-01",
        "valid_end": "2020-04-30",
    }


def _stage2_frame() -> pd.DataFrame:
    rows = []
    for race_number, date in enumerate(
        ["2020-01-03", "2020-01-17", "2020-02-03", "2020-02-17", "2020-03-03", "2020-03-17"]
    ):
        for horse, probabilities in enumerate(((0.65, 0.60, 0.55), (0.35, 0.40, 0.45))):
            rows.append(
                {
                    "race_id": f"r{race_number}",
                    "race_date": pd.Timestamp(date),
                    "n_runners": 2,
                    "is_winner": float(horse == (0 if race_number < 4 else 1)),
                    "finish_pos": horse + 1,
                    "p_market": probabilities[0],
                    "p_cl": probabilities[1],
                    "p_lgbm": probabilities[2],
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "change",
    [
        {"s1_end": "2020-02-01"},
        {"valid_end": "2026-04-01"},
    ],
)
def test_load_folds_rejects_bad_chronology(tmp_path, change: dict[str, str]) -> None:
    row = _fold_row() | change
    path = tmp_path / "folds.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Invalid chronological fold rows"):
        runner.load_folds(path)


def test_validate_periods_rejects_race_overlap() -> None:
    with pytest.raises(ValueError, match="Race ids overlap"):
        runner.validate_periods(
            {
                "s1": pd.DataFrame({"race_id": ["r1"]}),
                "s2": pd.DataFrame({"race_id": ["r1"]}),
            }
        )


def test_stage_two_does_not_read_valid_outcomes() -> None:
    s2 = _stage2_frame()
    valid = _stage2_frame().assign(race_id=lambda frame: "v" + frame.race_id)
    _, first, first_meta = runner.add_stage2_variants(s2, valid)
    blanked = valid.assign(is_winner=np.nan, finish_pos=np.nan)
    _, second, second_meta = runner.add_stage2_variants(s2, blanked)

    columns = [column for column in first if column.startswith("p_")]
    assert_frame_equal(first[columns], second[columns], check_exact=True)
    assert first_meta == second_meta


def test_stage_two_outcomes_change_blend_not_stage_one_predictions() -> None:
    s2 = _stage2_frame()
    valid = _stage2_frame().assign(race_id=lambda frame: "v" + frame.race_id)
    changed = s2.assign(is_winner=lambda frame: 1.0 - frame.is_winner)
    before_s2, before_valid, before_meta = runner.add_stage2_variants(s2, valid)
    after_s2, after_valid, after_meta = runner.add_stage2_variants(changed, valid)

    assert_frame_equal(before_s2[["p_cl", "p_lgbm"]], after_s2[["p_cl", "p_lgbm"]], check_exact=True)
    assert_frame_equal(before_valid[["p_cl", "p_lgbm"]], after_valid[["p_cl", "p_lgbm"]], check_exact=True)
    assert before_meta["theta"] != after_meta["theta"]


def test_forward_lambda_search_is_deterministic_and_strictly_past_only(monkeypatch) -> None:
    frame = _stage2_frame()
    observed: list[tuple[pd.Period, pd.Period]] = []
    original = runner.fit_blend

    def checked_fit(train: pd.DataFrame, columns: list[str], lam: float) -> np.ndarray:
        observed.append(
            (train.race_date.dt.to_period("M").max(), frame.loc[
                frame.race_date.dt.to_period("M") > train.race_date.dt.to_period("M").max(), "race_date"
            ].dt.to_period("M").min())
        )
        return original(train, columns, lam)

    monkeypatch.setattr(runner, "fit_blend", checked_fit)
    first, first_fallback = runner.forward_lambda_scores(frame, ["p_market", "p_cl", "p_lgbm"])
    second, second_fallback = runner.forward_lambda_scores(frame, ["p_market", "p_cl", "p_lgbm"])

    assert first == second
    assert not first_fallback and not second_fallback
    assert all(train_month < scored_month for train_month, scored_month in observed)


def test_stage_one_inputs_are_market_free() -> None:
    frame = pd.DataFrame(
        {"race_id": ["r1", "r1"], "finish_pos": [1, 2], "feature": [0.2, 0.1], "mkt_p": [0.9, 0.1]}
    )
    races = runner.stage1_races(frame, ["feature"], pd.Series({"feature": 0.0}), pd.Series({"feature": 1.0}))
    cl_features = get_features("conditional_logit")

    assert not races.mkt_ok.any()
    assert np.array_equal(races.mkt_logit, np.zeros(2))
    assert not set(cl_features).intersection(FORBIDDEN_INPUTS)
    assert not any(feature.startswith("mkt_") for feature in cl_features)


def test_resume_requires_matching_config_registry_and_data_hashes(tmp_path, monkeypatch) -> None:
    folds = pd.DataFrame([_fold_row()])
    data_path = tmp_path / "data.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(data_path)
    target = tmp_path / "out" / "primary" / "f1"
    target.mkdir(parents=True)
    provenance = {
        "config_name": "primary", "config_hash": runner.get_config("primary").fingerprint(),
        "registry_hash": "registry", "data_hash": "data", "git_commit": "commit",
    }
    (target / "meta.json").write_text(json.dumps(provenance))
    gate = {"git_commit": "commit", "git_dirty": False, "evidence": True, "gate": "passed"}
    monkeypatch.setattr(runner, "require_evidence_run", lambda repo, data: dict(gate))
    monkeypatch.setattr(runner, "load_folds", lambda _: folds.assign(**{
        column: pd.to_datetime(folds[column]) for column in runner.REQUIRED_FOLD_COLUMNS[2:]
    }))
    monkeypatch.setattr(runner, "sha256_file", lambda _: "data")
    monkeypatch.setattr(runner, "registry_hash", lambda: "registry")
    monkeypatch.setattr(runner, "build_features", lambda frame: frame)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="test\n"),
    )
    calls: list[object] = []
    monkeypatch.setattr(runner, "run_fold", lambda *args: calls.append(args) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_folds", "--folds", str(tmp_path / "folds.csv"), "--data", str(data_path), "--out", str(tmp_path / "out"),
         "--fold-ids", "f1", "--resume"],
    )

    assert runner.main() == 0
    assert not calls
    (target / "meta.json").write_text(json.dumps(provenance | {"registry_hash": "old"}))
    assert runner.main() == 0
    assert len(calls) == 1
    # Same hashes but output from a different commit must not be reused.
    (target / "meta.json").write_text(json.dumps(provenance | {"git_commit": "other"}))
    assert runner.main() == 0
    assert len(calls) == 2
    # Nor output from a different ablation config definition.
    (target / "meta.json").write_text(json.dumps(provenance | {"config_hash": "old"}))
    assert runner.main() == 0
    assert len(calls) == 3


def test_feature_selection_uses_only_inner_train_and_tune(monkeypatch) -> None:
    inner_train = pd.DataFrame({"race_id": ["a", "a"], "finish_pos": [1, 2], "feature": [0.1, 0.2]})
    inner_tune = pd.DataFrame({"race_id": ["b", "b"], "finish_pos": [1, 2], "feature": [0.3, 0.4]})
    s1 = pd.concat(
        [inner_train, inner_tune, pd.DataFrame({"race_id": ["c", "c"], "finish_pos": [1, 2], "feature": [0.5, 0.6]})],
        ignore_index=True,
    )
    selected_inputs: list[np.ndarray] = []

    def fake_select(train, tune, correlation, minimum_gain, features, base):
        selected_inputs.append(np.concatenate([train.X.ravel(), tune.X.ravel()]))
        return [0], pd.DataFrame()

    monkeypatch.setattr(runner, "select_features", fake_select)
    monkeypatch.setattr(runner, "fit_beta", lambda races, **kwargs: np.zeros(races.X.shape[1]))
    monkeypatch.setattr(runner, "tune_loss", lambda *args: 0.0)
    monkeypatch.setattr(runner, "standard_errors", lambda beta, *args: np.zeros_like(beta))
    monkeypatch.setattr(runner, "seg_lse", lambda utility, races: (None, np.full(len(utility), 0.5)))

    first, _ = runner.fit_conditional_logit(
        inner_train, inner_tune, s1, [inner_tune], ["feature"], [], 0.001
    )
    altered_s1 = s1.copy()
    altered_s1.loc[altered_s1.race_id.eq("c"), "feature"] = 999.0
    second, _ = runner.fit_conditional_logit(
        inner_train, inner_tune, altered_s1, [inner_tune], ["feature"], [], 0.001
    )

    assert first["selected"] == second["selected"] == ["feature"]
    assert np.array_equal(selected_inputs[0], selected_inputs[1])
