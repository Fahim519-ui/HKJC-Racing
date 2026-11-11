"""Contract tests for the one-command ablation coordinator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import model_logic.run_ablation as command
from model_logic.ablation_configs import CONFIGS
from model_logic.aggregate_folds import PRIMARY_CONFIG, PRIMARY_VARIANT, ladder_evidence_status


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": ["r1", "r1"],
            "horse_id": ["h1", "h2"],
            "race_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "role": ["valid", "valid"],
            "is_winner": [1.0, 0.0],
            "p_market": [0.6, 0.4],
            "p_cl": [0.55, 0.45],
            "p_lgbm": [0.52, 0.48],
            "p_uniform": [0.5, 0.5],
            "p_market_calibrated": [0.6, 0.4],
            "p_cl_only": [0.55, 0.45],
            "p_lgbm_only": [0.52, 0.48],
            "p_mkt+cl": [0.56, 0.44],
            "p_mkt+lgbm": [0.54, 0.46],
            "p_mkt+cl+lgbm": [0.56, 0.44],
        }
    )


GATE_RECORD = {"git_commit": "commit", "git_dirty": False, "evidence": True, "gate": "passed"}


def test_run_ablation_forwards_options_and_writes_combined_metrics(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out"
    captured: dict = {}
    monkeypatch.setattr(command, "require_evidence_run", lambda repo, data: dict(GATE_RECORD))

    def fake_run_baselines(**kwargs):
        for fold_id in ("f1",):
            directory = output / "_baselines" / fold_id
            directory.mkdir(parents=True)
            frame = _prediction_frame()[["race_id", "horse_id", "role", "is_winner", "p_uniform"]].copy()
            frame["p_draw_only"] = [0.55, 0.45]
            frame["p_original_cl"] = [0.56, 0.44]
            frame.to_parquet(directory / "baseline_predictions.parquet", index=False)
            (directory / "meta.json").write_text(json.dumps({
                "data_hash": "data", "registry_hash": "registry", "git_commit": "commit",
            }))
        return [{"fold_id": "f1"}]

    def fake_run_folds(**kwargs):
        captured.update(kwargs)
        for config in CONFIGS.values():
            directory = output / config.name / "f1"
            directory.mkdir(parents=True)
            _prediction_frame().to_parquet(directory / "predictions.parquet", index=False)
            (directory / "cl_selected_features.json").write_text(json.dumps(["feature"]) + "\n")
            (directory / "meta.json").write_text(
                json.dumps(
                    {
                        "primary": config.primary,
                        "unavailable_exclusions": [],
                        "race_counts": {"inner_tune": 200},
                        "data_hash": "data",
                        "registry_hash": "registry",
                        "git_commit": "commit",
                        "fold": {"era": "kaggle"},
                    }
                )
                + "\n"
            )
        return [{"config": "primary", "fold_id": "f1"}]

    monkeypatch.setattr(command, "run_baselines", fake_run_baselines)
    monkeypatch.setattr(command, "run_folds", fake_run_folds)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ablation",
            "--folds",
            "folds.csv",
            "--data",
            "data.parquet",
            "--out",
            str(output),
            "--seeds",
            "1",
            "--inner-tune-months",
            "3",
            "--min-gain",
            "0.002",
            "--fold-ids",
            "f1",
            "--resume",
        ],
    )

    assert command.main() == 0
    assert captured == {
        "folds_path": Path("folds.csv"),
        "data_path": Path("data.parquet"),
        "seed_count": 1,
        "output": output,
        "inner_tune_months": 3,
        "minimum_gain": 0.002,
        "fold_ids": ["f1"],
        "all_configs": True,
        "resume": True,
        "run_record": GATE_RECORD,
    }
    metrics = pd.read_csv(output / "ablation_metrics.csv")
    assert set(metrics["config"]) == set(CONFIGS)
    evidence = json.loads((output / "evidence_summary.json").read_text())
    assert evidence["eligible_folds"] == ["f1"]


def test_run_ablation_does_not_aggregate_after_fold_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(command, "require_evidence_run", lambda repo, data: dict(GATE_RECORD))
    monkeypatch.setattr(command, "run_baselines", lambda **kwargs: [])
    monkeypatch.setattr(command, "run_folds", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("failed")))
    aggregate_called = False

    def aggregate(**kwargs):
        nonlocal aggregate_called
        aggregate_called = True
        return pd.DataFrame()

    monkeypatch.setattr(command, "aggregate_folds", aggregate)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ablation", "--folds", "folds.csv", "--out", str(tmp_path / "out")],
    )

    with pytest.raises(RuntimeError, match="failed"):
        command.main()
    assert not aggregate_called


def _primary_metrics(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "config": PRIMARY_CONFIG,
                "variant": PRIMARY_VARIANT,
                "fold_id": fold_id,
                "gain_vs_calibrated": gain,
                "ci_vs_calibrated_low": low,
                "ci_vs_calibrated_high": high,
            }
            for fold_id, gain, low, high in rows
        ]
    )


def test_evidence_status_accepts_any_positive_eligible_ladder_candidate() -> None:
    metrics = _primary_metrics(
        [
            ("k1", 0.01, 0.001, 0.02),
            ("k2", -0.01, -0.02, 0.001),
            ("h1", 0.01, 0.001, 0.02),
        ]
    )
    metrics.loc[0, ["config", "variant"]] = ["no_trainer_rate", "mkt+cl"]
    status = ladder_evidence_status(
        metrics,
        {"k1": 200, "k2": 200, "h1": 140},
    )

    assert status["eligible_folds"] == ["k1", "k2"]
    assert status["watch_only_folds"] == ["h1"]
    assert status["positive_candidate_count"] == 1
    assert status["positive_candidates"] == ["no_trainer_rate/mkt+cl/k1"]
    assert status["intervals_reported_only"]
    assert status["eligible_for_freeze"]


def test_evidence_status_rejects_when_no_eligible_candidate_improves() -> None:
    status = ladder_evidence_status(
        _primary_metrics(
            [
                ("k1", -0.01, -0.02, 0.001),
                ("k2", 0.0, -0.001, 0.02),
                ("h1", 0.01, 0.001, 0.02),
            ]
        ),
        {"k1": 200, "k2": 200, "h1": 140},
    )

    assert status["eligible_folds"] == ["k1", "k2"]
    assert status["watch_only_folds"] == ["h1"]
    assert status["positive_candidate_count"] == 0
    assert not status["eligible_for_freeze"]
