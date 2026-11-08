"""Synthetic tests for analysis/eval_utils.py and analysis/final_scoring.py.

No project data is read: every test builds a small fake predictions file, so the real h_test
window is never touched. The fake "h_test" rows here are synthetic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import eval_utils as ev  # noqa: E402
import final_scoring as fs  # noqa: E402
import provenance as prov  # noqa: E402


@pytest.fixture(autouse=True)
def _published_commit(monkeypatch):
    """The fake source runs below record a made-up commit. Treat it as published so these
    tests exercise scoring; tests/test_provenance.py tests the provenance rules themselves."""
    monkeypatch.setattr(prov, "commit_on_remote", lambda *a, **k: (True, "test: published"))


def _fake(n_days: int = 8, races_per_day: int = 4, runners: int = 6, seed: int = 0,
          start: str = "2026-01-03", split: str = "h_s2") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp(start) + pd.Timedelta(days=7 * d)
        for r in range(races_per_day):
            rid = f"{split}_{d}_{r}"
            strength = rng.gamma(2.0, size=runners)
            p = strength / strength.sum()
            winner = rng.choice(runners, p=p)
            noise = rng.gamma(2.0, size=runners)
            for h in range(runners):
                rows.append({"race_id": rid, "horse_id": f"H{h}", "split": split,
                             "race_date": date, "is_winner": float(h == winner),
                             "market_complete": True, "_s": strength[h], "_n": noise[h]})
    df = pd.DataFrame(rows)
    g = df.groupby("race_id")
    df["p_market"] = df["_s"] / g["_s"].transform("sum")
    blend = df["_s"] * 0.7 + df["_n"] * 0.3
    df["p_model"] = blend / blend.groupby(df["race_id"]).transform("sum")
    df["p_uniform"] = ev.uniform_prob(df)
    beta = ev.fit_market_refit_beta(df["p_market"], df)
    df["p_market_refit"] = ev.market_refit_prob(df["p_market"], df, beta)
    return df.drop(columns=["_s", "_n"])


def _fake_file(tmp_path: Path, registry_version: str | None = None) -> Path:
    """Fake combined file plus its provenance sidecar, built from a fake source run whose
    meta.json records `registry_version` (default: the live registry, i.e. a valid run)."""
    s2 = _fake(split="h_s2", start="2026-01-03", seed=1)
    test = _fake(split="h_test", start="2026-04-04", seed=2)
    run = tmp_path / "source_run"
    run.mkdir(exist_ok=True)
    source = run / "predictions.parquet"
    pd.concat([s2, test], ignore_index=True).to_parquet(source, index=False)
    (run / "meta.json").write_text(json.dumps({
        "registry_version": registry_version or prov.current_registry()["version"],
        "git_commit": "abc1234"}))
    path = tmp_path / "preds.parquet"
    pd.concat([s2, test], ignore_index=True).to_parquet(path, index=False)
    prov.write_sidecar(path, [prov.source_entry(source, run / "meta.json")])
    return path


def _args(tmp_path: Path, preds: Path, *extra: str) -> list[str]:
    return ["--predictions", str(preds), "--output-root", str(tmp_path / "out"),
            "--freeze-record", str(tmp_path / "out" / "freeze.json"), *extra]


# ----------------------------------------------------------------------- eval_utils
def test_guard_blocks_test_rows_by_split_and_by_date():
    df = _fake(split="h_test", start="2026-04-04")
    with pytest.raises(RuntimeError):
        ev.check_not_test(df)
    no_split = df.drop(columns="split")                 # e.g. fold files: caught by date
    with pytest.raises(RuntimeError):
        ev.check_not_test(no_split)
    with pytest.raises(RuntimeError):
        ev.scoreboard(no_split, ["p_model"])
    ev.check_not_test(df, final_run=True)                # sanctioned run is allowed


def test_dead_heat_uses_log_of_mean_winner_probability():
    df = pd.DataFrame({"race_id": ["a"] * 3, "is_winner": [1.0, 1.0, 0.0]})
    loss = ev.race_losses(pd.Series([0.5, 0.3, 0.2]), df)
    assert loss["a"] == pytest.approx(-np.log(0.4))


def test_day_bootstrap_matches_manual_cluster_resample():
    df = _fake()
    a = ev.race_losses(df["p_market"], df)
    b = ev.race_losses(df["p_model"], df)
    days = ev.race_days(df)
    mean, lo, hi = ev.paired_bootstrap(a, b, n=500, seed=3, days=days)
    d = (a - b)
    by_day = d.groupby(days.reindex(d.index).to_numpy()).agg(["sum", "count"])
    pick = np.random.default_rng(3).integers(0, len(by_day), size=(500, len(by_day)))
    means = by_day["sum"].to_numpy()[pick].sum(1) / by_day["count"].to_numpy()[pick].sum(1)
    assert mean == pytest.approx(d.mean())
    assert (lo, hi) == pytest.approx(tuple(np.percentile(means, [2.5, 97.5])))


def test_day_bootstrap_matches_fold_runner_when_available():
    try:
        sys.path.insert(0, str(REPO))
        from model_logic.validation.evaluation import paired_bootstrap_gain
    except ImportError:
        pytest.skip("fold runner not importable")
    df = _fake(seed=5)
    a = ev.race_losses(df["p_market"], df)
    b = ev.race_losses(df["p_model"], df)
    days = ev.race_days(df)
    _, lo, hi = ev.paired_bootstrap(a, b, seed=0, days=days)
    assert (lo, hi) == pytest.approx(paired_bootstrap_gain(a, b, days.reindex(a.index), seed=0))


def test_scoreboard_records_day_bootstrap():
    table = ev.scoreboard(_fake(), ["p_market", "p_model"])
    assert table.attrs["bootstrap"] == "race-day clusters"
    assert table.attrs["n_days"] == 8
    assert table.attrs["n_races"] == 32


# -------------------------------------------------------------------- final_scoring
def test_dry_run_scores_h_s2_only(tmp_path):
    preds = _fake_file(tmp_path)
    assert fs.main(_args(tmp_path, preds)) == 0
    board = pd.read_csv(tmp_path / "out" / "dry_run_h_s2" / "scoreboard.csv")
    assert set(board["races"]) == {32}                  # the 32 h_s2 races, no test races
    assert not (tmp_path / "out" / "h_test").exists()


def test_final_needs_confirmation_and_freeze(tmp_path):
    preds = _fake_file(tmp_path)
    with pytest.raises(SystemExit, match="confirm"):
        fs.main(_args(tmp_path, preds, "--final"))
    with pytest.raises(SystemExit, match="freeze record"):
        fs.main(_args(tmp_path, preds, "--final", "--confirm", fs.CONFIRM))
    assert not (tmp_path / "out" / "h_test").exists()


def test_final_refuses_changed_file(tmp_path):
    preds = _fake_file(tmp_path)
    assert fs.main(_args(tmp_path, preds, "--freeze")) == 0
    df = pd.read_parquet(preds)
    df.loc[0, "p_model"] += 1e-9
    df.to_parquet(preds, index=False)
    with pytest.raises(SystemExit, match="changed since the freeze"):
        fs.main(_args(tmp_path, preds, "--final", "--confirm", fs.CONFIRM))


def test_freeze_and_final_refuse_stale_registry(tmp_path):
    preds = _fake_file(tmp_path, registry_version="v3.0")
    assert fs.main(_args(tmp_path, preds)) == 0               # dry run only warns
    report = (tmp_path / "out" / "dry_run_h_s2" / "report.md").read_text()
    assert "FAILED" in report and "v3.0" in report
    with pytest.raises(SystemExit, match="provenance check failed"):
        fs.main(_args(tmp_path, preds, "--freeze"))
    assert not (tmp_path / "out" / "freeze.json").exists()


def test_freeze_refuses_file_without_sidecar(tmp_path):
    preds = _fake_file(tmp_path)
    prov.sidecar_path(preds).unlink()
    with pytest.raises(SystemExit, match="provenance"):
        fs.main(_args(tmp_path, preds, "--freeze"))


def test_final_rechecks_provenance(tmp_path):
    preds = _fake_file(tmp_path)
    assert fs.main(_args(tmp_path, preds, "--freeze")) == 0
    side = prov.sidecar_path(preds)
    record = json.loads(side.read_text())
    record["sources"][0]["meta"]["registry_version"] = "v3.0"   # registry moved on after freeze
    side.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="provenance check failed"):
        fs.main(_args(tmp_path, preds, "--final", "--confirm", fs.CONFIRM))
    assert not (tmp_path / "out" / "h_test").exists()


def test_final_runs_once_on_test_rows_only(tmp_path):
    preds = _fake_file(tmp_path)
    assert fs.main(_args(tmp_path, preds, "--freeze")) == 0
    with pytest.raises(SystemExit, match="already exists"):
        fs.main(_args(tmp_path, preds, "--freeze"))
    assert fs.main(_args(tmp_path, preds, "--final", "--confirm", fs.CONFIRM)) == 0
    board = pd.read_csv(tmp_path / "out" / "h_test" / "scoreboard.csv")
    assert set(board["races"]) == {32}
    assert "p_market_refit" in set(board["model"])
    with pytest.raises(SystemExit, match="scored once"):
        fs.main(_args(tmp_path, preds, "--final", "--confirm", fs.CONFIRM))


def test_freeze_records_timestamp_and_race_dates(tmp_path):
    """Tier 3 (8 Nov): the freeze timestamp is recorded explicitly, with each split's date range
    and the number of races dated after the freeze day. Reported only; nothing is filtered."""
    preds = _fake_file(tmp_path)
    assert fs.main(_args(tmp_path, preds, "--freeze")) == 0
    record = json.loads((tmp_path / "out" / "freeze.json").read_text())
    assert record["freeze_timestamp_utc"] == record["created_utc"]
    assert record["race_dates_per_split"]["h_s2"][0] == "2026-01-03"
    assert record["race_dates_per_split"]["h_test"][0] == "2026-04-04"
    # the fake test races are all before today, so none is after the freeze
    assert record["races_after_freeze_per_split"] == {"h_s2": 0, "h_test": 0}
    assert fs.main(_args(tmp_path, preds, "--final", "--confirm", fs.CONFIRM)) == 0
    report = (tmp_path / "out" / "h_test" / "report.md").read_text()
    assert "Freeze timestamp (UTC)" in report and "0 of 32" in report


def test_race_date_summary_counts_only_days_after_the_freeze_day():
    from datetime import datetime, timezone
    meta = pd.DataFrame({"race_id": ["a", "a", "b", "c"], "split": ["h_test"] * 4,
                         "race_date": pd.to_datetime(["2026-11-12", "2026-11-12", "2026-11-13",
                                                      "2026-11-01"])})
    out = fs.race_date_summary(meta, datetime(2026, 11, 12, 15, 59, tzinfo=timezone.utc))
    assert out["races_after_freeze_per_split"] == {"h_test": 1}     # only 13 Nov counts
    assert out["race_dates_per_split"]["h_test"] == ["2026-11-01", "2026-11-13"]
    no_dates = fs.race_date_summary(meta.drop(columns="race_date"), datetime.now(timezone.utc))
    assert no_dates["races_after_freeze_per_split"] is None
