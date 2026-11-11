"""Quarantined (non-evidence) results: listed, marked, and refused by the provenance check."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import provenance as prov  # noqa: E402

ENTRIES = json.loads((REPO / "analysis" / "quarantine.json").read_text(encoding="utf-8"))


def test_every_quarantined_folder_exists_and_is_marked():
    assert ENTRIES, "analysis/quarantine.json lists nothing"
    for entry in ENTRIES:
        folder = REPO / entry["path"]
        assert folder.is_dir(), entry["path"]
        assert (folder / "NOT_EVIDENCE.md").exists(), entry["path"]
        assert entry["reason"].strip(), entry["path"]


def test_stale_fold_runs_are_quarantined_while_current_runs_are_evidence():
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "artifacts/folds"],
                             capture_output=True, text=True, check=True).stdout.split()
    assert "artifacts/folds/primary/k1/meta.json" in tracked
    current = REPO / "artifacts" / "folds" / "primary" / "k1"
    assert prov.check_run_dir(current).ok
    moved = REPO / "legacy_code" / "artifacts" / "quarantine" / "folds_v3.0_fc759fe" / "primary" / "k1"
    assert (moved / "meta.json").exists()


def test_provenance_refuses_runs_inside_quarantine():
    run = REPO / "legacy_code" / "artifacts" / "quarantine" / "folds_v3.0_fc759fe" / "primary" / "k1"
    report = prov.check_run_dir(run)
    assert any("quarantined" in problem for problem in report.problems)


def test_quarantine_match_uses_path_parts_not_prefixes(tmp_path):
    qfile = tmp_path / "quarantine.json"
    qfile.write_text(json.dumps([{"path": "artifacts/market_offset_lightgbm", "reason": "r"}]))
    inside = tmp_path / "artifacts" / "market_offset_lightgbm" / "x"
    lookalike = tmp_path / "artifacts" / "market_offset_lightgbm_v2" / "x"
    assert prov.quarantine_reason(inside, tmp_path, qfile)
    assert prov.quarantine_reason(lookalike, tmp_path, qfile) is None


def test_current_combined_file_has_an_evidence_source():
    report = prov.check_predictions_file(REPO / "artifacts" / "combined" /
                                         "combined_predictions.parquet")
    assert report.ok
