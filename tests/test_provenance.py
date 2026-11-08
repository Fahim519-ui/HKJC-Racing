"""Tests for analysis/provenance.py: runs from a stale registry or unpublished code are refused.

Each test builds a throwaway git repository with a tiny feature_registry.py and a fake
"origin/main" ref, so the checks run without network access or project data.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import provenance as prov  # noqa: E402

REGISTRY_SRC = '''REGISTRY_VERSION = "v3.1"


def registry_hash() -> str:
    return "hash-live"
'''


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> dict:
    """Repo with a published commit (on origin/main) and a later local-only commit."""
    root = tmp_path / "repo"
    (root / "model_logic").mkdir(parents=True)
    (root / "model_logic" / "feature_registry.py").write_text(REGISTRY_SRC)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "published")
    published = _git(root, "rev-parse", "--short", "HEAD")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    (root / "notes.txt").write_text("local only")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "not pushed")
    local = _git(root, "rev-parse", "--short", "HEAD")
    return {"root": root, "published": published, "local": local}


def _meta(version: str, commit: str, reg_hash: str = "hash-live") -> dict:
    return {"registry_version": version, "git_commit": commit, "registry_hash": reg_hash}


def test_current_registry_reads_version_and_hash(repo):
    reg = prov.current_registry(repo["root"] / "model_logic" / "feature_registry.py")
    assert reg == {"version": "v3.1", "hash": "hash-live"}


def test_live_registry_and_published_commit_pass(repo):
    report = prov.check_meta(_meta("v3.1", repo["published"]), "run", repo["root"])
    assert report.ok and not report.warnings


def test_stale_registry_version_is_refused(repo):
    report = prov.check_meta(_meta("v3.0", repo["published"]), "run", repo["root"])
    assert not report.ok
    assert any("v3.0" in p and "v3.1" in p for p in report.problems)


def test_local_only_commit_is_refused(repo):
    report = prov.check_meta(_meta("v3.1", repo["local"]), "run", repo["root"])
    assert not report.ok
    assert any("not on origin/main" in p for p in report.problems)


def test_unknown_commit_is_refused(repo):
    report = prov.check_meta(_meta("v3.1", "fc759fe"), "run", repo["root"])
    assert not report.ok
    assert any("never pushed" in p for p in report.problems)


def test_missing_commit_and_missing_remote_ref_are_refused(repo):
    assert not prov.check_meta(_meta("v3.1", None), "run", repo["root"]).ok
    report = prov.check_meta(_meta("v3.1", repo["published"]), "run", repo["root"],
                             remote_ref="origin/does-not-exist")
    assert any("git fetch" in p for p in report.problems)


def test_edited_registry_is_a_warning_not_a_refusal(repo):
    report = prov.check_meta(_meta("v3.1", repo["published"], "hash-old"), "run", repo["root"])
    assert report.ok
    assert any("registry_hash differs" in w for w in report.warnings)


def _fold(tmp: Path, name: str, meta: dict | None) -> Path:
    run = tmp / name
    run.mkdir(parents=True)
    pd.DataFrame({"race_id": ["r1", "r1"], "role": ["s2", "valid"],
                  "is_winner": [1.0, 0.0]}).to_parquet(run / "predictions.parquet", index=False)
    if meta is not None:
        (run / "meta.json").write_text(json.dumps(meta))
    return run


def test_fold_root_reports_every_bad_fold(repo, tmp_path):
    root = tmp_path / "folds"
    _fold(root, "k1", _meta("v3.1", repo["published"]))
    _fold(root, "k2", _meta("v3.0", "fc759fe"))          # the 6 Nov situation
    _fold(root, "h1", None)                               # no meta.json at all
    report = prov.check_fold_root(root, repo["root"])
    text = "\n".join(report.problems)
    assert "k2" in text and "h1" in text and "no meta.json" in text
    assert "folds/k1:" not in text


def test_load_fold_predictions_refuses_stale_unless_allowed(repo, tmp_path):
    run = _fold(tmp_path / "folds", "k1", _meta("v3.0", repo["published"]))
    with pytest.raises(SystemExit, match="provenance check failed"):
        prov.load_fold_predictions(run, role="s2", repo=repo["root"])
    frame, report = prov.load_fold_predictions(run, role="s2", allow_unverified=True,
                                               repo=repo["root"])
    assert len(frame) == 1 and not report.ok             # read, but flagged


def _combined(tmp: Path, meta: dict | None) -> Path:
    run = _fold(tmp, "source", meta)
    out = tmp / "combined.parquet"
    pd.read_parquet(run / "predictions.parquet").to_parquet(out, index=False)
    prov.write_sidecar(out, [prov.source_entry(run / "predictions.parquet",
                                               run / "meta.json")])
    return out


def test_combined_file_passes_with_valid_source(repo, tmp_path):
    out = _combined(tmp_path, _meta("v3.1", repo["published"]))
    assert prov.check_predictions_file(out, repo["root"]).ok


def test_combined_file_refused_when_source_has_no_meta(repo, tmp_path):
    out = _combined(tmp_path, None)
    report = prov.check_predictions_file(out, repo["root"])
    assert any("no meta.json" in p for p in report.problems)


def test_combined_file_refused_when_changed_or_sidecar_missing(repo, tmp_path):
    out = _combined(tmp_path, _meta("v3.1", repo["published"]))
    pd.DataFrame({"race_id": ["x"]}).to_parquet(out, index=False)
    assert any("changed after" in p for p in prov.check_predictions_file(out, repo["root"]).problems)
    prov.sidecar_path(out).unlink()
    assert not prov.check_predictions_file(out, repo["root"]).ok
