"""Run-gate for evidence runs (analysis/provenance.py require_evidence_run / smoke_record).

Each test builds a throwaway git repository with a tiny feature registry and a fake
origin/main ref, so nothing touches the project data or the network.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
def repo(tmp_path: Path) -> Path:
    """Clean repo whose HEAD is on origin/main, with a data file committed."""
    root = tmp_path / "repo"
    (root / "model_logic").mkdir(parents=True)
    (root / "model_logic" / "feature_registry.py").write_text(REGISTRY_SRC)
    (root / "data").mkdir()
    (root / "data" / "runners.parquet").write_bytes(b"fake data")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "published")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    return root


DATA = Path("data") / "runners.parquet"


def test_clean_pushed_checkout_passes(repo):
    record = prov.require_evidence_run(repo, DATA)
    assert record["gate"] == "passed" and record["evidence"] is True
    assert record["git_dirty"] is False
    assert record["git_commit"] == _git(repo, "rev-parse", "HEAD")
    assert len(record["git_commit"]) == 40                       # full hash, not --short
    assert record["registry_version"] == "v3.1" and record["registry_hash"] == "hash-live"
    assert record["data_hash"] == prov.sha256_file(repo / DATA)


def test_modified_tracked_file_is_refused(repo):
    (repo / "model_logic" / "feature_registry.py").write_text(REGISTRY_SRC + "# edit\n")
    with pytest.raises(SystemExit, match="uncommitted or untracked"):
        prov.require_evidence_run(repo, DATA)


def test_untracked_file_is_refused(repo):
    (repo / "scratch.py").write_text("x = 1\n")
    with pytest.raises(SystemExit, match="scratch.py"):
        prov.require_evidence_run(repo, DATA)


def test_ignored_file_does_not_block(repo):
    (repo / ".gitignore").write_text("artifacts/\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-q", "-m", "ignore artifacts")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    (repo / "artifacts").mkdir()
    (repo / "artifacts" / "cache.parquet").write_bytes(b"x")
    assert prov.require_evidence_run(repo, DATA)["gate"] == "passed"


def test_unpushed_commit_is_refused(repo):
    (repo / "notes.txt").write_text("local only\n")
    _git(repo, "add", "notes.txt")
    _git(repo, "commit", "-q", "-m", "not pushed")
    with pytest.raises(SystemExit, match="not on origin/main"):
        prov.require_evidence_run(repo, DATA)


def test_missing_data_file_is_refused(repo):
    with pytest.raises(SystemExit, match="data file not found"):
        prov.require_evidence_run(repo, Path("data") / "missing.parquet")


def test_every_reason_is_listed_at_once(repo):
    (repo / "scratch.py").write_text("x = 1\n")
    with pytest.raises(SystemExit) as stop:
        prov.require_evidence_run(repo, Path("data") / "missing.parquet")
    message = str(stop.value)
    assert "uncommitted or untracked" in message and "data file not found" in message


def test_smoke_never_writes_to_the_evidence_folder(repo):
    for out in (Path("artifacts") / "folds", Path("artifacts") / "folds" / "primary"):
        with pytest.raises(SystemExit, match="Refusing --smoke"):
            prov.smoke_record(repo, out)
    record = prov.smoke_record(repo, Path("artifacts") / "folds-smoke")
    assert record["evidence"] is False and record["gate"] == "smoke"


def test_smoke_records_are_refused_by_the_freeze_check(repo):
    record = prov.smoke_record(repo, Path("artifacts") / "folds-smoke")
    meta = {"registry_version": "v3.1", **record}
    assert not prov.check_meta(meta, "smoke", repo).ok


@pytest.mark.parametrize("module", ["model_logic.run_ablation", "model_logic.run_folds",
                                    "model_logic.run_baselines"])
def test_every_command_calls_the_gate(module, monkeypatch, tmp_path):
    """Each command-line entry point must call the gate before doing any work."""
    sys.path.insert(0, str(REPO))
    command = __import__(module, fromlist=["main"])

    def refuse(repo, data):
        raise SystemExit("gate refused (test)")

    monkeypatch.setattr(command, "require_evidence_run", refuse)
    monkeypatch.setattr(sys, "argv", [module, "--folds", str(tmp_path / "folds.csv"),
                                      "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="gate refused"):
        command.main()
    assert not (tmp_path / "out").exists()                       # nothing ran
