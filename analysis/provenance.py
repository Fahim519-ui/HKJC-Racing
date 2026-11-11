"""Provenance checks: refuse predictions that were not made by the current, published code.

Why this exists (7 Nov): every fold run in artifacts/folds/ recorded registry_version "v3.0"
and git commit fc759fe, a commit that is not on GitHub. The registry had already moved to
v3.1, so those runs never saw the track/distance, trajectory or interaction features, and
nobody could rebuild them from the repo. The scores themselves looked normal; only the
run's meta.json showed the problem. These checks make that mistake stop the pipeline
instead of reaching the one-shot h_test run.

A prediction run passes when its meta.json says:
  * registry_version == model_logic/feature_registry.py REGISTRY_VERSION (the live registry), and
  * git_commit is a commit that the remote branch (default origin/main) contains, i.e. the
    code that produced it is published and anyone can check it out.

registry_hash is also compared. A different hash with the same version means the registry
was edited after the run (for example a status or monotone sign). That is reported as a
warning, not an error, because some edits do not change the model inputs; the person
freezing must read it.

The remote check uses the local copy of the remote branch. Run `git fetch origin` first,
otherwise a commit pushed by someone else since your last fetch looks unpublished. Nothing
here fetches by itself.

What this cannot catch: run_folds.py records `git rev-parse HEAD` even when the working
tree has uncommitted edits, so a run can carry a published commit and still have used
local changes. Only a run from a clean checkout proves the code.

Command line (no outcomes are read; safe at any time):
    python analysis/provenance.py                      # combined file + primary folds
    python analysis/provenance.py --folds-root artifacts/folds/primary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY_FILE = REPO / "model_logic" / "feature_registry.py"
REMOTE_REF = "origin/main"
SIDECAR_SUFFIX = ".provenance.json"
QUARANTINE_FILE = Path(__file__).resolve().parent / "quarantine.json"
EVIDENCE_OUTPUT = Path("artifacts") / "folds"


# --------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_registry(registry_file: Path = REGISTRY_FILE) -> dict[str, str | None]:
    """Version and hash of the live registry. The version is read from the source text so
    this works without importing the model code; the hash needs the import."""
    text = registry_file.read_text(encoding="utf-8")
    match = re.search(r'^REGISTRY_VERSION\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"REGISTRY_VERSION not found in {registry_file}")
    reg_hash = None
    try:
        import importlib.util
        import sys

        for folder in (str(registry_file.parent.parent), str(registry_file.parent)):
            if folder not in sys.path:
                sys.path.insert(0, folder)
        spec = importlib.util.spec_from_file_location("_live_feature_registry", registry_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module          # dataclasses need the module registered
        spec.loader.exec_module(module)
        reg_hash = module.registry_hash()
    except Exception:  # hash is advisory; version alone decides pass/fail
        reg_hash = None
    return {"version": match.group(1), "hash": reg_hash}


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=False)


def commit_on_remote(commit: str | None, repo: Path = REPO,
                     remote_ref: str = REMOTE_REF) -> tuple[bool, str]:
    """True if `commit` exists and is an ancestor of (or equal to) `remote_ref`."""
    if not commit:
        return False, "no git_commit recorded"
    if _git(repo, "rev-parse", "--verify", "--quiet", f"{remote_ref}^{{commit}}").returncode != 0:
        return False, f"remote ref {remote_ref} not found locally (run `git fetch origin`)"
    resolved = _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    if resolved.returncode != 0:
        return False, (f"commit {commit} does not exist in this clone; it was never pushed "
                       f"(or run `git fetch origin` if a teammate pushed it recently)")
    full = resolved.stdout.strip()
    if _git(repo, "merge-base", "--is-ancestor", full, remote_ref).returncode != 0:
        return False, f"commit {commit} exists locally but is not on {remote_ref}"
    return True, f"commit {commit} is on {remote_ref}"


@dataclass
class Report:
    """Problems block the run; warnings are printed and recorded but do not block."""
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked: list[dict] = field(default_factory=list)
    # One entry per fold from check_fold_root: {"fold", "path", "ok", "problems"}. Lets callers
    # and tests identify folds by path parts instead of matching message text.
    folds: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def extend(self, other: "Report") -> None:
        self.problems += other.problems
        self.warnings += other.warnings
        self.checked += other.checked
        self.folds += other.folds

    def lines(self) -> list[str]:
        out = [f"PROBLEM: {p}" for p in self.problems]
        out += [f"WARNING: {w}" for w in self.warnings]
        return out or ["provenance OK"]


def check_meta(meta: dict, label: str, repo: Path = REPO, remote_ref: str = REMOTE_REF,
               registry: dict | None = None) -> Report:
    """Check one run's recorded registry version, registry hash and git commit."""
    registry = registry or current_registry(repo / "model_logic" / "feature_registry.py")
    report = Report()
    version = meta.get("registry_version")
    commit = meta.get("git_commit")
    if version != registry["version"]:
        report.problems.append(
            f"{label}: registry_version {version!r}, live registry is {registry['version']!r}; "
            f"re-run it on the current registry")
    published, why = commit_on_remote(commit, repo, remote_ref)
    if not published:
        report.problems.append(f"{label}: {why}")
    if meta.get("git_dirty") is True or meta.get("dirty_worktree") is True:
        report.problems.append(
            f"{label}: the run was made from a working tree with uncommitted changes "
            f"(git_dirty), so its commit does not identify the code; re-run from a clean checkout")
    if meta.get("evidence") is False:
        report.problems.append(
            f"{label}: marked evidence=false (a --smoke or unchecked run); it cannot be frozen")
    recorded_hash = meta.get("registry_hash")
    if (version == registry["version"] and recorded_hash and registry["hash"]
            and recorded_hash != registry["hash"]):
        report.warnings.append(
            f"{label}: registry_hash differs from the live registry (same version, edited "
            f"since the run); confirm the edit did not change any model input")
    report.checked.append({"label": label, "registry_version": version,
                           "registry_hash": recorded_hash, "git_commit": commit,
                           "commit_published": published})
    return report


def _label(path: Path, repo: Path = REPO) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(repo).resolve()))
    except ValueError:
        return str(path)


# ------------------------------------------------------------------------ quarantine
def load_quarantine(repo: Path = REPO, quarantine_file: Path | None = None) -> list[dict]:
    """Folders declared non-evidence (analysis/quarantine.json): [{"path", "reason"}]."""
    path = quarantine_file or Path(repo) / "analysis" / "quarantine.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def quarantine_reason(target: Path, repo: Path = REPO,
                      quarantine_file: Path | None = None) -> str | None:
    """Reason if `target` lies inside a quarantined folder. Compares resolved path parts, so
    it is independent of separators and of similar-looking folder names."""
    parts = Path(target).resolve().parts
    for entry in load_quarantine(repo, quarantine_file):
        q_parts = (Path(repo) / entry["path"]).resolve().parts
        if parts[:len(q_parts)] == q_parts:
            return f"{entry['path']} is quarantined: {entry['reason']}"
    return None


def check_run_dir(run_dir: Path, repo: Path = REPO, remote_ref: str = REMOTE_REF,
                  registry: dict | None = None) -> Report:
    """A model output folder must carry the meta.json written by the code that produced it,
    and must not lie in a quarantined folder."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    quarantined = quarantine_reason(run_dir, repo)
    if not meta_path.exists():
        report = Report()
        report.problems.append(f"{_label(run_dir, repo)}: no meta.json, so the registry version "
                               f"and code commit that produced these predictions are unknown")
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        report = check_meta(meta, _label(run_dir, repo), repo, remote_ref, registry)
    if quarantined:
        report.problems.append(f"{_label(run_dir, repo)}: {quarantined}")
    return report


def check_fold_root(folds_root: Path, repo: Path = REPO, remote_ref: str = REMOTE_REF) -> Report:
    """Every fold under a config folder (e.g. artifacts/folds/primary/{k1,k2,h1})."""
    registry = current_registry(repo / "model_logic" / "feature_registry.py")
    report = Report()
    folds = sorted(p.parent for p in Path(folds_root).glob("*/predictions.parquet"))
    if not folds:
        report.problems.append(f"{folds_root}: no */predictions.parquet found")
    for fold in folds:
        sub = check_run_dir(fold, repo, remote_ref, registry)
        report.extend(sub)
        report.folds.append({"fold": fold.name, "path": _label(fold, repo), "ok": sub.ok,
                             "problems": list(sub.problems)})
    return report


def load_fold_predictions(fold_dir: Path, role: str | None = None, allow_unverified: bool = False,
                          repo: Path = REPO, remote_ref: str = REMOTE_REF):
    """Read a fold runner's predictions.parquet only if its meta.json passes check_run_dir.
    Use this (not pd.read_parquet) whenever fold predictions feed a comparison or a table.
    allow_unverified=True reads a stale run anyway; the caller must label its output."""
    import pandas as pd

    checked = check_run_dir(Path(fold_dir), repo, remote_ref)
    if not checked.ok and not allow_unverified:
        require(checked, f"fold predictions {fold_dir}")
    frame = pd.read_parquet(Path(fold_dir) / "predictions.parquet")
    if role is not None:
        frame = frame.loc[frame["role"].eq(role)].reset_index(drop=True)
    return frame, checked


# ---------------------------------------------------- combined predictions (sidecar)
def sidecar_path(predictions: Path) -> Path:
    predictions = Path(predictions)
    return predictions.with_name(predictions.stem + SIDECAR_SUFFIX)


def source_entry(source: Path, meta_path: Path | None) -> dict:
    """What the combined-file builder records about one input predictions file."""
    entry = {"path": str(source), "sha256": sha256_file(Path(source)), "meta_path": None,
             "meta": None}
    if meta_path is not None and Path(meta_path).exists():
        entry["meta_path"] = str(meta_path)
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        entry["meta"] = {k: meta.get(k) for k in ("registry_version", "registry_hash",
                                                  "git_commit", "data_hash", "git_dirty",
                                                  "dirty_worktree", "evidence")}
    return entry


def write_sidecar(predictions: Path, sources: list[dict], repo: Path = REPO) -> Path:
    head = _git(repo, "rev-parse", "HEAD").stdout.strip() or None
    dirty = [line for line in _git(repo, "status", "--porcelain").stdout.splitlines() if line]
    record = {
        "predictions_sha256": sha256_file(Path(predictions)),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder_git_commit": head,
        "builder_dirty_files": dirty,
        "sources": sources,
    }
    path = sidecar_path(predictions)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def check_predictions_file(predictions: Path, repo: Path = REPO,
                           remote_ref: str = REMOTE_REF) -> Report:
    """The combined predictions file must have a sidecar that matches it, and every source
    it was built from must pass check_meta."""
    report = Report()
    side = sidecar_path(predictions)
    if not side.exists():
        report.problems.append(
            f"{Path(predictions).name}: no {side.name}; rebuild it with "
            f"analysis/build_combined_predictions.py so its sources are recorded")
        return report
    record = json.loads(side.read_text(encoding="utf-8"))
    if record.get("predictions_sha256") != sha256_file(Path(predictions)):
        report.problems.append(f"{Path(predictions).name}: file changed after {side.name} "
                               f"was written; rebuild it")
    if record.get("builder_dirty_files"):
        report.warnings.append(f"{Path(predictions).name}: built from a working tree with "
                               f"uncommitted changes: {record['builder_dirty_files']}")
    quarantined = quarantine_reason(Path(predictions), repo)
    if quarantined:
        report.problems.append(f"{Path(predictions).name}: {quarantined}")
    registry = current_registry(repo / "model_logic" / "feature_registry.py")
    sources = record.get("sources") or []
    if not sources:
        report.problems.append(f"{side.name}: lists no source predictions")
    for src in sources:
        label = f"source {src.get('path')}"
        if src.get("path"):
            source_path = Path(src["path"])
            source_path = source_path if source_path.is_absolute() else Path(repo) / source_path
            quarantined = quarantine_reason(source_path, repo)
            if quarantined:
                report.problems.append(f"{label}: {quarantined}")
        if not src.get("meta"):
            report.problems.append(
                f"{label}: no meta.json, so the registry version and code commit that "
                f"produced it are unknown")
            continue
        report.extend(check_meta(src["meta"], label, repo, remote_ref, registry))
    return report


# ------------------------------------------------------------------------ run-gate
def dirty_files(repo: Path = REPO) -> list[str]:
    """`git status --porcelain` lines: modified, staged and untracked files (ignored files are
    not listed)."""
    return [line for line in _git(repo, "status", "--porcelain").stdout.splitlines() if line]


def require_evidence_run(repo: Path, data_path: Path, remote_ref: str = REMOTE_REF) -> dict:
    """Refuse to start an evidence run unless the code and data can be identified: a clean
    working tree, HEAD contained in `remote_ref`, and an existing data file. Returns the
    provenance fields the run must record. Raises SystemExit listing every reason otherwise."""
    repo = Path(repo)
    reasons = []
    dirty = dirty_files(repo)
    if dirty:
        shown = ", ".join(dirty[:10]) + (" ..." if len(dirty) > 10 else "")
        reasons.append(f"working tree has uncommitted or untracked files: {shown}; commit, "
                       f"stash or remove them (or use --smoke for a non-evidence run)")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip() or None
    published, why = commit_on_remote(head, repo, remote_ref)
    if not published:
        reasons.append(f"{why}; run `git fetch origin`, then push your commit before an "
                       f"evidence run")
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = repo / data_path
    if not data_path.exists():
        reasons.append(f"data file not found: {data_path}")
    if reasons:
        raise SystemExit("Refusing evidence run (run-gate):\n  - " + "\n  - ".join(reasons))
    registry = current_registry(repo / "model_logic" / "feature_registry.py")
    return {"git_commit": head, "git_dirty": False, "evidence": True, "gate": "passed",
            "remote_ref": remote_ref, "data_hash": sha256_file(data_path),
            "registry_version": registry["version"], "registry_hash": registry["hash"]}


def smoke_record(repo: Path, out: Path) -> dict:
    """Provenance for a --smoke run: not evidence, never written to the evidence folder."""
    repo = Path(repo)
    out_parts = (Path(out) if Path(out).is_absolute() else repo / out).resolve().parts
    evidence_parts = (repo / EVIDENCE_OUTPUT).resolve().parts
    if out_parts[:len(evidence_parts)] == evidence_parts:
        raise SystemExit(f"Refusing --smoke output inside {EVIDENCE_OUTPUT.as_posix()}: smoke "
                         f"runs are not evidence; use e.g. --out artifacts/folds-smoke")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip() or None
    return {"git_commit": head, "git_dirty": bool(dirty_files(repo)), "evidence": False,
            "gate": "smoke"}


def require(report: Report, context: str) -> None:
    """Stop with every problem listed. Used by --freeze and --final."""
    if not report.ok:
        details = "\n  - ".join(report.problems)
        raise SystemExit(f"Refusing {context}: provenance check failed.\n  - {details}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--predictions", type=Path,
                        default=REPO / "artifacts/combined/combined_predictions.parquet")
    parser.add_argument("--folds-root", type=Path, default=REPO / "artifacts/folds/primary")
    parser.add_argument("--remote-ref", default=REMOTE_REF)
    args = parser.parse_args(argv)
    registry = current_registry()
    print(f"Live registry: {registry['version']} (hash {registry['hash']})")
    combined = check_predictions_file(args.predictions, remote_ref=args.remote_ref)
    folds = check_fold_root(args.folds_root, remote_ref=args.remote_ref)
    for title, report in (("Combined predictions", combined), ("Fold runs", folds)):
        print(f"\n{title}:")
        print("\n".join(f"  {line}" for line in report.lines()))
    return 0 if combined.ok and folds.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
