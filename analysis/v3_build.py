"""Clean raw-run regeneration of the v3.1 feature build, stamped with its provenance.

Task (status sheet of 8 Nov, Tier 1, Nayeon): regenerate every feature from the raw runners
file on main, show that the v3 columns are reproduced, that source isolation and same-day
invariance hold, and that nothing depends on version2_runners_merged.parquet. The output is
a build stamped with the data hash, the registry hash and the builder-code hash.

What it does (no model is fitted and no metric is computed, so it is safe at any time):
  1. Refuses an input that already carries built feature columns, and any builder module
     that mentions version2_runners_merged.
  2. Builds twice from the raw runners and requires identical output (determinism).
  3. Perturbation checks on the v3.1 columns (the same rules as tests/test_v3_features_leakage.py):
     same-day shuffle, cut-off, source isolation and market shuffle.
  4. Writes the build (parquet) plus a manifest with SHA-256 hashes of the data, the registry,
     the builder source files and every v3 column.

Usage:
    python analysis/v3_build.py                 # build + checks, write manifest
    python analysis/v3_build.py --check         # rebuild and compare with the committed manifest
Outputs:
    artifacts/v3_build/features_v3.parquet      (git-ignored; large)
    artifacts/v3_build/features_v3.parquet.provenance.json
    analysis/v3_build_results/build_manifest.json   (committed copy of the manifest)
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model_logic import feature_builder, jt_partnership_features, track_distance_features, trajectory_features
from model_logic import feature_registry
from model_logic.feature_builder import build_features
from model_logic.feature_registry import REGISTRY, REGISTRY_VERSION, ACTIVE_VERSION, registry_hash

DEFAULT_DATA = REPO / "data" / "clean data" / "runners_merged.parquet"
DEFAULT_OUT = REPO / "artifacts" / "v3_build"
MANIFEST = REPO / "analysis" / "v3_build_results" / "build_manifest.json"
BUILDER_MODULES = (feature_builder, jt_partnership_features, track_distance_features,
                   trajectory_features, feature_registry)
KEY = ["source", "race_id", "horse_id"]
RESULT_COLS = [
    "finish_pos", "is_winner", "is_top3", "finish_time_s", "running_positions", "finish_status",
    "finish_code_raw", "finished", "dead_heat", "kaggle_fg_figure", "kaggle_path",
]
MARKET_COLS = ["win_odds", "implied_prob", "market_prob"]
BUILT_PREFIXES = ("hf_", "jt_", "jtp_", "rc_", "td_", "tr_", "int_", "mkt_", "w3_")
V3_COLUMNS = [spec.name for spec in REGISTRY if spec.added_in == "v3.1"]
V3_NUMERIC = [name for name in V3_COLUMNS if name != "td_distance_band"]
HISTORY_COLUMNS = [name for name in V3_NUMERIC if name.startswith(("td_", "tr_fin", "tr_last"))]


# ------------------------------------------------------------------------- hashing
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def builder_source_hash() -> dict[str, str]:
    files = {Path(inspect.getsourcefile(module)).relative_to(REPO).as_posix(): sha256_file(Path(inspect.getsourcefile(module)))
             for module in BUILDER_MODULES}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"combined": combined, "files": files}


def column_hash(frame: pd.DataFrame, column: str) -> str:
    """Hash of one column in key order; NaN is canonicalised, strings hashed as UTF-8."""
    values = frame[column]
    digest = hashlib.sha256()
    if values.dtype == object or pd.api.types.is_string_dtype(values):
        digest.update("\x1f".join(values.astype(str)).encode("utf-8"))
    else:
        array = values.to_numpy(np.float64, na_value=np.nan).copy()
        array[np.isnan(array)] = np.nan
        digest.update(array.tobytes())
    return digest.hexdigest()


def column_summary(frame: pd.DataFrame, column: str) -> dict:
    if column == "td_distance_band":
        return {"counts": {str(k): int(v) for k, v in frame[column].value_counts().sort_index().items()}}
    values = frame[column].astype(float)
    return {"n_missing": int(values.isna().sum()), "sum": float(values.sum()),
            "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


# ------------------------------------------------------------------------- building
def load_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    built = sorted(column for column in raw.columns if column.startswith(BUILT_PREFIXES))
    if built:
        raise SystemExit(f"Input already contains built feature columns {built[:5]}...: "
                         "pass the raw runners file, not a feature file.")
    raw["race_date"] = pd.to_datetime(raw["race_date"])
    return raw


def build(raw: pd.DataFrame) -> pd.DataFrame:
    return build_features(raw).set_index(KEY).sort_index()


def moved(before: pd.DataFrame, after: pd.DataFrame, columns: list[str]) -> pd.Series:
    common = before.index.intersection(after.index)
    a = before.loc[common, columns].to_numpy(float)
    b = after.loc[common, columns].to_numpy(float)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return pd.Series(~same.all(1), index=common)


def shuffle_results(raw: pd.DataFrame, mask: pd.Series, seed: int = 17) -> pd.DataFrame:
    out = raw.copy()
    rng = np.random.default_rng(seed)
    rows = out.index[mask & out["started"].astype(bool)]
    for _, index in out.loc[rows].groupby("race_id").groups.items():
        index = np.asarray(index)
        if len(index) > 1:
            out.loc[index, RESULT_COLS] = out.loc[rng.permutation(index), RESULT_COLS].to_numpy()
    return out


def middle_day(raw: pd.DataFrame, source: str) -> pd.Timestamp:
    days = np.sort(raw.loc[raw["source"].eq(source), "race_date"].unique())
    return pd.Timestamp(days[len(days) // 2])


# --------------------------------------------------------------------------- checks
def run_checks(raw: pd.DataFrame, base: pd.DataFrame) -> dict:
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, detail: str) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")

    sources = [inspect.getsource(module) for module in BUILDER_MODULES]
    record("no_version2_dependence", all("version2_runners_merged" not in text for text in sources),
           "no builder module mentions version2_runners_merged")

    again = build(raw)
    same = base.index.equals(again.index) and not moved(base, again, V3_NUMERIC).any()
    record("deterministic_rebuild", same, "two builds from the same raw file are identical")

    for source in ("kaggle", "hkjc"):
        day = middle_day(raw, source)
        after = build(shuffle_results(raw, raw["source"].eq(source) & raw["race_date"].eq(day)))
        change = moved(base, after, V3_NUMERIC)
        dates = base.loc[change.index, "race_date"]
        in_source = change.index.get_level_values("source") == source
        early = int(change[(dates <= day).to_numpy() | ~in_source].sum())
        later = bool(moved(base, after, HISTORY_COLUMNS)[(dates > day).to_numpy() & in_source].any())
        record(f"same_day_invariance_{source}", early == 0 and later,
               f"results of {day.date()} shuffled: {early} rows on/before that day or in the other "
               f"source moved; later history features moved: {later}")

        cutoff = day
        after = build(raw.loc[raw["source"].ne(source) | (raw["race_date"] <= cutoff)])
        changed = int(moved(base, after, V3_NUMERIC).sum())
        record(f"cutoff_{source}", changed == 0,
               f"races after {cutoff.date()} deleted: {changed} earlier rows changed")

    after = build(shuffle_results(raw, raw["source"].eq("kaggle")))
    change = moved(base, after, V3_NUMERIC)
    hkjc = change.index.get_level_values("source") == "hkjc"
    record("source_isolation", not change[hkjc].any() and change[~hkjc].any(),
           f"all Kaggle results shuffled: {int(change[hkjc].sum())} HKJC rows moved")

    shuffled = raw.copy()
    permuted = shuffled[MARKET_COLS].iloc[np.random.default_rng(5).permutation(len(raw))]
    shuffled[MARKET_COLS] = permuted.set_axis(shuffled.index)
    changed = int(moved(base, build(shuffled), V3_NUMERIC).sum())
    record("market_exclusion", changed == 0, f"odds shuffled: {changed} rows moved")
    return checks


# ------------------------------------------------------------------------ provenance
def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict:
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    remotes = [ref.strip() for ref in git("branch", "-r", "--contains", commit).splitlines() if ref.strip()]
    return {"commit": commit, "worktree_clean": dirty == "", "remote_branches_containing_commit": remotes}


def frozen_definitions() -> dict:
    td, tr = track_distance_features, trajectory_features
    return {
        "distance_bands_m": [[label, None if np.isinf(upper) else int(upper)] for label, upper in td.DISTANCE_BANDS],
        "pseudo_counts": td.K,
        "source_defaults": td.SOURCE_DEFAULTS,
        "days_since_edges": list(tr.DAYS_SINCE_EDGES),
        "class_deadband": tr.CLASS_DEADBAND,
        "field_size_centre": tr.FIELD_SIZE_CENTRE,
    }


def stamp(raw_path: Path, base: pd.DataFrame, checks: dict | None) -> dict:
    frame = base.reset_index()
    import lightgbm, pyarrow, scipy  # noqa: E401  (versions only)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"path": raw_path.relative_to(REPO).as_posix() if raw_path.is_relative_to(REPO) else str(raw_path),
                 "sha256": sha256_file(raw_path)},
        "registry": {"version": REGISTRY_VERSION, "active_version": ACTIVE_VERSION, "hash": registry_hash()},
        "builder_source": builder_source_hash(),
        "git": git_state(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "pyarrow": pyarrow.__version__, "scipy": scipy.__version__, "lightgbm": lightgbm.__version__},
        "build": {"rows": int(len(frame)), "columns": int(frame.shape[1]),
                  "races": int(frame["race_id"].nunique()),
                  "rows_by_source": {k: int(v) for k, v in frame["source"].value_counts().sort_index().items()},
                  "v3_columns": V3_COLUMNS},
        "frozen_definitions": frozen_definitions(),
        "v3_column_sha256": {name: column_hash(frame, name) for name in V3_COLUMNS},
        "v3_column_summary": {name: column_summary(frame, name) for name in V3_COLUMNS},
        "checks": checks,
    }


def compare(reference: dict, current: dict) -> list[str]:
    """Reproduction check: same inputs must give the same v3 columns."""
    problems = []
    for key, label in (("data", "data sha256"), ("registry", "registry hash")):
        ref = reference[key]["sha256" if key == "data" else "hash"]
        cur = current[key]["sha256" if key == "data" else "hash"]
        if ref != cur:
            problems.append(f"{label} differs: manifest {ref[:12]} vs now {cur[:12]}")
    if reference["frozen_definitions"] != current["frozen_definitions"]:
        problems.append("frozen definitions (bands, pseudo-counts, buckets) changed")
    for name in V3_COLUMNS:
        if reference["v3_column_sha256"].get(name) == current["v3_column_sha256"][name]:
            continue
        ref, cur = reference["v3_column_summary"].get(name), current["v3_column_summary"][name]
        close = ref is not None and all(
            np.isclose(ref[k], cur[k], rtol=1e-12, atol=1e-12) if isinstance(ref[k], float) else ref[k] == cur[k]
            for k in ref)
        problems.append(f"{name}: hash differs" + (" (summaries equal to 1e-12: platform rounding)" if close else ""))
    return problems


# ------------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="compare with the committed manifest")
    parser.add_argument("--allow-dirty", action="store_true", help="stamp a build from uncommitted code")
    parser.add_argument("--skip-checks", action="store_true", help="hash only (with --check)")
    args = parser.parse_args()

    raw_path = args.data.resolve()
    print(f"Raw runners: {raw_path}")
    raw = load_raw(raw_path)
    base = build(raw)
    print(f"Built {len(base)} rows x {base.shape[1] + len(KEY)} columns; {len(V3_COLUMNS)} v3.1 columns")

    if args.check:
        reference = json.loads(MANIFEST.read_text(encoding="utf-8"))
        current = stamp(raw_path, base, None)
        problems = compare(reference, current)
        for problem in problems:
            print(f"  DIFF  {problem}")
        exact = not problems
        print("REPRODUCED: every v3 column matches the manifest bit for bit" if exact
              else f"NOT REPRODUCED: {len(problems)} difference(s)")
        return 0 if exact else 1

    state = git_state()
    if not state["worktree_clean"] and not args.allow_dirty:
        raise SystemExit("Refusing to stamp a build from uncommitted code (use --allow-dirty for a draft).")
    print("Checks:")
    checks = None if args.skip_checks else run_checks(raw, base)
    manifest = stamp(raw_path, base, checks)
    failed = [name for name, result in (checks or {}).items() if not result["passed"]]

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "features_v3.parquet"
    base.reset_index().to_parquet(target, index=False)
    manifest["output"] = {"path": target.name, "sha256": sha256_file(target)}
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (args.out / f"{target.name}.provenance.json").write_text(text, encoding="utf-8")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(text, encoding="utf-8")
    print(f"Wrote {target} and {MANIFEST.relative_to(REPO)}")
    print(f"data {manifest['data']['sha256'][:12]}  registry {REGISTRY_VERSION} {manifest['registry']['hash'][:12]}  "
          f"builder {manifest['builder_source']['combined'][:12]}  commit {state['commit'][:7]}"
          f"{'' if state['worktree_clean'] else ' (dirty)'}")
    if failed:
        print(f"FAILED checks: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
