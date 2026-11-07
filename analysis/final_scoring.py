"""Week 5 final scoring: score the frozen predictions on h_test ONCE.

Three modes, run in this order:

  1. Dry run (default, safe any time). Scores the h_s2 rows only, with exactly the code path
     the final run uses. h_s2 is the Stage-2 fitting window, so these numbers are IN-SAMPLE
     for p_joint and p_market_refit: a smoke test of the pipeline, not evidence.
         python analysis/final_scoring.py

  2. Freeze. Records what will be scored: SHA-256 and MD5 of the predictions file, git commit
     (and whether the tree is clean), row counts per split, columns, and optionally the S3
     version id of the uploaded copy. Reads no outcome of any h_test race.
         python analysis/final_scoring.py --freeze [--s3-uri s3://<bucket>/predictions/combined_predictions.parquet]

  3. Final, one shot. Refuses unless: the freeze record exists, the predictions file still
     matches its SHA-256, the confirmation phrase is given, and no final result exists yet.

Provenance (added 7 Nov, analysis/provenance.py): --freeze and --final also refuse unless the
predictions file has a matching .provenance.json sidecar and every model run it was built
from recorded the live registry version and a git commit that is on origin/main. Run
`git fetch origin` first. The dry run only warns, so the pipeline can still be tested on
older files. This stops a stale run (like the 6 Nov folds: registry v3.0, unpublished commit
fc759fe) from being frozen or scored.
         python analysis/final_scoring.py --final --confirm "score h_test once"

Rules kept here (docs/DATA_RULES.md, WEEK5_PLAN.md):
  * Nothing is refitted on h_test. p_market_refit carries the beta fitted on h_s2
    (build_combined_predictions.py); the Stage-2 blend p_joint was fitted on h_s2.
  * All models are scored on the same races (every column present, market complete).
  * The CI resamples whole race days (eval_utils.paired_bootstrap).
  * The final output folder is written once; a second --final run is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import eval_utils as ev  # noqa: E402
import provenance as prov  # noqa: E402

PREDICTIONS = REPO / "artifacts/combined/combined_predictions.parquet"
OUT_ROOT = REPO / "analysis/final_scoring_results"
FREEZE = OUT_ROOT / "freeze_record.json"
CONFIRM = "score h_test once"
BASELINE = "p_market_refit"
RAW_MARKET = "p_market"
# Every model column we may score; absent ones (e.g. Track B before it exists) are skipped.
CANDIDATES = ["p_uniform", "p_market", "p_power", "p_shin", "p_market_refit",
              "p_cl", "p_lgbm", "p_joint", "p_b_base", "p_b_rating", "p_b_running", "p_b_all"]


# ------------------------------------------------------------------------- helpers
def file_hashes(path: Path) -> dict[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              text=True, check=False).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "dirty_files": [l for l in run("status", "--porcelain").splitlines() if l]}


def s3_version(uri: str, profile: str, region: str) -> dict[str, object]:
    """VersionId and ETag of the uploaded copy (no data is downloaded)."""
    if not uri.startswith("s3://"):
        raise ValueError("--s3-uri must start with s3://")
    bucket, _, key = uri[5:].partition("/")
    cmd = ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key,
           "--region", region, "--profile", profile, "--output", "json"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"aws s3api head-object failed: {res.stderr.strip()}")
    head = json.loads(res.stdout)
    return {"uri": uri, "version_id": head.get("VersionId"),
            "etag": head.get("ETag", "").strip('"'), "last_modified": head.get("LastModified")}


def model_columns(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    cols = requested or CANDIDATES
    return [c for c in cols if c in df.columns]


def check_probabilities(df: pd.DataFrame, cols: list[str]) -> None:
    """Every scored column must be a probability distribution within each race."""
    for c in cols:
        if df[c].isna().any():
            raise ValueError(f"{c} has NaN")
        sums = df.groupby("race_id")[c].sum()
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise ValueError(f"{c} does not sum to 1 within every race")


def score(df: pd.DataFrame, cols: list[str], final_run: bool) -> tuple[pd.DataFrame, list[str]]:
    check_probabilities(df, cols)
    table = ev.scoreboard(df, cols, baseline=BASELINE, raw_market=RAW_MARKET, final_run=final_run)
    lines = [f"Races scored: {table.attrs['n_races']}  (bootstrap: {table.attrs['bootstrap']}, "
             f"{table.attrs['n_days']} race days)"]
    for _, row in table.iterrows():
        lines.append(f"- `{row['model']}`: log loss {row['log_loss']:.4f}, pseudo-R2 "
                     f"{row['pseudo_r2']:.4f}; vs refit market: {ev.verdict(row, 'vs_refit')}; "
                     f"vs raw market: {ev.verdict(row, 'vs_market')}")
    return table, lines


def write_outputs(out: Path, df: pd.DataFrame, cols: list[str], table: pd.DataFrame,
                  lines: list[str], header: list[str], overwrite: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=overwrite)
    table.to_csv(out / "scoreboard.csv", index=False)
    scored = df.loc[ev._complete_race_mask(df, cols)]          # same races as the scoreboard
    calib = []
    for c in cols:
        t = ev.calibration_table(scored[c], scored).reset_index()
        t.insert(0, "model", c)
        calib.append(t.rename(columns={"_bin": "bin"}))
    pd.concat(calib).to_csv(out / "calibration.csv", index=False)
    show = table.copy()
    num = show.select_dtypes("number").columns
    show[num] = show[num].round(4)
    text = header + ["", "## Verdicts", "", *lines, "", "## Scoreboard", "", "```",
                     show.to_string(index=False), "```", "",
                     "Gain > 0 means the model has lower log loss than the baseline. 'Better' "
                     "needs the whole 95% CI above 0."]
    (out / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- modes
def provenance_report(args) -> "prov.Report":
    return prov.check_predictions_file(args.predictions, remote_ref=args.remote_ref)


def dry_run(args) -> int:
    df = ev.load_predictions(args.predictions, split="h_s2")       # h_s2 only, guarded
    cols = model_columns(df, args.models)
    checked = provenance_report(args)
    if not checked.ok:
        print("Provenance check FAILED (dry run continues; --freeze and --final would refuse):")
    print("\n".join(f"  {line}" for line in checked.lines()))
    table, lines = score(df, cols, final_run=False)
    out = args.output_root / "dry_run_h_s2"
    header = ["# Final-scoring DRY RUN on h_s2 (not the test)", "",
              f"Predictions: `{args.predictions.name}`, SHA-256 {file_hashes(args.predictions)['sha256']}.",
              "h_s2 is the Stage-2 fitting window: p_joint and p_market_refit were fitted on it, "
              "so these numbers are in-sample. This checks the pipeline only; it is not evidence.",
              "", "## Provenance", "",
              ("Passed: every source run used the live registry and published code."
               if checked.ok else
               "**FAILED.** These predictions could not be frozen or scored on h_test as they are:"),
              *[f"- {line}" for line in checked.lines() if not line.startswith("provenance OK")]]
    write_outputs(out, df, cols, table, lines, header, overwrite=True)
    print("\n".join(lines))
    print(f"\nDry run written to {out}")
    return 0


def freeze(args) -> int:
    if args.freeze_record.exists():
        raise SystemExit(f"{args.freeze_record} already exists; delete it deliberately to re-freeze.")
    checked = provenance_report(args)
    prov.require(checked, "--freeze")
    meta = pd.read_parquet(args.predictions, columns=["race_id", "split"])   # no outcomes read
    cols = list(pq.read_schema(args.predictions).names)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "predictions": (str(args.predictions.resolve().relative_to(REPO))
                        if args.predictions.resolve().is_relative_to(REPO) else str(args.predictions)),
        **file_hashes(args.predictions),
        "rows_per_split": meta["split"].value_counts().to_dict(),
        "races_per_split": meta.groupby("split")["race_id"].nunique().to_dict(),
        "columns": cols,
        "models_to_score": model_columns(pd.DataFrame(columns=cols), args.models),
        "baseline": BASELINE,
        "git": git_state(),
        "provenance": {"registry": prov.current_registry(), "remote_ref": args.remote_ref,
                       "sources": checked.checked, "warnings": checked.warnings},
    }
    if args.s3_uri:
        s3 = s3_version(args.s3_uri, args.profile, args.region)
        s3["etag_matches_local_md5"] = s3["etag"] == record["md5"]
        record["s3"] = s3
    if record["git"]["dirty_files"]:
        print("WARNING: git tree is not clean; the freeze record lists the dirty files.")
    args.freeze_record.parent.mkdir(parents=True, exist_ok=True)
    args.freeze_record.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, default=str))
    return 0


def final(args) -> int:
    if args.confirm != CONFIRM:
        raise SystemExit(f'Refusing: pass --confirm "{CONFIRM}" for the one-shot test scoring.')
    if not args.freeze_record.exists():
        raise SystemExit("Refusing: no freeze record. Run --freeze first and commit it.")
    record = json.loads(args.freeze_record.read_text())
    if file_hashes(args.predictions)["sha256"] != record["sha256"]:
        raise SystemExit("Refusing: predictions file changed since the freeze (SHA-256 differs).")
    prov.require(provenance_report(args), "--final")
    out = args.output_root / "h_test"
    if out.exists():
        raise SystemExit(f"Refusing: {out} exists. h_test is scored once.")
    df = ev.load_predictions(args.predictions, split=None, final_run=True)
    df = df.loc[df["split"].eq("h_test")].reset_index(drop=True)
    cols = [c for c in record["models_to_score"] if c in df.columns]
    table, lines = score(df, cols, final_run=True)
    header = ["# FINAL h_test scoring (one shot)", "",
              f"Run at {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC from git "
              f"{git_state()['commit']}.",
              f"Predictions SHA-256 {record['sha256']} (matches freeze record of {record['created_utc']}).",
              "Nothing was refitted on h_test: p_market_refit and p_joint carry h_s2 fits."]
    write_outputs(out, df, cols, table, lines, header)
    print("\n".join(lines))
    print(f"\nFinal results written to {out}. Commit this folder; do not re-run.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true", help="write the freeze record")
    mode.add_argument("--final", action="store_true", help="one-shot h_test scoring")
    parser.add_argument("--confirm", default="", help=f'must be "{CONFIRM}" with --final')
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--models", nargs="*", default=None, help="model columns (default: all present)")
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--freeze-record", type=Path, default=FREEZE)
    parser.add_argument("--s3-uri", default=None)
    parser.add_argument("--profile", default="cloudpc")
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--remote-ref", default=prov.REMOTE_REF,
                        help="branch whose history must contain each run's git commit")
    args = parser.parse_args(argv)
    if args.final:
        return final(args)
    if args.freeze:
        return freeze(args)
    return dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
