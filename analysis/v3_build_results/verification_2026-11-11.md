# Independent re-run of the v3.1 raw build and history-depth tables (11 Nov)

Requested by the 11 Nov status sheet ("v3 raw rebuild/history-depth work: claimed merged, not
rerun here"; priority plan step 2). Everything below was run from a fresh clone of
`Fahim519-ui/HKJC-Racing` `main`, with no local edits, before any output was written.

| Item | Value |
|---|---|
| Commit | `7e51b542e18e6c9b2d31960956a1228f45844a3c` (contained in `origin/main`, worktree clean) |
| Raw data | `data/clean data/runners_merged.parquet`, SHA-256 `d2f8418a5972bd0ede7d809ebf9f204014e7fb534081b53cd926b140794e09b0` |
| Registry | `v3.1`, hash `50ff25aaa1daef9b54bfb1439a41c2386987eb600f033f85abfeec3f07b008ac` |
| Builder source | SHA-256 `07133c4e382fc490919bd6b5ce48698c9d9ad55375f5077a223a30131195c9c5` (unchanged since the 8 Nov reference build) |
| Environment | Python 3.13.18, numpy 2.5.3, pandas 3.0.5, pyarrow 26.0.1 |

## Steps and results

The two test suites were run as separate commands, not chained, as the status sheet asks.

| # | Command | Result |
|---|---|---|
| 1 | `python -m pytest tests/` | 85 passed in 42.11s |
| 2 | `python -m pytest "leakage test/test_leakage.py"` | 26 passed in 19.03s |
| 3 | `python analysis/v3_build.py --check` | REPRODUCED: every v3 column matches the manifest bit for bit |
| 4 | `python analysis/v3_build.py` (full stamped rebuild with checks) | 8 of 8 checks passed |
| 5 | `python analysis/history_depth.py` | the three CSV tables are byte-identical to the committed ones |

Step 4 checks:

- PASS  no_version2_dependence: no builder module mentions version2_runners_merged
- PASS  deterministic_rebuild: two builds from the same raw file are identical
- PASS  same_day_invariance_kaggle: results of 2019-02-24 shuffled: 0 rows on/before that day or in the other source moved; later history features moved: True
- PASS  cutoff_kaggle: races after 2019-02-24 deleted: 0 earlier rows changed
- PASS  same_day_invariance_hkjc: results of 2026-03-01 shuffled: 0 rows on/before that day or in the other source moved; later history features moved: True
- PASS  cutoff_hkjc: races after 2026-03-01 deleted: 0 earlier rows changed
- PASS  source_isolation: all Kaggle results shuffled: 0 HKJC rows moved
- PASS  market_exclusion: odds shuffled: 0 rows moved

## Comparison with the committed outputs

- `build_manifest.json`: every field is identical except `generated_utc` and `git` (commit and
  timestamp of this run). All 19 v3.1 column hashes, the frozen definitions and the summaries match.
- This run used numpy 2.5.3 and pandas 3.0.5, not the versions pinned in `requirements-lock.txt`
  (numpy 2.5.4, pandas 3.1.1), and the v3 columns were still bit-identical.
- The rebuilt feature file has the same SHA-256 as the 8 Nov build (`ea2873583fbdb77b...`).
- `history_depth_by_window.csv`, `starts_per_horse.csv` and `missingness_by_window.csv` are
  byte-identical; the report and `meta.json` differ only in the timestamp and commit line.

The committed manifest and tables are therefore left unchanged: they remain the reference, and
this file records that a clean checkout of `7e51b54` reproduces them.

No model was fitted, no metric was computed, and h_test outcomes were not read.

This re-run was made by the feature owner's session. For a check by a second person, run
step 3 (`python analysis/v3_build.py --check`, about 5 seconds) on another machine.
