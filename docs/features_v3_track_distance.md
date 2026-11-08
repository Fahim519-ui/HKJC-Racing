# v3.1 features: track/distance form, trajectory, interactions

Owner: Nayeon Kim. Roadmap of 4 Nov, Works 1, 3 and 4; Task division "Nayeon: features".
Code: `model_logic/track_distance_features.py`, `model_logic/trajectory_features.py`,
called from `build_features()` in `model_logic/feature_builder.py`.
Registry: `model_logic/feature_registry.py` (`added_in="v3.1"`), snapshot
`model_logic/feature_registry_snapshot_v3.1.json`.
Tests: `tests/test_v3_features_leakage.py`.

Every feature is rebuilt from the raw runners file (`data/clean data/runners_merged.parquet`)
by `build_features()`. Nothing reads `version2_runners_merged.parquet` or any other
pre-built feature file.

## Frozen definitions (8 Nov)

The definitions below are frozen for the evaluation rerun. Changing any of them is a new
feature version: bump `REGISTRY_VERSION`, regenerate the build and rerun every fold.
`tests/test_v3_features_leakage.py::test_frozen_definitions` fails if a constant moves, and
`python analysis/v3_build.py --check` fails if any v3 column changes.

| Definition | Frozen value | Where |
|---|---|---|
| Distance bands | sprint <= 1200 m, mile <= 1650 m, middle <= 2000 m, staying > 2000 m | `track_distance_features.DISTANCE_BANDS` |
| Pseudo-counts K | career 5, track 5, distance band 5, track x distance 5 (starts) | `track_distance_features.K` |
| Source defaults (before a source's first day only) | win 1/12, top3 3/12, fin_pct 0.5, weight 1 | `track_distance_features.SOURCE_DEFAULTS` |
| Hierarchy | track x band -> track -> career -> prior-day source rate; band -> career | `add_track_distance_features` |
| History cut-off | strictly earlier race dates; every row of the current date excluded | `prior_day_sums` |
| Layoff buckets | <= 14, 15-28, 29-60, 61-180, > 180 days; NaN on a first start | `trajectory_features.DAYS_SINCE_EDGES` |
| Class-change deadband | 0.10 on the log prize ratio | `trajectory_features.CLASS_DEADBAND` |
| Field-size centre | 12 runners | `trajectory_features.FIELD_SIZE_CENTRE` |

### Stamped build

`python analysis/v3_build.py` regenerates every feature from the raw runners file, runs the
checks below on the v3 columns and writes `analysis/v3_build_results/build_manifest.json`
(the build itself goes to `artifacts/v3_build/`, which git ignores). Reference build:

| Field | Value |
|---|---|
| Raw data | `data/clean data/runners_merged.parquet`, SHA-256 `d2f8418a5972bd0ede7d809ebf9f204014e7fb534081b53cd926b140794e09b0` |
| Registry | `v3.1`, hash `50ff25aaa1daef9b54bfb1439a41c2386987eb600f033f85abfeec3f07b008ac` |
| Builder source (5 files) | SHA-256 `07133c4e382fc490919bd6b5ce48698c9d9ad55375f5077a223a30131195c9c5` |
| Code | commit `1b986de` on `main`, clean worktree |
| Rows | 38942 runners (26924 Kaggle, 12018 HKJC), 3179 races |

Checks, all passed: no builder module mentions `version2_runners_merged`; an input that
already carries built feature columns is refused; two builds are identical; shuffling one
day's results moves nothing on or before that day or in the other source (Kaggle and HKJC);
deleting every race after a mid-season date leaves earlier rows identical (both sources);
shuffling all Kaggle results moves no HKJC row; shuffling odds moves nothing.

To confirm a checkout reproduces the reference build bit for bit:

```
python analysis/v3_build.py --check
```

History depth and missingness for the stability report: `python analysis/history_depth.py`
writes `analysis/history_depth_results/`.

## Rules every feature follows

1. **Earlier dates only.** History totals are taken over strictly earlier race dates. Every
   row of the current date is excluded, including earlier races that day. The totals are
   exclusive cumulative sums per group, so a value is bit-for-bit unchanged if any race on
   or after its date is altered or deleted.
2. **Source-separated.** Kaggle (2017-20) and HKJC (2025-26) histories never mix: every
   group key starts with `source`.
3. **No whole-period statistics.** Priors are either constants or running totals of earlier
   dates. Layoff buckets use fixed edges, not quantiles of the data.
4. **No post-race or market inputs** for the current race.

## Track/distance form (9 model features)

For the horse's history at today's track, today's distance band and today's
track x distance-band combination, three measures are tracked:

| Measure | Outcome per earlier start | Better is |
|---|---|---|
| `win` | 1 if won | higher |
| `top3` | 1 if placed 1-3 | higher |
| `fin_pct` | (finish position - 1) / (starters - 1); non-finishers skipped | lower |

**Distance bands:** sprint <= 1200 m (1000, 1200), mile <= 1650 m (1400, 1600, 1650),
middle <= 2000 m (1800, 2000), staying > 2000 m (2200, 2400).

**Hierarchical smoothing.** Each level shrinks toward its parent with a pseudo-count K (in starts):

```
source prior   = (earlier-date source total + default) / (earlier-date source starts + 1)
                 defaults: win 1/12, top3 3/12, fin_pct 0.5 (used only before a source's first day)
career         = (horse total + 5 * source prior) / (horse starts + 5)
track          = (horse-at-track total + 5 * career) / (starts there + 5)
distance band  = (horse-in-band total + 5 * career) / (starts there + 5)
track x band   = (horse-at-track-and-band total + 5 * track) / (starts there + 5)
```

A horse with no starts at a level gets its parent's value. The estimate moves to its own
record as starts accumulate. K = 5 at every level is fixed in advance and not tuned on
any split. It can be tuned inside each fold's training period later. The model features
are race-relative: value minus the race mean.

| Feature | Level | Measure | Monotone |
|---|---|---|---|
| `td_track_win_rel` | track | win | +1 |
| `td_track_top3_rel` | track | top3 | +1 |
| `td_track_fin_pct_rel` | track | fin_pct | -1 |
| `td_distance_band_win_rel` | distance band | win | +1 |
| `td_distance_band_top3_rel` | distance band | top3 | +1 |
| `td_distance_band_fin_pct_rel` | distance band | fin_pct | -1 |
| `td_track_distance_win_rel` | track x band | win | +1 |
| `td_track_distance_top3_rel` | track x band | top3 | +1 |
| `td_track_distance_fin_pct_rel` | track x band | fin_pct | -1 |

Helper columns (registered, not in any model set): `td_distance_band` and the history
depths `td_track_starts`, `td_distance_band_starts` and `td_track_distance_starts`, for the
stability audit.

## Trajectory (4 model features)

| Feature | Definition | Window |
|---|---|---|
| `tr_last_minus_avg3_fin_pct_rel` | last finish percentile minus mean of last three, race-relative. Negative = latest run better than recent average | last 3 starts |
| `tr_fin_pct_slope3_rel` | least-squares slope of the last (up to) three finish percentiles per start, race-relative; needs 2 finishes. Negative = improving | last 3 starts |
| `tr_days_since_bucket` | days since previous start: 0 <= 14, 1 = 15-28, 2 = 29-60, 3 = 61-180, 4 > 180; NaN on a first start (`hf_first_start` flags it) | previous start |
| `tr_class_change_direction` | sign of the log prize ratio to the previous start: +1 up, -1 down; changes within 10 % (seasonal prize adjustments) and first starts are 0 | previous start |

## Interactions (2 model features)

| Feature | Definition |
|---|---|
| `int_recent_form_x_class_change` | `hf_avg3_fin_pct_rel` (0 if missing) x `tr_class_change_direction` |
| `int_jt_strength_x_field_size` | mean of `jt_jockey_win365_rel` and `jt_trainer_win365_rel` x (`n_runners` - 12) / 12 |

`int_jt_strength_x_field_size` uses trainer rates, so it is in `TRAINER_RATE_FEATURES`
and is dropped by the `no_trainer_rate` ablations.

## Model sets

`feature_registry.ACTIVE_VERSION = "v3"`. Registry-native callers (`run_folds.py`) now get:

| Call | v2 (frozen) | v3 (active) |
|---|---|---|
| `get_features("conditional_logit")` | 17 | 32 = 17 + 15 |
| `get_features("lgbm")` | 23 | 38 = 23 + 15 |

Pass `version="v2"` for the old sets. The scripts that read the registry directly
(`conditional_logit_tuning_v2.py`, `lightgbm_independent_probability_v2.py`, the
market-offset scripts and the stability audit) now use the v3 sets by default. To reproduce
their published v2 numbers, set `ACTIVE_VERSION = "v2"` or pass `version="v2"`. The deprecated
aliases `feature_builder.ALL_FEATURES`, `BASE_FEATURES` and `CANDIDATE_FEATURES` stay pinned
to v2. Every v2 column produced by `build_features()` is bit-for-bit unchanged
(`tests/test_feature_context_parity.py`).

Status comes from the stability audit (`analysis/feature_stability_results/stability_report.md`,
6 Nov): 13 of the 15 audited v3.1 model features are `keep`; `tr_days_since_bucket` and
`tr_fin_pct_slope3_rel` are `watch`. The `td_*_starts` history-depth columns and
`td_distance_band` are not model inputs and stay `unaudited`.

## Exploratory check (not used for any decision)

A ridge conditional logit was fitted on `k_train` and scored on `k_tune`. HKJC data was not
used, and `h_test` was not touched.

| Feature set | k_tune log loss | Gain over v2 |
|---|---|---|
| v2 conditional-logit 17 | 2.1807 | - |
| + 9 track/distance | 2.1745 | +0.0062 |
| + 6 trajectory/interaction | 2.1804 | +0.0003 |
| + all 15 | 2.1745 | +0.0062 |

This is a single split. Whether each feature is kept is decided by the fold runner's
forward selection and ablations (`no_track_distance`).

## Tests (`tests/test_v3_features_leakage.py`)

- **Same-day shuffle:** one day's results are permuted within races. No v3 feature moves on
  that day, on earlier days, or in the other source, and later history features do move.
- **Cut-off:** races after a mid-season date are deleted. Every v3 feature on or before the
  date is identical.
- **Source isolation:** shuffling all Kaggle results leaves every HKJC value unchanged.
- **Market exclusion:** shuffling odds moves nothing.
- **Construction:**
  - race-relative features have mean 0 in every race;
  - a level with no history equals its parent;
  - bucket and direction values are in their fixed sets;
  - every model feature is registered and documented here;
  - the builder does not depend on `version2_runners_merged.parquet`.
