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

Pass `version="v2"` for the old sets. The legacy aliases `feature_builder.ALL_FEATURES`,
`BASE_FEATURES` and `CANDIDATE_FEATURES` stay pinned to v2, so the standalone v2 scripts
reproduce their published numbers. Every v2 column produced by `build_features()` is
bit-for-bit unchanged (`tests/test_feature_context_parity.py`).

Status is `unaudited` for every v3.1 feature until the stability audit assigns keep/watch/review.

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
