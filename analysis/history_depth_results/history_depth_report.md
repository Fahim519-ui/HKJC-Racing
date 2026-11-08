# History depth and missingness (generated)

Generated 2026-11-08T07:16:34+00:00 from `data/clean data/runners_merged.parquet` (SHA-256 `d2f8418a5972...`), registry `v3.1` (hash `50ff25aaa1da...`), commit `1b986de`.

Owner: Nayeon. For Inns's stability report (status sheet of 8 Nov, Tier 1). Only pre-race information is read: start counts come from earlier race dates, and missingness is whether a value exists. **h_test is information only** and must not be used to keep or drop a feature.

## 1. Earlier-date starts at race time (per runner row)

How much history each runner had when its features were built. `career` is every earlier start in the source; the other levels count only earlier starts at today's track, distance band, or both. With pseudo-count K = 5, a level with fewer than 5 starts is still at least half its parent (see `docs/features_v3_track_distance.md`).

| level | window | source | n_rows | mean | p25 | p50 | p75 | zero_% |
|---|---|---|---|---|---|---|---|---|
| career | k_train | kaggle | 15836 | 7.3 | 3.0 | 6.0 | 11.0 | 6.6 |
| career | k_early | kaggle | 4711 | 3.9 | 2.0 | 4.0 | 6.0 | 8.3 |
| career | k_tune | kaggle | 9299 | 12.6 | 5.0 | 11.0 | 19.0 | 5.2 |
| career | h_s2 | hkjc | 5287 | 3.7 | 2.0 | 3.0 | 5.0 | 8.0 |
| career | h_test | hkjc | 4978 | 7.4 | 5.0 | 8.0 | 10.0 | 4.3 |
| distance_band | k_train | kaggle | 15836 | 4.9 | 1.0 | 4.0 | 7.0 | 14.7 |
| distance_band | k_early | kaggle | 4711 | 2.7 | 1.0 | 2.0 | 4.0 | 19.4 |
| distance_band | k_tune | kaggle | 9299 | 8.3 | 2.0 | 6.0 | 13.0 | 10.5 |
| distance_band | h_s2 | hkjc | 5287 | 2.6 | 1.0 | 2.0 | 4.0 | 18.2 |
| distance_band | h_test | hkjc | 4978 | 5.2 | 2.0 | 5.0 | 8.0 | 10.3 |
| track | k_train | kaggle | 15836 | 4.7 | 1.0 | 4.0 | 7.0 | 12.7 |
| track | k_early | kaggle | 4711 | 2.6 | 1.0 | 2.0 | 4.0 | 17.4 |
| track | k_tune | kaggle | 9299 | 8.0 | 2.0 | 7.0 | 12.0 | 9.6 |
| track | h_s2 | hkjc | 5287 | 2.5 | 1.0 | 2.0 | 4.0 | 17.8 |
| track | h_test | hkjc | 4978 | 5.0 | 2.0 | 5.0 | 7.0 | 9.0 |
| track_distance | k_train | kaggle | 15836 | 3.2 | 1.0 | 2.0 | 5.0 | 23.0 |
| track_distance | k_early | kaggle | 4711 | 1.8 | 0.0 | 1.0 | 3.0 | 30.0 |
| track_distance | k_tune | kaggle | 9299 | 5.4 | 1.0 | 4.0 | 8.0 | 17.1 |
| track_distance | h_s2 | hkjc | 5287 | 1.8 | 0.0 | 1.0 | 3.0 | 28.6 |
| track_distance | h_test | hkjc | 4978 | 3.6 | 1.0 | 3.0 | 6.0 | 16.9 |

Full table with p10/p90, share below K and the fold windows: `history_depth_by_window.csv`.

## 2. Starts per horse within each source

| source | horses | starts total | starts mean | starts median | tracks mean | tracks median | distance bands mean | distance bands median | track distance mean | track distance median | starts p90 | pct horses 1 start | pct horses ge5 starts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hkjc | 1501 | 12018 | 8.01 | 8.00 | 1.62 | 2.00 | 1.68 | 2.00 | 2.37 | 2.00 | 14.00 | 7.53 | 72.82 |
| kaggle | 2296 | 26924 | 11.73 | 10.00 | 1.71 | 2.00 | 1.87 | 2.00 | 2.80 | 3.00 | 25.00 | 8.93 | 74.74 |

## 3. Absolute missingness of model inputs (% of rows)

Inputs with any missing value in the main windows; the other 27 model inputs are never missing. Counts and the fold windows: `missingness_by_window.csv`.

| feature | k_train | k_early | k_tune | h_s2 | h_test |
|---|---|---|---|---|---|
| tr_fin_pct_slope3_rel | 13.36 | 18.38 | 9.79 | 18.93 | 8.84 |
| hf_last_fin_pct_rel | 6.55 | 8.26 | 5.15 | 8.11 | 4.40 |
| tr_last_minus_avg3_fin_pct_rel | 6.55 | 8.26 | 5.15 | 8.11 | 4.40 |
| hf_class_change | 6.55 | 8.26 | 5.15 | 8.02 | 4.30 |
| hf_avg3_fin_pct_rk | 6.55 | 8.26 | 5.15 | 8.02 | 4.30 |
| hf_avg3_fin_pct_rel | 6.55 | 8.26 | 5.15 | 8.02 | 4.30 |
| hf_class_change_rk | 6.55 | 8.26 | 5.15 | 8.02 | 4.30 |
| tr_days_since_bucket | 6.55 | 8.26 | 5.15 | 8.02 | 4.30 |
| hf_days_since | 6.55 | 8.26 | 5.15 | 8.02 | 4.30 |
| w3_horse_past_top3_rate | 6.55 | 8.26 | 5.15 | 8.02 | 4.30 |
| w3_horse_past_win_rate | 6.55 | 8.26 | 5.15 | 8.02 | 4.30 |

Never missing: `hf_first_start`, `hf_win5_s`, `hf_win5_s_rk`, `int_jt_strength_x_field_size`, `int_recent_form_x_class_change`, `jt_jockey_win365_rel`, `jt_jockey_win365_rk`, `jt_trainer_win365_rel`, `jt_trainer_win365_rk`, `jtp_pair_effect365_rel`, `jtp_pair_effect_last_50_rel`, `n_runners`, `rc_draw_pct`, `rc_draw_x_hv`, `rc_draw_x_sprint`, `rc_weight_rel`, `rc_weight_x_dist`, `td_distance_band_fin_pct_rel`, `td_distance_band_top3_rel`, `td_distance_band_win_rel`, `td_track_distance_fin_pct_rel`, `td_track_distance_top3_rel`, `td_track_distance_win_rel`, `td_track_fin_pct_rel`, `td_track_top3_rel`, `td_track_win_rel`, `tr_class_change_direction`.

## Notes for the stability report

- Missing values come from short histories: a first start has no previous finish, and `tr_fin_pct_slope3_rel` needs two earlier finishes. Their rates follow the zero-start shares in table 1.
- The `td_*_rel` rates are never missing by construction: a runner without history at a level inherits the parent level (track x distance -> track -> career -> prior-day source rate). Their shift between sources is therefore driven by history depth, not by blanks.
- HKJC history starts on 2025-09-07 and Kaggle and HKJC histories are never mixed, so h_s2 runners have much shorter histories than k_train or k_tune at every level, close to k_early (the history-matched Kaggle window). Compare h_s2 with k_early before reading a Kaggle->HKJC shift as a change in racing. By h_test, HKJC depth is back near k_train.
