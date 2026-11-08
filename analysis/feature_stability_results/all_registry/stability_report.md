# Feature stability audit (generated)

Generated 2026-11-08 19:05 from `runners_merged.parquet` (SHA-256 `d2f8418a5972...`), repo commit `506a056`. 55 features.

**Result:** 2 review, 20 watch, 33 keep.

Thresholds: SMD 0.10 / 0.25, PSI 0.10 / 0.25, blank rate 5 pp, effect difference 2 SE. Effects use k_train, k_early, k_tune and h_s2 only, never h_test. The h_test columns are information only.

**This is a warning list, not a feature-selection rule.** Features are chosen on k_tune only (docs/DATA_RULES.md). Confirm any change on k_tune before acting on a flag here.

## Summary

Columns: shift of HKJC Stage 2 (h_s2) against Kaggle train, and against Kaggle early (history-matched). Coefficients are per 1 SD of the feature in Kaggle train.

| Feature | Status | SMD h_s2 | SMD vs early | PSI h_s2 | PSI vs early | PSI tune | Blank pp | Coef k_train | Coef k_early | Coef h_s2 | Reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `jt_trainer_win365` | **review** | -0.17 | -0.12 | 0.288 | 0.292 | 0.267 | +0.0 | +0.291 ± 0.025 | +0.332 ± 0.043 | +0.092 ± 0.071 | large Kaggle->HKJC shift, also against the history-matched window; large drift inside Kaggle (train->tune); effect weaker in HKJC, also against the history-matched window |
| `jt_trainer_win365_rel` | **review** | +0.00 | +0.00 | 0.264 | 0.252 | 0.136 | +0.0 | +0.281 ± 0.024 | +0.320 ± 0.042 | +0.088 ± 0.069 | large Kaggle->HKJC shift, also against the history-matched window; moderate drift inside Kaggle (train->tune); effect weaker in HKJC, also against the history-matched window |
| `hf_class_change` | **watch** | +0.01 | -0.01 | 0.094 | 0.001 | 0.122 | +1.5 | +0.111 ± 0.038 | +0.025 ± 0.066 | +0.129 ± 0.069 | moderate drift inside Kaggle (train->tune) |
| `hf_days_since` | **watch** | -0.18 | +0.03 | 0.079 | 0.013 | 0.024 | +1.5 | +0.001 ± 0.033 | +0.152 ± 0.077 | +0.036 ± 0.078 | moderate shift, mostly from shorter HKJC histories |
| `hf_win5_s` | **watch** | -0.00 | +0.00 | 0.312 | 0.012 | 0.090 | +0.0 | +0.341 ± 0.025 | +0.365 ± 0.045 | +0.377 ± 0.046 | large shift, mostly from shorter HKJC histories |
| `jt_jockey_win365` | **watch** | -0.08 | -0.04 | 0.214 | 0.274 | 0.250 | +0.0 | +0.487 ± 0.022 | +0.448 ± 0.046 | +0.378 ± 0.048 | moderate Kaggle->HKJC shift; large drift inside Kaggle (train->tune); effect weaker in HKJC, but not against the history-matched window |
| `jt_jockey_win365_rel` | **watch** | +0.00 | -0.00 | 0.196 | 0.193 | 0.133 | +0.0 | +0.479 ± 0.022 | +0.441 ± 0.045 | +0.372 ± 0.047 | moderate Kaggle->HKJC shift; moderate drift inside Kaggle (train->tune); effect weaker in HKJC, but not against the history-matched window |
| `jt_jockey_win365_rk` | **watch** | -0.00 | -0.00 | 0.002 | 0.001 | 0.001 | +0.0 | +0.551 ± 0.030 | +0.401 ± 0.054 | +0.306 ± 0.050 | effect weaker in HKJC, but not against the history-matched window |
| `jt_trainer_win365_rk` | **watch** | -0.00 | -0.00 | 0.002 | 0.002 | 0.001 | +0.0 | +0.298 ± 0.028 | +0.374 ± 0.053 | +0.084 ± 0.049 | effect weaker in HKJC, also against the history-matched window |
| `jtp_pair_baseline365` | **watch** | -0.16 | -0.12 | 0.116 | 0.116 | 0.125 | +0.0 | +0.511 ± 0.024 | +0.511 ± 0.046 | +0.399 ± 0.055 | moderate Kaggle->HKJC shift; moderate drift inside Kaggle (train->tune) |
| `jtp_pair_rides365` | **watch** | -0.33 | -0.04 | 0.102 | 0.019 | 0.051 | +0.0 | +0.142 ± 0.025 | +0.091 ± 0.074 | +0.168 ± 0.091 | large shift, mostly from shorter HKJC histories; moderate drift inside Kaggle (train->tune) |
| `jtp_pair_rides_last_50` | **watch** | -0.41 | +0.03 | 0.185 | 0.019 | 0.209 | +0.0 | +0.152 ± 0.031 | +0.132 ± 0.061 | +0.136 ± 0.062 | large shift, mostly from shorter HKJC histories; large drift inside Kaggle (train->tune) |
| `jtp_pair_win365` | **watch** | -0.14 | -0.06 | 0.105 | 0.077 | 0.048 | +0.0 | +0.481 ± 0.022 | +0.543 ± 0.048 | +0.449 ± 0.060 | moderate shift, mostly from shorter HKJC histories |
| `jtp_pair_win365_rel` | **watch** | -0.00 | -0.00 | 0.186 | 0.093 | 0.017 | +0.0 | +0.467 ± 0.022 | +0.527 ± 0.047 | +0.436 ± 0.058 | moderate shift, mostly from shorter HKJC histories |
| `jtp_pair_win_last_50` | **watch** | -0.07 | -0.01 | 0.112 | 0.102 | 0.033 | +0.0 | +0.464 ± 0.022 | +0.495 ± 0.044 | +0.420 ± 0.063 | moderate Kaggle->HKJC shift |
| `jtp_pair_win_last_50_rel` | **watch** | -0.00 | -0.00 | 0.149 | 0.116 | 0.015 | +0.0 | +0.451 ± 0.022 | +0.481 ± 0.043 | +0.409 ± 0.061 | moderate Kaggle->HKJC shift |
| `n_runners` | **watch** | +0.24 | +0.10 | 0.057 | 0.016 | 0.016 | +0.0 | race-level | race-level | race-level | moderate Kaggle->HKJC shift; moderate drift inside Kaggle (train->tune) |
| `td_distance_band_starts` | **watch** | -0.63 | -0.03 | 0.793 | 0.009 | 0.277 | +0.0 | +0.030 ± 0.033 | +0.160 ± 0.125 | +0.114 ± 0.127 | large shift, mostly from shorter HKJC histories; large drift inside Kaggle (train->tune) |
| `td_track_distance_starts` | **watch** | -0.52 | +0.00 | 0.553 | 0.003 | 0.207 | +0.0 | +0.093 ± 0.031 | +0.081 ± 0.110 | +0.177 ± 0.109 | large shift, mostly from shorter HKJC histories; large drift inside Kaggle (train->tune) |
| `td_track_starts` | **watch** | -0.68 | -0.03 | 0.910 | 0.001 | 0.357 | +0.0 | +0.056 ± 0.033 | +0.030 ± 0.122 | +0.085 ± 0.121 | large shift, mostly from shorter HKJC histories; large drift inside Kaggle (train->tune) |
| `tr_days_since_bucket` | **watch** | -0.14 | +0.03 | 0.074 | 0.004 | 0.018 | +1.5 | +0.004 ± 0.032 | +0.117 ± 0.068 | +0.066 ± 0.067 | moderate Kaggle->HKJC shift |
| `tr_fin_pct_slope3_rel` | **watch** | +0.00 | +0.00 | 0.027 | 0.002 | 0.017 | +5.6 | -0.173 ± 0.030 | -0.092 ± 0.053 | -0.143 ± 0.050 | blank rate differs by +5.6 pp |
| `hf_avg3_fin_pct` | **keep** | -0.03 | -0.01 | 0.009 | 0.003 | 0.006 | +1.5 | -0.696 ± 0.034 | -0.659 ± 0.061 | -0.642 ± 0.057 | - |
| `hf_avg3_fin_pct_rel` | **keep** | +0.00 | +0.00 | 0.007 | 0.003 | 0.005 | +1.5 | -0.652 ± 0.032 | -0.612 ± 0.058 | -0.603 ± 0.054 | - |
| `hf_avg3_fin_pct_rk` | **keep** | -0.00 | -0.00 | 0.003 | 0.000 | 0.004 | +1.5 | -0.598 ± 0.031 | -0.587 ± 0.058 | -0.602 ± 0.055 | - |
| `hf_class_change_rk` | **keep** | -0.00 | -0.00 | 0.046 | 0.034 | 0.025 | +1.5 | +0.097 ± 0.029 | +0.069 ± 0.056 | +0.108 ± 0.051 | - |
| `hf_first_start` | **keep** | +0.06 | -0.01 | 0.003 | 0.000 | 0.004 | +0.0 | -0.103 ± 0.036 | -0.064 ± 0.056 | +0.005 ± 0.049 | - |
| `hf_last_fin_pct` | **keep** | -0.02 | -0.00 | 0.007 | 0.002 | 0.005 | +1.6 | -0.623 ± 0.033 | -0.570 ± 0.061 | -0.604 ± 0.058 | - |
| `hf_last_fin_pct_rel` | **keep** | +0.00 | +0.00 | 0.006 | 0.003 | 0.005 | +1.6 | -0.589 ± 0.032 | -0.533 ± 0.058 | -0.573 ± 0.055 | - |
| `hf_win5_s_rk` | **keep** | -0.00 | -0.00 | 0.060 | 0.008 | 0.019 | +0.0 | +0.231 ± 0.027 | +0.197 ± 0.049 | +0.255 ± 0.047 | - |
| `int_jt_strength_x_field_size` | **keep** | +0.00 | +0.00 | 0.088 | 0.054 | 0.006 | +0.0 | +0.064 ± 0.026 | +0.142 ± 0.055 | +0.142 ± 0.068 | - |
| `int_recent_form_x_class_change` | **keep** | +0.02 | +0.02 | 0.003 | 0.005 | 0.001 | +0.0 | -0.097 ± 0.026 | -0.081 ± 0.047 | -0.132 ± 0.048 | - |
| `jtp_pair_baseline_last_50` | **keep** | -0.07 | -0.03 | 0.086 | 0.112 | 0.007 | +0.0 | +0.455 ± 0.024 | +0.466 ± 0.046 | +0.330 ± 0.062 | - |
| `jtp_pair_effect365` | **keep** | -0.05 | +0.07 | 0.033 | 0.015 | 0.025 | +0.0 | +0.349 ± 0.023 | +0.435 ± 0.058 | +0.287 ± 0.063 | - |
| `jtp_pair_effect365_rel` | **keep** | +0.00 | +0.00 | 0.047 | 0.009 | 0.010 | +0.0 | +0.334 ± 0.022 | +0.415 ± 0.056 | +0.274 ± 0.060 | - |
| `jtp_pair_effect_last_50` | **keep** | -0.03 | +0.02 | 0.037 | 0.015 | 0.050 | +0.0 | +0.365 ± 0.023 | +0.410 ± 0.047 | +0.297 ± 0.058 | - |
| `jtp_pair_effect_last_50_rel` | **keep** | +0.00 | +0.00 | 0.044 | 0.014 | 0.039 | +0.0 | +0.350 ± 0.022 | +0.393 ± 0.045 | +0.285 ± 0.056 | - |
| `rc_draw_pct` | **keep** | +0.00 | +0.00 | 0.002 | 0.001 | 0.001 | +0.0 | -0.175 ± 0.028 | -0.157 ± 0.052 | -0.196 ± 0.049 | - |
| `rc_draw_x_hv` | **keep** | +0.04 | +0.03 | 0.003 | 0.001 | 0.000 | +0.0 | -0.294 ± 0.045 | -0.184 ± 0.081 | -0.260 ± 0.075 | - |
| `rc_draw_x_sprint` | **keep** | +0.04 | +0.08 | 0.003 | 0.010 | 0.002 | +0.0 | -0.181 ± 0.043 | -0.167 ± 0.082 | -0.152 ± 0.072 | - |
| `rc_weight_rel` | **keep** | -0.00 | -0.00 | 0.039 | 0.062 | 0.006 | +0.0 | +0.178 ± 0.028 | +0.207 ± 0.052 | +0.102 ± 0.053 | - |
| `rc_weight_x_dist` | **keep** | -0.00 | -0.00 | 0.031 | 0.045 | 0.005 | +0.0 | +0.012 ± 0.027 | -0.042 ± 0.048 | -0.060 ± 0.056 | - |
| `td_distance_band_fin_pct_rel` | **keep** | +0.00 | +0.00 | 0.006 | 0.006 | 0.005 | +0.0 | -0.562 ± 0.030 | -0.568 ± 0.057 | -0.580 ± 0.053 | - |
| `td_distance_band_top3_rel` | **keep** | +0.00 | +0.00 | 0.014 | 0.006 | 0.006 | +0.0 | +0.435 ± 0.026 | +0.430 ± 0.045 | +0.466 ± 0.042 | - |
| `td_distance_band_win_rel` | **keep** | +0.00 | -0.00 | 0.045 | 0.017 | 0.035 | +0.0 | +0.326 ± 0.022 | +0.311 ± 0.039 | +0.303 ± 0.039 | - |
| `td_track_distance_fin_pct_rel` | **keep** | +0.00 | -0.00 | 0.004 | 0.002 | 0.003 | +0.0 | -0.600 ± 0.031 | -0.585 ± 0.057 | -0.608 ± 0.054 | - |
| `td_track_distance_top3_rel` | **keep** | +0.00 | +0.00 | 0.016 | 0.009 | 0.007 | +0.0 | +0.449 ± 0.025 | +0.426 ± 0.045 | +0.485 ± 0.042 | - |
| `td_track_distance_win_rel` | **keep** | -0.00 | +0.00 | 0.031 | 0.016 | 0.027 | +0.0 | +0.331 ± 0.022 | +0.302 ± 0.038 | +0.301 ± 0.037 | - |
| `td_track_fin_pct_rel` | **keep** | -0.00 | -0.00 | 0.006 | 0.003 | 0.001 | +0.0 | -0.590 ± 0.031 | -0.607 ± 0.058 | -0.628 ± 0.055 | - |
| `td_track_top3_rel` | **keep** | -0.00 | +0.00 | 0.015 | 0.014 | 0.005 | +0.0 | +0.449 ± 0.026 | +0.440 ± 0.045 | +0.489 ± 0.042 | - |
| `td_track_win_rel` | **keep** | +0.00 | +0.00 | 0.049 | 0.020 | 0.029 | +0.0 | +0.328 ± 0.022 | +0.306 ± 0.038 | +0.313 ± 0.039 | - |
| `tr_class_change_direction` | **keep** | -0.00 | -0.00 | 0.002 | 0.003 | 0.000 | +0.0 | +0.121 ± 0.032 | +0.071 ± 0.058 | +0.134 ± 0.060 | - |
| `tr_last_minus_avg3_fin_pct_rel` | **keep** | -0.00 | -0.00 | 0.010 | 0.002 | 0.005 | +1.6 | -0.152 ± 0.029 | -0.078 ± 0.056 | -0.148 ± 0.054 | - |
| `w3_horse_past_top3_rate` | **keep** | -0.00 | +0.00 | 0.040 | 0.002 | 0.014 | +1.5 | +0.509 ± 0.027 | +0.496 ± 0.048 | +0.517 ± 0.045 | - |
| `w3_horse_past_win_rate` | **keep** | +0.00 | +0.00 | 0.010 | 0.000 | 0.006 | +1.5 | +0.354 ± 0.023 | +0.349 ± 0.039 | +0.310 ± 0.040 | - |

## Agreement with the feature registry

Registry `v3.1` (hash `50ff25aaa1da...`). 56 implemented features; 1 not audited in this run (all registry features). Text columns reported but not audited (no SMD/PSI for categories): `td_distance_band`.

These features have a different status in the registry than in this audit. Ask the registry owner to update the label, or explain why not.

| Feature | This audit | Registry | Why this audit says so |
|---|---|---|---|
| `jtp_pair_baseline365` | **watch** | unaudited | moderate Kaggle->HKJC shift; moderate drift inside Kaggle (train->tune) |
| `jtp_pair_baseline_last_50` | **keep** | unaudited | - |
| `jtp_pair_rides365` | **watch** | unaudited | large shift, mostly from shorter HKJC histories; moderate drift inside Kaggle (train->tune) |
| `jtp_pair_rides_last_50` | **watch** | unaudited | large shift, mostly from shorter HKJC histories; large drift inside Kaggle (train->tune) |
| `jtp_pair_win365` | **watch** | unaudited | moderate shift, mostly from shorter HKJC histories |
| `jtp_pair_win365_rel` | **watch** | keep | moderate shift, mostly from shorter HKJC histories |
| `jtp_pair_win_last_50` | **watch** | unaudited | moderate Kaggle->HKJC shift |
| `jtp_pair_win_last_50_rel` | **watch** | keep | moderate Kaggle->HKJC shift |
| `td_distance_band_starts` | **watch** | unaudited | large shift, mostly from shorter HKJC histories; large drift inside Kaggle (train->tune) |
| `td_track_distance_starts` | **watch** | unaudited | large shift, mostly from shorter HKJC histories; large drift inside Kaggle (train->tune) |
| `td_track_starts` | **watch** | unaudited | large shift, mostly from shorter HKJC histories; large drift inside Kaggle (train->tune) |

## h_test (information only, do not use for feature choice)

| Feature | SMD h_test vs h_s2 | PSI h_test vs h_s2 | Blank pp |
|---|---|---|---|
| `jt_trainer_win365` | +0.03 | 0.117 | +0.0 |
| `jt_trainer_win365_rel` | -0.00 | 0.040 | +0.0 |
| `hf_class_change` | -0.07 | 0.030 | -3.7 |
| `hf_days_since` | +0.42 | 0.254 | -3.7 |
| `hf_win5_s` | -0.01 | 0.712 | +0.0 |
| `jt_jockey_win365` | +0.03 | 0.147 | +0.0 |
| `jt_jockey_win365_rel` | +0.00 | 0.056 | +0.0 |
| `jt_jockey_win365_rk` | +0.00 | 0.000 | +0.0 |
| `jt_trainer_win365_rk` | +0.00 | 0.001 | +0.0 |
| `jtp_pair_baseline365` | +0.04 | 0.014 | +0.0 |
| `jtp_pair_rides365` | +0.68 | 0.505 | +0.0 |
| `jtp_pair_rides_last_50` | +0.74 | 0.520 | +0.0 |
| `jtp_pair_win365` | +0.06 | 0.020 | +0.0 |
| `jtp_pair_win365_rel` | +0.00 | 0.020 | +0.0 |
| `jtp_pair_win_last_50` | -0.01 | 0.037 | +0.0 |
| `jtp_pair_win_last_50_rel` | -0.00 | 0.036 | +0.0 |
| `n_runners` | -0.03 | 0.005 | +0.0 |
| `td_distance_band_starts` | +0.86 | 0.812 | +0.0 |
| `td_track_distance_starts` | +0.72 | 0.489 | +0.0 |
| `td_track_starts` | +0.90 | 0.775 | +0.0 |
| `tr_days_since_bucket` | +0.39 | 0.331 | -3.7 |
| `tr_fin_pct_slope3_rel` | +0.00 | 0.098 | -10.1 |
| `hf_avg3_fin_pct` | +0.01 | 0.037 | -3.7 |
| `hf_avg3_fin_pct_rel` | -0.00 | 0.031 | -3.7 |
| `hf_avg3_fin_pct_rk` | -0.01 | 0.025 | -3.7 |
| `hf_class_change_rk` | -0.01 | 0.075 | -3.7 |
| `hf_first_start` | -0.16 | 0.025 | +0.0 |
| `hf_last_fin_pct` | +0.02 | 0.025 | -3.7 |
| `hf_last_fin_pct_rel` | -0.00 | 0.027 | -3.7 |
| `hf_win5_s_rk` | +0.00 | 0.134 | +0.0 |
| `int_jt_strength_x_field_size` | -0.00 | 0.012 | +0.0 |
| `int_recent_form_x_class_change` | -0.00 | 0.000 | +0.0 |
| `jtp_pair_baseline_last_50` | -0.02 | 0.063 | +0.0 |
| `jtp_pair_effect365` | +0.06 | 0.077 | +0.0 |
| `jtp_pair_effect365_rel` | -0.00 | 0.054 | +0.0 |
| `jtp_pair_effect_last_50` | +0.02 | 0.051 | +0.0 |
| `jtp_pair_effect_last_50_rel` | -0.00 | 0.034 | +0.0 |
| `rc_draw_pct` | +0.00 | 0.000 | +0.0 |
| `rc_draw_x_hv` | -0.08 | 0.008 | +0.0 |
| `rc_draw_x_sprint` | +0.03 | 0.001 | +0.0 |
| `rc_weight_rel` | +0.00 | 0.013 | +0.0 |
| `rc_weight_x_dist` | +0.00 | 0.007 | +0.0 |
| `td_distance_band_fin_pct_rel` | -0.00 | 0.030 | +0.0 |
| `td_distance_band_top3_rel` | -0.00 | 0.023 | +0.0 |
| `td_distance_band_win_rel` | -0.00 | 0.065 | +0.0 |
| `td_track_distance_fin_pct_rel` | -0.00 | 0.018 | +0.0 |
| `td_track_distance_top3_rel` | -0.00 | 0.016 | +0.0 |
| `td_track_distance_win_rel` | +0.00 | 0.052 | +0.0 |
| `td_track_fin_pct_rel` | +0.00 | 0.028 | +0.0 |
| `td_track_top3_rel` | +0.00 | 0.017 | +0.0 |
| `td_track_win_rel` | -0.00 | 0.050 | +0.0 |
| `tr_class_change_direction` | -0.03 | 0.002 | +0.0 |
| `tr_last_minus_avg3_fin_pct_rel` | +0.00 | 0.040 | -3.7 |
| `w3_horse_past_top3_rate` | -0.01 | 0.113 | -3.7 |
| `w3_horse_past_win_rate` | -0.03 | 0.045 | -3.7 |

Files: `stability_summary.csv`, `window_stats.csv`, `smd_by_feature.png`, `audit_meta.json`.
