# Overfitting and selection checks, v3.1 fold runs (generated)

Source: `artifacts/folds/` (runs on commit `1042a3d`, all pass the provenance check). Built by `analysis/overfitting_check.py`; no model is refitted and no h_test row is read. Bootstrap: 2000 race-day resamples, seed 0.

## 1. In-sample (s2) vs out-of-sample (valid) gain vs calibrated market

Mean in-sample gain +0.00167; mean out-of-sample gain +0.00175; in-sample larger in 15 of 36 cases. **No sign of the blend overfitting.**

| config | blend | fold | gain_in_sample_s2 | gain_out_of_sample_valid | in_minus_out |
|---|---|---|---|---|---|
| no_track_distance | mkt+cl | h1 | 0.00621 | 0.00679 | -0.00058 |
| no_track_distance | mkt+cl | k1 | -3e-05 | -0.00035 | 0.00033 |
| no_track_distance | mkt+cl | k2 | 0.00012 | 0.00077 | -0.00065 |
| no_track_distance | mkt+cl+lgbm | h1 | 0.00712 | 0.00598 | 0.00115 |
| no_track_distance | mkt+cl+lgbm | k1 | 0.0001 | -0.00023 | 0.00032 |
| no_track_distance | mkt+cl+lgbm | k2 | 0.00016 | 0.00094 | -0.00078 |
| no_track_distance | mkt+lgbm | h1 | 0.00577 | 0.00293 | 0.00284 |
| no_track_distance | mkt+lgbm | k1 | -0.0003 | 0.0003 | -0.0006 |
| no_track_distance | mkt+lgbm | k2 | 0.0001 | 0.001 | -0.0009 |
| no_trainer_no_track_distance | mkt+cl | h1 | 0.00389 | 0.00605 | -0.00216 |
| no_trainer_no_track_distance | mkt+cl | k1 | 0.00019 | -0.00057 | 0.00076 |
| no_trainer_no_track_distance | mkt+cl | k2 | 0.00012 | 0.00077 | -0.00065 |
| no_trainer_no_track_distance | mkt+cl+lgbm | h1 | 0.00499 | 0.00656 | -0.00157 |
| no_trainer_no_track_distance | mkt+cl+lgbm | k1 | 0.00022 | -0.00054 | 0.00076 |
| no_trainer_no_track_distance | mkt+cl+lgbm | k2 | 0.00012 | 0.00076 | -0.00064 |
| no_trainer_no_track_distance | mkt+lgbm | h1 | 0.00437 | 0.0045 | -0.00013 |
| no_trainer_no_track_distance | mkt+lgbm | k1 | -0.00025 | 0.00013 | -0.00038 |
| no_trainer_no_track_distance | mkt+lgbm | k2 | -1e-05 | 0.00074 | -0.00075 |
| no_trainer_rate | mkt+cl | h1 | 0.0031 | 0.00444 | -0.00133 |
| no_trainer_rate | mkt+cl | k1 | 0.00011 | -0.00049 | 0.0006 |
| no_trainer_rate | mkt+cl | k2 | -3e-05 | 0.00067 | -0.0007 |
| no_trainer_rate | mkt+cl+lgbm | h1 | 0.00401 | 0.00446 | -0.00045 |
| no_trainer_rate | mkt+cl+lgbm | k1 | 7e-05 | -0.00067 | 0.00074 |
| no_trainer_rate | mkt+cl+lgbm | k2 | 0.00031 | 0.00037 | -5e-05 |
| no_trainer_rate | mkt+lgbm | h1 | 0.00359 | 0.00291 | 0.00068 |
| no_trainer_rate | mkt+lgbm | k1 | -0.00011 | -0.00016 | 5e-05 |
| no_trainer_rate | mkt+lgbm | k2 | 0.00028 | 0.00046 | -0.00019 |
| primary | mkt+cl | h1 | 0.00517 | 0.00512 | 5e-05 |
| primary | mkt+cl | k1 | -0.00011 | -0.00024 | 0.00014 |
| primary | mkt+cl | k2 | -3e-05 | 0.00067 | -0.0007 |
| primary | mkt+cl+lgbm | h1 | 0.00599 | 0.00506 | 0.00092 |
| primary | mkt+cl+lgbm | k1 | -0.00014 | -0.00041 | 0.00027 |
| primary | mkt+cl+lgbm | k2 | 0.00014 | 0.00065 | -0.00051 |
| primary | mkt+lgbm | h1 | 0.00484 | 0.00307 | 0.00177 |
| primary | mkt+lgbm | k1 | -0.00022 | -0.00011 | -0.00011 |
| primary | mkt+lgbm | k2 | 0.00014 | 0.00068 | -0.00054 |

## 2. Stability across folds (primary config)

| fold | cl_features | lgbm_best_iter | lgbm_rounds | seeds | blend_lambda | beta_calibrated_market | mkt+cl: beta | mkt+cl: alpha_cl | mkt+lgbm: alpha_lgbm | mkt+cl+lgbm: alpha_cl | mkt+cl+lgbm: alpha_lgbm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k1 | 12 | 344 | 504 | 17 118 | 0.1 | 1.106 | 1.067 | 0.034 | 0.025 | 0.028 | 0.011 |
| k2 | 9 | 508 | 647 | 17 118 | 0.1 | 1.06 | 1.049 | -0.006 | -0.018 | 0.005 | -0.02 |
| h1 | 9 | 310 | 330 | 17 118 | 0.1 | 1.005 | 1.058 | -0.119 | -0.117 | -0.08 | -0.073 |

Conditional-logit features selected in all three folds: 6 of 16 ever selected (`hf_avg3_fin_pct_rel`, `hf_days_since`, `jt_jockey_win365_rel`, `rc_draw_pct`, `rc_draw_x_hv`, `rc_weight_rel`). Overlap between folds (Jaccard): k1-k2 0.40, k1-h1 0.50, k2-h1 0.64. The model weights (alpha) are small and change sign between folds, and the blend picks the strongest penalty in every fold: **the models add almost no information beyond the calibrated market.**

## 3. Best of N candidates vs chance

Each row bootstraps whole race days with every candidate centred at zero (no real edge) and records the largest gain. `selection_adjusted_p` is the share of resamples whose best gain is at least the observed best.

| folds | candidates | races | race_days | best | best_gain | best_ci_low | best_ci_high | chance_95th_of_max | selection_adjusted_p |
|---|---|---|---|---|---|---|---|---|---|
| k1+k2 | 12 | 756 | 80 | no_track_distance / mkt+lgbm | 0.00065 | -0.00077 | 0.00198 | 0.00144 | 0.309 |
| k1+k2+h1 | 12 | 927 | 97 | no_track_distance / mkt+cl | 0.00142 | -0.00076 | 0.00347 | 0.00203 | 0.167 |

**The best configuration is within what picking the winner of the candidates produces by chance.** Top candidates (pooled k1+k2+h1): `no_track_distance / mkt+cl` +0.00142, `no_track_distance / mkt+cl+lgbm` +0.00139, `no_trainer_no_track_distance / mkt+cl+lgbm` +0.00130, `no_trainer_no_track_distance / mkt+cl` +0.00120, `no_trainer_no_track_distance / mkt+lgbm` +0.00119.

## Figures

- `primary_gain_by_fold.png`: primary blend gain per fold with its 95% race-day interval.
- `best_of_n_vs_chance.png`: the best observed gain against the chance distribution of the best.

## Plain-language reading

The blend is not overfitting: it does as well out of sample as in sample. But the models add almost no information beyond the betting market: their blend weights stay near zero and change sign between folds, the selected logit features are unstable, and the best of the candidate configurations does no better than the luckiest of them would by chance.
