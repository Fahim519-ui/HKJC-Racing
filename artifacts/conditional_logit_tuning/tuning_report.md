# Conditional-logit tuning report

## Status: SELECTED — DOES NOT BEAT MARKET

Validation window: Kaggle 2019/20 only (`k_tune`).
- Market per-race log loss: **1.9947**
- Selected conditional-logit per-race log loss: **2.1801**
- Market improvement: **-0.1854** (positive means the model is better)

The model used no odds or market-derived values as Stage-1 features. The Apr-Oct 2026 HKJC test window was predicted but not scored.

## Week-3 feature integration

The input table includes the Week-3 engineered columns. The model recomputes `w3_horse_past_win_rate` and `w3_horse_past_top3_rate` from earlier starts within each source before considering them for selection.

It excludes `early_position` and `position_change`: they are parsed from the current race's running positions and are available only after that race. It also excludes the pushed `jockey_past_win_rate` and `trainer_past_win_rate`: their values change between a person's rides on the same day. The model instead uses the existing leakage-safe 365-day `jt_*` features.

## Selected features

- `jt_jockey_win365_rel`
- `hf_avg3_fin_pct_rel`
- `rc_draw_pct`
- `rc_draw_x_hv`
- `rc_weight_rel`
- `hf_days_since`
- `hf_last_fin_pct_rel`
- `hf_first_start`
- `jt_trainer_win365_rel`
- `hf_win5_s`

## Selection trace

```
 step             candidate  log_loss  improvement  accepted                                                                                                                                                                           selected_features
    0                (base)    2.1964          NaN      True                                                                                          jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since
    1   hf_last_fin_pct_rel    2.1871       0.0093      True                                                                     jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel
    2        hf_first_start    2.1848       0.0023      True                                                     jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel, hf_first_start
    3 jt_trainer_win365_rel    2.1822       0.0025      True                              jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel, hf_first_start, jt_trainer_win365_rel
    4             hf_win5_s    2.1801       0.0021      True                   jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel, hf_first_start, jt_trainer_win365_rel, hf_win5_s
    5      rc_draw_x_sprint    2.1794       0.0007     False jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel, hf_first_start, jt_trainer_win365_rel, hf_win5_s, rc_draw_x_sprint
```

## Correlation check

See `feature_correlations.csv`. Any absolute Spearman correlation above 0.9 must be resolved before using a feature set for the final model. The conditional-logit coefficient estimates fitted on `k_train` are in `tuning_coefficients.csv`.
