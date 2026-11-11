# Conditional-logit tuning report

## Status: SELECTED - standalone DOES NOT BEAT market; blend ADDS information over market

Validation window: Kaggle 2019/20 (`k_tune`), races with complete market prices.
- Market per-race log loss: **1.9949**
- Selected conditional logit: **2.3238** (gain -0.3289)
- Benter blend softmax(a*mkt + b*u): **1.9947** (gain +0.0002; a=1.013, b=0.041; 2 in-sample parameters)
- Within-race corr(utility, market logit): 0.680

Hyper-parameters: lambda=0.0, l1_ratio=0.5, lambda_mkt=2.0; stage weights=[1.0].

Odds are used only inside the training-time orthogonality penalty; no odds or market-derived value is a model input. The Apr-Oct 2026 HKJC window is predicted, never scored. Feature selection and penalty choice both use `k_tune`, so its gains are mildly optimistic.

## Selected features

- `jt_jockey_win365_rel`
- `hf_avg3_fin_pct_rel`
- `rc_draw_pct`
- `rc_draw_x_hv`
- `rc_weight_rel`
- `hf_days_since`
- `tr_fin_pct_slope3_rel`
- `jtp_pair_effect365_rel`
- `hf_first_start`
- `hf_win5_s`

## Selection trace

```
 step              candidate  log_loss  improvement  accepted           note                                                                                                                                                                              selected_features
    0                 (base)    2.1967          NaN      True                                                                                                            jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since
    1   tr_days_since_bucket       NaN          NaN     False |spearman|>0.9                                                                                                                                                                                               
    2  tr_fin_pct_slope3_rel    2.1862       0.0105      True                                                                                     jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, tr_fin_pct_slope3_rel
    3 jtp_pair_effect365_rel    2.1816       0.0046      True                                                             jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, tr_fin_pct_slope3_rel, jtp_pair_effect365_rel
    4         hf_first_start    2.1793       0.0023      True                                             jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, tr_fin_pct_slope3_rel, jtp_pair_effect365_rel, hf_first_start
    5              hf_win5_s    2.1778       0.0015      True                                  jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, tr_fin_pct_slope3_rel, jtp_pair_effect365_rel, hf_first_start, hf_win5_s
    6       rc_draw_x_sprint    2.1770       0.0008     False                jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, tr_fin_pct_slope3_rel, jtp_pair_effect365_rel, hf_first_start, hf_win5_s, rc_draw_x_sprint
```

## Penalty grid (k_tune)

```
  lam  lam_mkt  market_nll  model_nll  blend_nll  blend_a_mkt  blend_b_model  corr_mkt  blend_gain  model_gain  nnz
0.000      0.0      1.9949     2.1778     1.9947       1.0325        -0.0141    0.8031      0.0002     -0.1829   10
0.000      0.1      1.9949     2.1796     1.9947       1.0293        -0.0108    0.7976      0.0002     -0.1846   10
0.000      0.5      1.9949     2.2181     1.9948       1.0213         0.0030    0.7708      0.0002     -0.2231   10
0.000      2.0      1.9949     2.3238     1.9947       1.0131         0.0407    0.6803      0.0002     -0.3289   10
0.003      0.0      1.9949     2.1777     1.9947       1.0324        -0.0141    0.8030      0.0002     -0.1827   10
0.003      0.1      1.9949     2.1798     1.9947       1.0292        -0.0107    0.7974      0.0002     -0.1849   10
0.003      0.5      1.9949     2.2187     1.9948       1.0213         0.0031    0.7707      0.0002     -0.2238   10
0.003      2.0      1.9949     2.3245     1.9947       1.0139         0.0373    0.6809      0.0002     -0.3295   10
0.010      0.0      1.9949     2.1775     1.9947       1.0321        -0.0139    0.8025      0.0002     -0.1826   10
0.010      0.1      1.9949     2.1806     1.9947       1.0290        -0.0105    0.7970      0.0002     -0.1857   10
0.010      0.5      1.9949     2.2203     1.9948       1.0211         0.0035    0.7704      0.0002     -0.2254   10
0.010      2.0      1.9949     2.3260     1.9947       1.0156         0.0303    0.6826      0.0002     -0.3310    8
```

## Coefficients (train+tune refit, standardised features)

```
               feature  coefficient  standard_error
  jt_jockey_win365_rel       0.1608          0.0210
   hf_avg3_fin_pct_rel      -0.0206          0.0197
           rc_draw_pct      -0.0213          0.0234
          rc_draw_x_hv      -0.0968          0.0363
         rc_weight_rel      -0.0266          0.0197
         hf_days_since       0.0036          0.0225
 tr_fin_pct_slope3_rel      -0.0573          0.0206
jtp_pair_effect365_rel       0.0357          0.0199
        hf_first_start      -0.0077          0.0209
             hf_win5_s       0.0906          0.0202
```

SEs are model-based and approximate whenever a penalty or extra stage is active.
