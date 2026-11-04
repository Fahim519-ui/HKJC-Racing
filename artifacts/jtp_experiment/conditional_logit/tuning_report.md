# Conditional-logit tuning report

## Status: SELECTED - standalone DOES NOT BEAT market; blend ADDS information over market

Validation window: Kaggle 2019/20 (`k_tune`), races with complete market prices.
- Market per-race log loss: **1.9949**
- Selected conditional logit: **2.1790** (gain -0.1841)
- Benter blend softmax(a*mkt + b*u): **1.9947** (gain +0.0002; a=1.036, b=-0.020; 2 in-sample parameters)
- Within-race corr(utility, market logit): 0.804

Hyper-parameters: lambda=0.0, l1_ratio=0.5, lambda_mkt=0.0; stage weights=[1.0].

Odds are used only inside the training-time orthogonality penalty; no odds or market-derived value is a model input. The Apr-Oct 2026 HKJC window is predicted, never scored. Feature selection and penalty choice both use `k_tune`, so its gains are mildly optimistic.

## Selected features

- `jt_jockey_win365_rel`
- `hf_avg3_fin_pct_rel`
- `rc_draw_pct`
- `rc_draw_x_hv`
- `rc_weight_rel`
- `hf_days_since`
- `hf_last_fin_pct_rel`
- `jtp_pair_effect365_rel`
- `hf_first_start`
- `hf_win5_s`

## Selection trace

```
 step              candidate  log_loss  improvement  accepted note                                                                                                                                                                            selected_features
    0                 (base)    2.1967          NaN      True                                                                                                jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since
    1    hf_last_fin_pct_rel    2.1874       0.0092      True                                                                           jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel
    2 jtp_pair_effect365_rel    2.1828       0.0046      True                                                   jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel, jtp_pair_effect365_rel
    3         hf_first_start    2.1805       0.0023      True                                   jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel, jtp_pair_effect365_rel, hf_first_start
    4              hf_win5_s    2.1790       0.0015      True                        jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel, jtp_pair_effect365_rel, hf_first_start, hf_win5_s
    5       rc_draw_x_sprint    2.1783       0.0008     False      jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel, jtp_pair_effect365_rel, hf_first_start, hf_win5_s, rc_draw_x_sprint
```

## Penalty grid (k_tune)

```
  lam  lam_mkt  market_nll  model_nll  blend_nll  blend_a_mkt  blend_b_model  corr_mkt  blend_gain  model_gain  nnz
0.000      0.0      1.9949     2.1790     1.9947       1.0364        -0.0197    0.8039      0.0002     -0.1841   10
0.000      0.1      1.9949     2.1806     1.9947       1.0331        -0.0170    0.7984      0.0002     -0.1857   10
0.000      0.5      1.9949     2.2188     1.9948       1.0248        -0.0050    0.7716      0.0002     -0.2239   10
0.000      2.0      1.9949     2.3242     1.9947       1.0158         0.0291    0.6815      0.0002     -0.3292   10
0.003      0.0      1.9949     2.1789     1.9947       1.0362        -0.0196    0.8037      0.0002     -0.1840   10
0.003      0.1      1.9949     2.1809     1.9947       1.0330        -0.0168    0.7982      0.0002     -0.1860   10
0.003      0.5      1.9949     2.2194     1.9948       1.0247        -0.0047    0.7715      0.0002     -0.2245   10
0.003      2.0      1.9949     2.3247     1.9947       1.0169         0.0248    0.6853      0.0002     -0.3297   10
0.010      0.0      1.9949     2.1787     1.9947       1.0358        -0.0193    0.8031      0.0002     -0.1838   10
0.010      0.1      1.9949     2.1816     1.9947       1.0326        -0.0164    0.7976      0.0002     -0.1867   10
0.010      0.5      1.9949     2.2209     1.9948       1.0243        -0.0039    0.7711      0.0002     -0.2260   10
0.010      2.0      1.9949     2.3258     1.9948       1.0192         0.0149    0.6938      0.0002     -0.3309    8
```

## Coefficients (train+tune refit, standardised features)

```
               feature  coefficient  standard_error
  jt_jockey_win365_rel       0.3262          0.0223
   hf_avg3_fin_pct_rel      -0.3140          0.0403
           rc_draw_pct      -0.1108          0.0291
          rc_draw_x_hv      -0.2186          0.0465
         rc_weight_rel       0.0069          0.0240
         hf_days_since      -0.0025          0.0279
   hf_last_fin_pct_rel      -0.2256          0.0362
jtp_pair_effect365_rel       0.0698          0.0217
        hf_first_start      -0.0731          0.0303
             hf_win5_s       0.0695          0.0238
```

SEs are model-based and approximate whenever a penalty or extra stage is active.
