# Conditional-logit tuning report

## Status: SELECTED - standalone DOES NOT BEAT market; blend ADDS information over market

Validation window: Kaggle 2019/20 (`k_tune`), races with complete market prices.
- Market per-race log loss: **1.9949**
- Selected conditional logit: **2.1804** (gain -0.1855)
- Benter blend softmax(a*mkt + b*u): **1.9946** (gain +0.0003; a=1.047, b=-0.035; 2 in-sample parameters)
- Within-race corr(utility, market logit): 0.808

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
- `jt_trainer_win365_rel`
- `hf_first_start`
- `hf_win5_s`

## Selection trace

```
 step             candidate  log_loss  improvement  accepted note                                                                                                                                                                           selected_features
    0                (base)    2.1967          NaN      True                                                                                               jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since
    1   hf_last_fin_pct_rel    2.1874       0.0092      True                                                                          jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel
    2 jt_trainer_win365_rel    2.1851       0.0023      True                                                   jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel, jt_trainer_win365_rel
    3        hf_first_start    2.1825       0.0026      True                                   jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel, jt_trainer_win365_rel, hf_first_start
    4             hf_win5_s    2.1804       0.0021      True                        jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel, jt_trainer_win365_rel, hf_first_start, hf_win5_s
    5      rc_draw_x_sprint    2.1797       0.0007     False      jt_jockey_win365_rel, hf_avg3_fin_pct_rel, rc_draw_pct, rc_draw_x_hv, rc_weight_rel, hf_days_since, hf_last_fin_pct_rel, jt_trainer_win365_rel, hf_first_start, hf_win5_s, rc_draw_x_sprint
```

## Penalty grid (k_tune)

```
  lam  lam_mkt  market_nll  model_nll  blend_nll  blend_a_mkt  blend_b_model  corr_mkt  blend_gain  model_gain  nnz
0.000      0.0      1.9949     2.1804     1.9946       1.0469        -0.0352    0.8076      0.0003     -0.1855   10
0.000      0.1      1.9949     2.1829     1.9946       1.0436        -0.0345    0.8019      0.0003     -0.1879   10
0.000      0.5      1.9949     2.2231     1.9947       1.0349        -0.0283    0.7742      0.0002     -0.2282   10
0.000      2.0      1.9949     2.3298     1.9948       1.0239        -0.0055    0.6789      0.0002     -0.3348   10
0.003      0.0      1.9949     2.1803     1.9946       1.0467        -0.0351    0.8074      0.0003     -0.1853   10
0.003      0.1      1.9949     2.1831     1.9946       1.0434        -0.0344    0.8017      0.0003     -0.1882   10
0.003      0.5      1.9949     2.2237     1.9947       1.0347        -0.0279    0.7742      0.0002     -0.2288   10
0.003      2.0      1.9949     2.3302     1.9948       1.0249        -0.0102    0.6829      0.0002     -0.3352   10
0.010      0.0      1.9949     2.1801     1.9946       1.0462        -0.0349    0.8069      0.0003     -0.1852   10
0.010      0.1      1.9949     2.1838     1.9946       1.0430        -0.0340    0.8013      0.0003     -0.1889   10
0.010      0.5      1.9949     2.2251     1.9947       1.0342        -0.0271    0.7739      0.0002     -0.2302   10
0.010      2.0      1.9949     2.3311     1.9947       1.0275        -0.0217    0.6919      0.0002     -0.3362    9
```

## Coefficients (train+tune refit, standardised features)

```
              feature  coefficient  standard_error
 jt_jockey_win365_rel       0.3472          0.0199
  hf_avg3_fin_pct_rel      -0.3036          0.0403
          rc_draw_pct      -0.1109          0.0291
         rc_draw_x_hv      -0.2178          0.0466
        rc_weight_rel       0.0050          0.0240
        hf_days_since       0.0021          0.0278
  hf_last_fin_pct_rel      -0.2290          0.0363
jt_trainer_win365_rel       0.1076          0.0214
       hf_first_start      -0.0782          0.0303
            hf_win5_s       0.0691          0.0236
```

SEs are model-based and approximate whenever a penalty or extra stage is active.
