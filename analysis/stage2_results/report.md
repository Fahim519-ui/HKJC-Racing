# Stage 2 alpha/beta table

Blend: score = beta * log p_market + alpha * log p_model, softmax within race. Plain (unpenalised) conditional logit, so each weight has a standard error. All fits are in-sample on their Stage-2 window; out-of-sample evidence is in the fold validation. SEs treat races as independent (same-day races are correlated, so true SEs are larger). `penalised_coef` is the ridge-penalised weight the team's scripts used (h_s2: lightgbm_independent_probability_v2/joint_coefficients.csv; folds: blend.json; for the folds' 'market only' row it is the calibrated-market beta).

Windows: h_s2 = HKJC 2025-11..2026-03 Stage-2 rows of the combined file; k1_s2, k2_s2, h1_s2 = s2 rows of the fold runner (see model_logic/validation/folds.csv). h1_s2 lies inside h_s2. h_test is not read.

## Coefficients

```
window        spec          term   coef    se      z        p  penalised_coef
  h_s2 market only beta (market)  1.041 0.059 17.619 1.77e-69             NaN
  h_s2      mkt+cl beta (market)  1.237 0.086 14.399 5.23e-47           1.214
  h_s2      mkt+cl      alpha_cl -0.344 0.106 -3.235  0.00122          -0.307
  h_s2    mkt+lgbm beta (market)  1.169 0.083 14.124 2.72e-45           1.133
  h_s2    mkt+lgbm    alpha_lgbm -0.205 0.091 -2.253   0.0243          -0.159
  h_s2 mkt+cl+lgbm beta (market)  1.229 0.086 14.241 5.07e-46           1.225
  h_s2 mkt+cl+lgbm      alpha_cl -0.657 0.237 -2.771  0.00559          -0.544
  h_s2 mkt+cl+lgbm    alpha_lgbm  0.301 0.204  1.476     0.14           0.208
 k1_s2 market only beta (market)  1.103 0.057 19.260 1.15e-82           1.106
 k1_s2      mkt+cl beta (market)  1.072 0.086 12.423 1.97e-35           1.062
 k1_s2      mkt+cl      alpha_cl  0.050 0.107  0.466    0.641           0.044
 k1_s2    mkt+lgbm beta (market)  1.116 0.082 13.637  2.4e-42           1.078
 k1_s2    mkt+lgbm    alpha_lgbm -0.021 0.092 -0.233    0.816           0.010
 k1_s2 mkt+cl+lgbm beta (market)  1.080 0.087 12.477 9.94e-36           1.066
 k1_s2 mkt+cl+lgbm      alpha_cl  0.247 0.198  1.247    0.212           0.052
 k1_s2 mkt+cl+lgbm    alpha_lgbm -0.201 0.171 -1.175     0.24          -0.015
 k2_s2 market only beta (market)  1.059 0.062 16.952 1.85e-64           1.060
 k2_s2      mkt+cl beta (market)  1.102 0.092 11.965  5.4e-33           1.056
 k2_s2      mkt+cl      alpha_cl -0.063 0.100 -0.631    0.528          -0.020
 k2_s2    mkt+lgbm beta (market)  1.085 0.092 11.750  7.1e-32           1.051
 k2_s2    mkt+lgbm    alpha_lgbm -0.038 0.102 -0.375    0.708          -0.009
 k2_s2 mkt+cl+lgbm beta (market)  1.095 0.094 11.648 2.36e-31           1.056
 k2_s2 mkt+cl+lgbm      alpha_cl -0.120 0.196 -0.613     0.54          -0.022
 k2_s2 mkt+cl+lgbm    alpha_lgbm  0.068 0.200  0.340    0.734           0.003
 h1_s2 market only beta (market)  1.000 0.076 13.233 5.64e-40           1.005
 h1_s2      mkt+cl beta (market)  1.157 0.110 10.562 4.45e-26           1.062
 h1_s2      mkt+cl      alpha_cl -0.270 0.133 -2.028   0.0426          -0.129
 h1_s2    mkt+lgbm beta (market)  1.131 0.106 10.621 2.37e-26           1.054
 h1_s2    mkt+lgbm    alpha_lgbm -0.232 0.130 -1.784   0.0745          -0.116
 h1_s2 mkt+cl+lgbm beta (market)  1.156 0.110 10.523 6.74e-26           1.074
 h1_s2 mkt+cl+lgbm      alpha_cl -0.312 0.316 -0.987    0.324          -0.093
 h1_s2 mkt+cl+lgbm    alpha_lgbm  0.046 0.309  0.147    0.883          -0.066
```

## Fit and likelihood-ratio test against market only

```
window        spec  races  runners  loglik  in_sample_log_loss  gain_vs_market_only  lr_stat  lr_df    lr_p
  h_s2 market only    423     5287 -868.76             2.03747              0.00000      NaN    NaN        
  h_s2      mkt+cl    423     5287 -863.48             2.02450              0.01297    10.56    1.0 0.00115
  h_s2    mkt+lgbm    423     5287 -866.22             2.03116              0.00631     5.07    1.0  0.0243
  h_s2 mkt+cl+lgbm    423     5287 -862.40             2.02195              0.01551    12.71    2.0 0.00174
 k1_s2 market only    426     5199 -802.07             1.87488              0.00000      NaN    NaN        
 k1_s2      mkt+cl    426     5199 -801.96             1.87462              0.00026     0.22    1.0   0.641
 k1_s2    mkt+lgbm    426     5199 -802.04             1.87484              0.00004     0.05    1.0   0.816
 k1_s2 mkt+cl+lgbm    426     5199 -801.28             1.87321              0.00167     1.58    2.0   0.454
 k2_s2 market only    379     4669 -748.40             1.97033              0.00000      NaN    NaN        
 k2_s2      mkt+cl    379     4669 -748.20             1.96981              0.00052     0.40    1.0   0.528
 k2_s2    mkt+lgbm    379     4669 -748.33             1.97009              0.00024     0.14    1.0   0.708
 k2_s2 mkt+cl+lgbm    379     4669 -748.14             1.96977              0.00056     0.51    2.0   0.775
 h1_s2 market only    252     3175 -528.94             2.08855              0.00000      NaN    NaN        
 h1_s2      mkt+cl    252     3175 -526.87             2.08001              0.00854     4.13    1.0  0.0422
 h1_s2    mkt+lgbm    252     3175 -527.35             2.08180              0.00675     3.18    1.0  0.0747
 h1_s2 mkt+cl+lgbm    252     3175 -526.86             2.08000              0.00855     4.15    2.0   0.126
```

Read: alpha with |z| < 2 is not distinguishable from 0, i.e. the blend does not lean on the model beyond the odds. A negative alpha means the model, given the odds, points the wrong way on that window. beta > 1 is the favourite-longshot correction (favourites win more often than the odds imply).
