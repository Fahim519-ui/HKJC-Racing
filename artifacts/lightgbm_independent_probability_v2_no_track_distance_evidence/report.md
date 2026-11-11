# Odds-free listwise LightGBM

Stage 1 optimises per-race softmax likelihood on Kaggle data and never uses odds or market-derived columns. `k_tune` selects only the round count (343 early-stopped, 541 on train+tune, 5-seed bag).
Stage-2 ridge strength is selected by leave-one-month-out CV within `h_s2`; `h_test` is scored once per variant.

## Per-race log loss

```
window                     model  per_race_log_loss  market_gain
k_tune                market_raw           1.994693     0.000000
k_tune               market_norm           1.994693     0.000000
k_tune                  lightgbm           2.178465    -0.183771
  h_s2                    market           2.038380     0.000000
h_test                    market           2.056461     0.000000
  h_s2                  lightgbm           2.315038    -0.276657
h_test                  lightgbm           2.258272    -0.201811
  h_s2         conditional_logit           2.420804    -0.382424
h_test         conditional_logit           2.405263    -0.348801
  h_s2       mkt+lgbm (month-CV)           2.036340     0.002040
h_test    mkt+lgbm (fit on h_s2)           2.059574    -0.003113
  h_s2         mkt+cl (month-CV)           2.040282    -0.001902
h_test      mkt+cl (fit on h_s2)           2.056398     0.000063
  h_s2    mkt+lgbm+cl (month-CV)           2.036628     0.001753
h_test mkt+lgbm+cl (fit on h_s2)           2.059333    -0.002872
```

## Blend coefficients

```
    variant  lambda  p_market    p_lgbm      p_cl
   mkt+lgbm    0.03  1.135611 -0.173067       NaN
     mkt+cl    0.10  1.038412       NaN -0.030637
mkt+lgbm+cl    0.03  1.134434 -0.183551  0.052092
```
