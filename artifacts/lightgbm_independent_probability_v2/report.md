# Odds-free listwise LightGBM

Stage 1 optimises per-race softmax likelihood on Kaggle data and never uses odds or market-derived columns. `k_tune` selects only the round count (397 early-stopped, 627 on train+tune, 5-seed bag).
Stage-2 ridge strength is selected by leave-one-month-out CV within `h_s2`; `h_test` is scored once per variant.

## Per-race log loss

```
window                     model  per_race_log_loss  market_gain
k_tune                market_raw           1.994693     0.000000
k_tune               market_norm           1.994693     0.000000
k_tune                  lightgbm           2.176784    -0.182091
  h_s2                    market           2.038380     0.000000
h_test                    market           2.056461     0.000000
  h_s2                  lightgbm           2.317078    -0.278697
h_test                  lightgbm           2.267514    -0.211053
  h_s2         conditional_logit           2.315355    -0.276975
h_test         conditional_logit           2.261698    -0.205237
  h_s2       mkt+lgbm (month-CV)           2.037507     0.000873
h_test    mkt+lgbm (fit on h_s2)           2.058333    -0.001872
  h_s2         mkt+cl (month-CV)           2.031422     0.006958
h_test      mkt+cl (fit on h_s2)           2.062315    -0.005854
  h_s2    mkt+lgbm+cl (month-CV)           2.030295     0.008085
h_test mkt+lgbm+cl (fit on h_s2)           2.064174    -0.007713
```

## Blend coefficients

```
    variant  lambda  p_market    p_lgbm      p_cl
   mkt+lgbm   0.030  1.132617 -0.159149       NaN
     mkt+cl   0.010  1.214363       NaN -0.306745
mkt+lgbm+cl   0.003  1.225289  0.207863 -0.544160
```
