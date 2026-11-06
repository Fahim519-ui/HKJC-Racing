# Odds-free listwise LightGBM

Stage 1 optimises per-race softmax likelihood on Kaggle data and never uses odds or market-derived columns. `k_tune` selects only the round count (359 early-stopped, 567 on train+tune, 5-seed bag).
Stage-2 ridge strength is selected by leave-one-month-out CV within `h_s2`; `h_test` is scored once per variant.

## Per-race log loss

```
window                     model  per_race_log_loss  market_gain
k_tune                market_raw           1.994693     0.000000
k_tune               market_norm           1.994693     0.000000
k_tune                  lightgbm           2.178752    -0.184059
  h_s2                    market           2.038380     0.000000
h_test                    market           2.056461     0.000000
  h_s2                  lightgbm           2.314891    -0.276511
h_test                  lightgbm           2.259204    -0.202743
  h_s2         conditional_logit           2.315355    -0.276975
h_test         conditional_logit           2.261698    -0.205237
  h_s2       mkt+lgbm (month-CV)           2.036905     0.001475
h_test    mkt+lgbm (fit on h_s2)           2.059431    -0.002970
  h_s2         mkt+cl (month-CV)           2.031422     0.006958
h_test      mkt+cl (fit on h_s2)           2.062315    -0.005854
  h_s2    mkt+lgbm+cl (month-CV)           2.030476     0.007904
h_test mkt+lgbm+cl (fit on h_s2)           2.062914    -0.006453
```

## Blend coefficients

```
    variant  lambda  p_market    p_lgbm      p_cl
   mkt+lgbm   0.030  1.136343 -0.171364       NaN
     mkt+cl   0.010  1.214363       NaN -0.306745
mkt+lgbm+cl   0.003  1.226748  0.181746 -0.511721
```
