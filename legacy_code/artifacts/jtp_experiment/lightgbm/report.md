# Odds-free listwise LightGBM

> **7 Nov:** the `h_test` (HKJC Apr-Oct 2026) rows were removed from this report and from `metrics.csv`. The test window is scored once, in Week 5, by `analysis/final_scoring.py`. These numbers had been committed on 4 Nov and were seen early; see `docs/report_data.md` (Limitations). The scripts that wrote them still score h_test when re-run, so do not re-run them before the final scoring without removing that.

Stage 1 optimises per-race softmax likelihood on Kaggle data and never uses odds or market-derived columns. `k_tune` selects only the round count (359 early-stopped, 567 on train+tune, 5-seed bag).
Stage-2 ridge strength is selected by leave-one-month-out CV within `h_s2`; `h_test` is scored once per variant.

## Per-race log loss

```
window                     model  per_race_log_loss  market_gain
k_tune                market_raw           1.994693     0.000000
k_tune               market_norm           1.994693     0.000000
k_tune                  lightgbm           2.178752    -0.184059
  h_s2                    market           2.038380     0.000000
  h_s2                  lightgbm           2.314891    -0.276511
  h_s2         conditional_logit           2.310224    -0.271843
  h_s2       mkt+lgbm (month-CV)           2.036905     0.001475
  h_s2         mkt+cl (month-CV)           2.033855     0.004525
  h_s2    mkt+lgbm+cl (month-CV)           2.035026     0.003354
```

## Blend coefficients

```
    variant  lambda  p_market    p_lgbm      p_cl
   mkt+lgbm    0.03  1.136343 -0.171364       NaN
     mkt+cl    0.03  1.158814       NaN -0.225062
mkt+lgbm+cl    0.03  1.165502 -0.045118 -0.189919
```
