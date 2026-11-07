# Odds-free LightGBM probability model

> **7 Nov:** the `h_test` (HKJC Apr-Oct 2026) rows were removed from this report and from `metrics.csv`. The test window is scored once, in Week 5, by `analysis/final_scoring.py`. These numbers had been committed on 4 Nov and were seen early; see `docs/report_data.md` (Limitations). The scripts that wrote them still score h_test when re-run, so do not re-run them before the final scoring without removing that.

Stage 1 uses only the listed physical and strictly lagged historical features. It excludes odds, implied probability, normalized market probability, and every other market-derived column. The binary classifier output is normalized within race.

- LightGBM early-stopping trees: 205
- `k_tune` selects only the tree count; it is not an unbiased model comparison.
- The joint conditional-logit blend is fit on `h_s2` and evaluated on the later `h_test` window.

## Per-race log loss

```
window             model  per_race_log_loss  market_gain
k_tune            market           1.994693     0.000000
k_tune          lightgbm           2.183708    -0.189015
  h_s2            market           2.038380     0.000000
  h_s2          lightgbm           2.309663    -0.271283
  h_s2 conditional_logit           2.315355    -0.276975
  h_s2   joint_in_sample           2.024395     0.013985
```

## Joint blend coefficients

```
p_market    1.234302
p_lgbm      0.069294
p_cl       -0.404479
```
