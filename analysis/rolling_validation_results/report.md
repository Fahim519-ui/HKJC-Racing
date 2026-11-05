# Rolling validation (Kaggle era, chronological)

Each fold trains on races strictly before its window; the LightGBM tree count is chosen by an inner split inside training only; the blend weights are fitted on training and scored once on the next period. Gain is against the market-only refit (favourite-longshot corrected), with a paired race-level bootstrap over pooled fold races. Kaggle only; HKJC is never touched here.

## Per-fold log loss

```
                  fold  races  trees  beta  market  refit     cl   lgbm  mkt_cl  mkt_lgbm  mkt_both
2019-02-01..2019-08-01    426    109 1.060  1.8814 1.8779 2.0957 2.1473  1.8774    1.8806    1.8862
2019-08-01..2020-02-01    379    136 1.075  1.9721 1.9710 2.1884 2.1812  1.9719    1.9711    1.9758
2020-02-01..2020-08-01    377    118 1.072  2.0179 2.0204 2.1739 2.1818  2.0198    2.0192    2.0213
```

## Pooled, all features

```
   model  pooled_log_loss  gain_vs_refit   ci_lo   ci_hi  beats_refit
  market           1.9540        -0.0008 -0.0040  0.0023        False
   refit           1.9532         0.0000  0.0000  0.0000        False
      cl           2.1504        -0.1972 -0.2296 -0.1621        False
    lgbm           2.1692        -0.2160 -0.2498 -0.1809        False
  mkt_cl           1.9531         0.0001 -0.0015  0.0017        False
mkt_lgbm           1.9538        -0.0006 -0.0031  0.0018        False
mkt_both           1.9580        -0.0048 -0.0088 -0.0010        False
```

## Pooled, trainer features removed

```
   model  pooled_log_loss  gain_vs_refit   ci_lo   ci_hi  beats_refit
  market           1.9540        -0.0008 -0.0040  0.0023        False
   refit           1.9532         0.0000  0.0000  0.0000        False
      cl           2.1493        -0.1961 -0.2288 -0.1602        False
    lgbm           2.1633        -0.2101 -0.2440 -0.1736        False
  mkt_cl           1.9535        -0.0003 -0.0023  0.0018        False
mkt_lgbm           1.9552        -0.0020 -0.0054  0.0016        False
mkt_both           1.9593        -0.0061 -0.0106 -0.0018        False
```

Read: a model 'beats' the refit only if its whole 95% CI is above 0. In-era gains here do not prove an HKJC edge; the one-shot HKJC test is scored separately in Week 5.
