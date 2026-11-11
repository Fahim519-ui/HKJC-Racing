# Rolling validation (Kaggle era, chronological)

Each fold trains on races strictly before its window; the LightGBM tree count is chosen by an inner split inside training only; the blend weights are fitted on training and scored once on the next period. Gain is against the market-only refit (favourite-longshot corrected), with a paired bootstrap that resamples whole race days over the pooled fold races (dead heats scored as -log of the mean winner probability). Kaggle only; HKJC is never touched here.

## Per-fold log loss

```
                  fold  races  trees  beta  cl_feats  market  refit     cl   lgbm  mkt_cl  mkt_lgbm  mkt_both
2019-02-01..2019-08-01    426    138 1.060        22  1.8794 1.8758 2.0935 2.1252  1.8752    1.8775    1.8787
2019-08-01..2020-02-01    379    164 1.077        23  1.9716 1.9704 2.1857 2.1836  1.9710    1.9704    1.9710
2020-02-01..2020-08-01    377    140 1.073        24  2.0179 2.0205 2.1653 2.1646  2.0195    2.0193    2.0197
```

## Pooled, all features

```
   model  pooled_log_loss  gain_vs_refit   ci_lo   ci_hi  beats_refit
  market           1.9531        -0.0009 -0.0044  0.0028        False
   refit           1.9523         0.0000  0.0000  0.0000        False
      cl           2.1460        -0.1937 -0.2288 -0.1579        False
    lgbm           2.1565        -0.2042 -0.2413 -0.1681        False
  mkt_cl           1.9519         0.0003 -0.0013  0.0020        False
mkt_lgbm           1.9525        -0.0003 -0.0028  0.0022        False
mkt_both           1.9533        -0.0010 -0.0040  0.0021        False
```

## Pooled, trainer features removed

```
   model  pooled_log_loss  gain_vs_refit   ci_lo   ci_hi  beats_refit
  market           1.9531        -0.0009 -0.0044  0.0028        False
   refit           1.9523         0.0000  0.0000  0.0000        False
      cl           2.1444        -0.1922 -0.2275 -0.1558        False
    lgbm           2.1578        -0.2055 -0.2430 -0.1678        False
  mkt_cl           1.9523        -0.0000 -0.0025  0.0024        False
mkt_lgbm           1.9535        -0.0013 -0.0048  0.0022        False
mkt_both           1.9553        -0.0030 -0.0074  0.0013        False
```

## Logit features

Candidates: the registry's conditional-logit list (32 features, v3.1). LightGBM gets all of them plus context columns. For the unpenalised logit, each fold drops (on its training window only) a feature with |Spearman| > 0.9 against an earlier kept one, or one that is an exact linear combination of kept ones; `cl_feats` in the fold table is the number kept. Dropped per fold (all features run):

```
                  fold                                                                                                                                                                                                                                                                               dropped
2019-02-01..2019-08-01 jtp_pair_effect_last_50_rel, td_distance_band_win_rel, td_distance_band_top3_rel, td_distance_band_fin_pct_rel, td_track_distance_win_rel, td_track_distance_top3_rel, td_track_distance_fin_pct_rel, tr_days_since_bucket, tr_class_change_direction, tr_last_minus_avg3_fin_pct_rel
2019-08-01..2020-02-01                              td_distance_band_win_rel, td_distance_band_top3_rel, td_distance_band_fin_pct_rel, td_track_distance_win_rel, td_track_distance_top3_rel, td_track_distance_fin_pct_rel, tr_days_since_bucket, tr_class_change_direction, tr_last_minus_avg3_fin_pct_rel
2020-02-01..2020-08-01                                                         td_distance_band_win_rel, td_distance_band_top3_rel, td_distance_band_fin_pct_rel, td_track_distance_win_rel, td_track_distance_top3_rel, td_track_distance_fin_pct_rel, tr_days_since_bucket, tr_last_minus_avg3_fin_pct_rel
```

Read: a model 'beats' the refit only if its whole 95% CI is above 0. In-era gains here do not prove an HKJC edge; the one-shot HKJC test is scored separately in Week 5.
