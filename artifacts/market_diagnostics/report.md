# Market diagnostics

## Baseline probability audit

```
 races  rows  missing_market_probabilities  nonpositive_market_probabilities  mean_race_probability_sum  min_race_probability_sum  max_race_probability_sum  max_absolute_sum_error
  3179 38942                             0                                 0                        1.0                       1.0                       1.0            2.220446e-16
```

## Market-offset feature-group tests

Each group is fitted on `k_train` with `log(market probability)` as a fixed conditional-logit offset. The likelihood-ratio test is in-sample; `k_tune_gain` is the out-of-sample per-race log-loss improvement over the market.

```
                      group  n_features  train_lr_statistic  train_lr_p_value  k_tune_market_log_loss  k_tune_offset_log_loss  k_tune_gain
                 horse_form           8            7.174793          0.517897                1.994936                1.995618    -0.000682
             jockey_trainer           4            5.262232          0.261433                1.994936                1.997310    -0.002374
                draw_weight           2            0.112819          0.945152                1.994936                1.994298     0.000639
track_distance_interactions           3            1.736710          0.628803                1.994936                1.993131     0.001805
```

## Calibration

See `market_calibration_deciles.csv` and `market_calibration_segments.csv`. Segment results are nested by probability decile because whole-race segment averages are mechanically calibrated after race-level normalization. Positive calibration error means the market underestimates that group.
