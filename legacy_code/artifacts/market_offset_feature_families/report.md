# Market-offset LightGBM feature-family attribution

Each model uses `init_score = log(market probability)` and identical walk-forward folds. `only_*` tests standalone residual signal; `without_*` tests the full model after removing that family. Positive `gain_vs_full` means a set outperformed the full model.

## Summary

```
                            feature_set  n_features  positive_gain_folds  weighted_market_log_loss  weighted_residual_log_loss  weighted_market_gain  mean_fold_gain  gain_vs_full
       only_track_distance_interactions           3                    3                  1.969105                    1.966133              0.002972        0.002981      0.001651
                 without_jockey_trainer          13                    4                  1.969105                    1.967379              0.001726        0.001782      0.000406
                     without_horse_form           9                    3                  1.969105                    1.967556              0.001549        0.001592      0.000229
                  only_rc_weight_x_dist           1                    4                  1.969105                    1.967617              0.001488        0.001522      0.000168
    without_track_distance_interactions          14                    4                  1.969105                    1.967673              0.001432        0.001472      0.000112
                       only_draw_weight           2                    3                  1.969105                    1.967697              0.001408        0.001451      0.000087
                                   full          17                    4                  1.969105                    1.967785              0.001320        0.001357      0.000000
track_distance_without_rc_draw_x_sprint           2                    3                  1.969105                    1.968071              0.001034        0.001053     -0.000286
    track_distance_without_rc_draw_x_hv           2                    4                  1.969105                    1.968381              0.000724        0.000741     -0.000597
                    only_jockey_trainer           4                    3                  1.969105                    1.968602              0.000503        0.000515     -0.000818
                    without_draw_weight          15                    3                  1.969105                    1.968636              0.000469        0.000482     -0.000852
                      only_rc_draw_x_hv           1                    2                  1.969105                    1.968788              0.000317        0.000316     -0.001003
                  only_rc_draw_x_sprint           1                    3                  1.969105                    1.968905              0.000200        0.000206     -0.001121
                        only_horse_form           8                    3                  1.969105                    1.968926              0.000179        0.000183     -0.001142
track_distance_without_rc_weight_x_dist           2                    3                  1.969105                    1.968980              0.000125        0.000125     -0.001195
```
