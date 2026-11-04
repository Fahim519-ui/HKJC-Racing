# Market-offset LightGBM diagnostic

Trees are trained with `init_score = log(normalized market probability)` and a per-race softmax loss. This is diagnostic only: it uses the market inside the learner and is not a valid odds-free Stage-1 model.

## Walk-forward folds

```
validation_start validation_end  market_log_loss  residual_log_loss  best_iteration  train_races  valid_races  train_rows  valid_rows  market_gain
      2018-08-01     2019-02-01         2.015968           2.012760             249          653          381        7931        4495     0.003208
      2019-02-01     2019-08-01         1.881351           1.881331               1         1034          426       12426        5199     0.000020
      2019-08-01     2020-02-01         1.972091           1.970387             353         1460          379       17625        4669     0.001704
      2020-02-01     2020-08-01         2.017902           2.017407             179         1839          377       22294        4630     0.000496
```

## Summary

```
 folds  positive_gain_folds  weighted_market_log_loss  weighted_residual_log_loss  weighted_market_gain
     4                    4                  1.969105                    1.967785               0.00132
```
