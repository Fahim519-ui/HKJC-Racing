# Walk-forward residual-model evidence

This diagnostic tests whether the current pre-race feature set contains
nonlinear information beyond the market. LightGBM was initialized with
`log(normalized market probability)` and trained with a per-race softmax loss,
so its trees learn only residual adjustments to the market.

| Validation period | Market log loss | Market-offset LightGBM log loss | Gain |
|---|---:|---:|---:|
| Aug 2018-Jan 2019 | 2.015968 | 2.012760 | +0.003208 |
| Feb 2019-Jul 2019 | 1.881351 | 1.881331 | +0.000020 |
| Aug 2019-Jan 2020 | 1.972091 | 1.970387 | +0.001704 |
| Feb 2020-Jul 2020 | 2.017902 | 2.017407 | +0.000496 |
| **Race-weighted total** | **1.969105** | **1.967785** | **+0.001320** |

The residual model improved log loss in all four chronological walk-forward
folds. The edge is small, but its repeated direction indicates that the
existing feature set has some nonlinear information not fully reflected in
market odds.

## Important limitation

This is **not** a deployable Stage-1 model result: market probabilities are
used inside the learner as its initialization. It is evidence for further
feature and interaction work, especially track/distance interactions, not a
claim that the standalone model beats the market.

## Reproducibility

The fold metrics are in [`fold_metrics.csv`](fold_metrics.csv), produced by
[`model_logic/market_offset_lightgbm.py`](../../model_logic/market_offset_lightgbm.py).
