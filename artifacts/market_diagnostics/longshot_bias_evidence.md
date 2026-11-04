# Longshot-bias calibration evidence

This table aggregates the two lowest and two highest market-probability
deciles from the chronological calibration results. `Observed - predicted` is
negative when the market assigns too much win probability to that group.

| Split | Group | Runners | Mean market probability | Observed win rate | Observed - predicted | Observed / predicted |
|---|---|---:|---:|---:|---:|---:|
| Kaggle train | Longshots (deciles 1-2) | 3,072 | 0.009525 | 0.008464 | -0.001061 | 0.889 |
| Kaggle train | Favourites (deciles 9-10) | 3,135 | 0.227290 | 0.240510 | +0.013221 | 1.058 |
| Kaggle tune | Longshots (deciles 1-2) | 2,123 | 0.007892 | 0.005181 | -0.002710 | 0.657 |
| Kaggle tune | Favourites (deciles 9-10) | 1,908 | 0.225578 | 0.220650 | -0.004929 | 0.978 |
| HKJC Stage 2 | Longshots (deciles 1-2) | 981 | 0.009953 | 0.007136 | -0.002817 | 0.717 |
| HKJC test | Longshots (deciles 1-2) | 982 | 0.009584 | 0.008147 | -0.001438 | 0.850 |
| HKJC test | Favourites (deciles 9-10) | 1,000 | 0.217025 | 0.219000 | +0.001975 | 1.009 |

## Interpretation

Longshots have lower observed win rates than their market probabilities in all
four chronological splits. The held-out HKJC test shows an observed/expected
ratio of **0.850** for longshots: their observed win rate was 0.81% versus a
market probability of 0.96%. Favourites were close to calibrated on the same
held-out test (ratio **1.009**).

This is evidence of a stable longshot overpricing pattern, but not evidence of
a profitable strategy by itself. Any market-calibration spline must be fitted
on earlier data and evaluated once on a later period.

## Reproducibility

The source decile data are in
[`market_calibration_deciles.csv`](market_calibration_deciles.csv), generated
by [`model_logic/market_diagnostics.py`](../../model_logic/market_diagnostics.py).
