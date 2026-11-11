# Longshot-bias calibration evidence

> **7 Nov:** the HKJC test (`h_test`, Apr-Oct 2026) rows and the sentences that quoted them
> were removed, here and in `market_calibration_deciles.csv` / `market_calibration_segments.csv`.
> The test window is scored once, in Week 5. Those numbers were committed on 4 Nov and seen
> early; see `docs/report_data.md` (Limitations). The HKJC Stage 2 favourites row was missing
> before and is added from the same decile file. `model_logic/market_diagnostics.py` still
> writes h_test rows when re-run.

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
| HKJC Stage 2 | Favourites (deciles 9-10) | 1,037 | 0.219502 | 0.230473 | +0.010970 | 1.050 |

## Interpretation

Longshots have lower observed win rates than their market probabilities in all
three non-test splits (observed/predicted 0.889, 0.657 and 0.717). In HKJC
Stage 2 their observed win rate was 0.71% against a market probability of
1.00%. Favourites won slightly more often than the market implied in Kaggle
train (1.058) and HKJC Stage 2 (1.050), and slightly less in Kaggle tune
(0.978). Longshot groups hold only about 1,000-3,000 runners and about 7-26
winners each, so each ratio is noisy; the consistent direction is the evidence.

This is evidence of a stable longshot overpricing pattern, but not evidence of
a profitable strategy by itself. Any market-calibration spline must be fitted
on earlier data and evaluated once on a later period.

## Reproducibility

The source decile data are in
[`market_calibration_deciles.csv`](market_calibration_deciles.csv), generated
by [`model_logic/market_diagnostics.py`](../../model_logic/market_diagnostics.py).
