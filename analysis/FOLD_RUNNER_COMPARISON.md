# Fahim's fold runner vs eval_utils / rolling_validation (checked 6 Nov)

Question: do Fahim's fold results (`model_logic/run_folds.py` + `aggregate_folds.py`, primary
config) agree with our evaluation code? Short answer: yes. The losses and gains are identical,
and after two fixes on our side the confidence intervals are identical too. Both pipelines say
the same thing: no blend shows a clear gain over the calibrated market.

## 1. Same predictions, our scoring code

`eval_utils` was run on Fahim's own prediction files,
`artifacts/folds/primary/{k1,k2,h1}/predictions.parquet` (`role == "valid"` rows), against his
`fold_metrics.csv`. Gain is calibrated-market loss minus blend loss (positive means the blend
is better). CI is the 95% race-day bootstrap, seed 0.

| Fold | Races | Days | Blend | Gain (both) | Our CI | His CI |
|---|---|---|---|---|---|---|
| k1 | 379 | 40 | mkt+cl | -0.00043 | -0.00292 to +0.00186 | -0.00292 to +0.00186 |
| k1 | 379 | 40 | mkt+lgbm | +0.00035 | -0.00176 to +0.00234 | -0.00176 to +0.00234 |
| k1 | 379 | 40 | mkt+cl+lgbm | -0.00031 | -0.00259 to +0.00175 | -0.00259 to +0.00175 |
| k2 | 377 | 40 | mkt+cl | +0.00071 | -0.00100 to +0.00247 | -0.00100 to +0.00247 |
| k2 | 377 | 40 | mkt+lgbm | +0.00079 | -0.00056 to +0.00213 | -0.00056 to +0.00213 |
| k2 | 377 | 40 | mkt+cl+lgbm | +0.00066 | -0.00098 to +0.00236 | -0.00098 to +0.00236 |
| h1 | 171 | 17 | mkt+cl | +0.00788 | -0.00145 to +0.01651 | -0.00145 to +0.01651 |
| h1 | 171 | 17 | mkt+lgbm | +0.00360 | -0.00430 to +0.01083 | -0.00430 to +0.01083 |
| h1 | 171 | 17 | mkt+cl+lgbm | +0.00741 | -0.00354 to +0.01730 | -0.00354 to +0.01730 |
| pooled | 927 | 97 | mkt+cl | +0.00157 | -0.00052 to +0.00362 | -0.00048 to +0.00373 |
| pooled | 927 | 97 | mkt+lgbm | +0.00113 | -0.00058 to +0.00279 | -0.00063 to +0.00286 |
| pooled | 927 | 97 | mkt+cl+lgbm | +0.00151 | -0.00070 to +0.00371 | -0.00072 to +0.00385 |

Pooled CIs differ in the 4th decimal only because the race days reach the random generator in
a different order (his pooled frame is concatenated h1, k1, k2). Feeding his day order to our
function gives his numbers exactly. That is Monte Carlo noise, not a method difference.

No interval excludes zero in any fold or pooled.

## 2. What had to change in eval_utils to match

1. **Dead heats.** We used the mean of `-log p` over the winners; he (and
   `week34_starter.py`) uses `-log` of the mean winner probability. We switched to his rule.
   It changes only dead-heat races (1 in k1 valid, 2 in h1 valid, 3 in h_s2). Side effect:
   the h_s2 market refit beta in the combined file moved from 1.0454 to 1.0537 (only
   `p_market_refit` changed, max 0.005 per runner).
2. **Bootstrap unit.** We resampled single races; he resamples race days. Day resampling is the
   right call because same-day races are correlated, so we made it the default. Effect: the
   h1 mkt+cl interval, which was just above zero with single races (+0.0002 to +0.0155), now
   includes zero (-0.0015 to +0.0165). So "h1 mkt+cl beats the market" was never a robust
   finding.

Also added: the h_test guard now blocks by date (>= 2026-04-01) as well as by `split`,
because the fold files have no `split` column. All fold files pass (latest h1 date 2026-03-29).

## 3. The two pipelines are different experiments

| | Fahim's fold runner | `rolling_validation.py` |
|---|---|---|
| Windows | k1 valid Sep 2019-Jan 2020, k2 Feb-Jun 2020, h1 Feb-Mar 2026 (HKJC) | Feb-Jul 2019, Aug 2019-Jan 2020, Feb-Jul 2020 (Kaggle only) |
| Logit features | forward selection per fold | all registry features, collinear ones pruned |
| Blend | ridge-penalised, lambda by forward month CV, fitted on s2 | unpenalised, fitted on out-of-fold inner split |
| Baseline | calibrated market (beta on s2) | market refit (beta on training window) |
| Pooled best gain | mkt+cl +0.0016 (-0.0005 to +0.0036) | mkt_cl +0.0003 (-0.0013 to +0.0020) |

Different windows and fitting choices, same conclusion: no detectable edge over the
calibrated market. Neither is a reason to expect a positive h_test result.

## 4. Things to keep in mind

- **h1 is not independent of the main model.** h1's validation window (Feb-Mar 2026) sits
  inside h_s2 (Nov 2025-Mar 2026), which is where the main pipeline fits its Stage 2 blend
  `p_joint`. So h1 is a fair fold for the fold runner itself, but it is not fresh evidence for
  `p_joint`. Only h_test is.
- **The HKJC gains come from a negative model weight.** In `stage2_results/` the Stage 2 weight
  on the logit is negative on HKJC windows (h_s2: alpha_cl = -0.34, SE 0.11; h1_s2: -0.27,
  SE 0.13) and about zero on Kaggle windows (k1_s2 +0.05, k2_s2 -0.06, both SE 0.10). A blend
  that gains by going against the model, with beta rising to about 1.2 at the same time, is
  acting as an extra favourite-longshot correction rather than adding the model's information.
  The h1 gain of +0.008 should be read that way.
- h1 has 171 races on 17 days, so its intervals are wide.

Reproduce: the comparison used `eval_utils.race_losses`, `race_days` and `paired_bootstrap(...,
days=..., seed=0)` on the files above; `tests/test_eval_and_final_scoring.py` checks the
bootstrap against `paired_bootstrap_gain` on synthetic data.
