# Conditional-logit tuning result

**Status:** selected as the current Stage-1 conditional logit. It does not beat
the market on the fixed Kaggle 2019/20 tuning window.

## Data and validation

- Input: `data/clean data/runners_merged_with_features.parquet`, the latest
  Week-3 merged feature table (38,942 modelling rows).
- Fit window: Kaggle, 1 Nov 2017-31 Jul 2019 (`k_train`).
- Feature-selection window: Kaggle, 1 Aug 2019-21 Jun 2020 (`k_tune`).
- The Apr-Oct 2026 HKJC test window was not scored.
- Per-race log loss is lower-is-better.

## Week-3 feature audit

The pushed table was used as the input dataset. Its historical horse form rates
were recomputed within `source` using only earlier starts before being eligible
for modelling. This follows the project's source-separation and leakage rules.

The following pushed columns were excluded from the Stage-1 pre-race model:

- `early_position` and `position_change` are calculated from the current
  race's `running_positions`, which are known only after the race.
- `jockey_past_win_rate` and `trainer_past_win_rate` change between rides by
  the same person on a single race day. That uses earlier same-day outcomes;
  the project rule requires earlier race days only.

The existing smoothed 365-day `jt_jockey_win365_rel` and
`jt_trainer_win365_rel` features remain valid replacements.

## Selection method

The model began with the six planned strong features and fitted a conditional
logit without an intercept. At each step, every remaining candidate was tried
on `k_tune`; the best candidate was retained only if it reduced log loss by at
least 0.001.

| Step | Retained feature | `k_tune` log loss | Improvement |
|---:|---|---:|---:|
| 0 | Base six features | 2.1964 | — |
| 1 | `hf_last_fin_pct_rel` | 2.1871 | 0.0093 |
| 2 | `hf_first_start` | 2.1848 | 0.0023 |
| 3 | `jt_trainer_win365_rel` | 2.1822 | 0.0025 |
| 4 | `hf_win5_s` | **2.1801** | 0.0021 |

The best remaining candidate, `rc_draw_x_sprint`, improved log loss by only
0.0007, so selection stopped. The safe recomputed Week-3 past-three-race
features were therefore not retained: no remaining candidate met the threshold.

The selected features were:

1. `jt_jockey_win365_rel`
2. `hf_avg3_fin_pct_rel`
3. `rc_draw_pct`
4. `rc_draw_x_hv`
5. `rc_weight_rel`
6. `hf_days_since`
7. `hf_last_fin_pct_rel`
8. `hf_first_start`
9. `jt_trainer_win365_rel`
10. `hf_win5_s`

The largest absolute selected-feature Spearman correlation was 0.739 between
`hf_avg3_fin_pct_rel` and `hf_last_fin_pct_rel`, below the 0.9 exclusion
threshold.

## Result

| Model | `k_tune` per-race log loss |
|---|---:|
| Market probability | **1.9947** |
| Initial six-feature logit | 2.1964 |
| Selected ten-feature logit | 2.1801 |

The selected model improves the six-feature starting point by 0.0163. The
previous 12-feature starter model scored 2.1804, so the selected model's
improvement over that directly comparable starter is 0.0003. It remains 0.1854
worse than the market. It is the strongest validated conditional-logit version
in the current feature scope, but it must not be presented as beating the
market. Its Stage-1 predictions are saved for Stage 2 and the untouched Week-5
test evaluation.

The selected `k_train` coefficients are written by
`modeling/conditional_logit_tuning.py` to
`artifacts/conditional_logit_tuning/tuning_coefficients.csv`. The largest
well-estimated effects are the expected ones: stronger relative jockey and
trainer win rates are positive, while worse recent finish percentiles and an
outside draw—especially at Happy Valley—are negative.

## Next modelling step

Keep the market benchmark as the primary baseline. A valid attempt to improve
the Stage-1 model needs additional *pre-race* signal, such as source-separated
distance/track form and a correctly lagged jockey-trainer pair rate. Do not add
current-race running positions or same-day outcomes.
