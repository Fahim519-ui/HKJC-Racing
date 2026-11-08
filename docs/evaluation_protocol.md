# Evaluation Protocol

This document is the decision protocol for chronological validation, ablations,
freezing, and final scoring. It supplements `docs/DATA_RULES.md`; if an older
artifact or report conflicts with this protocol, this protocol governs model
selection and freeze decisions.

## Scope and provenance

Forward-fold evaluation uses `model_logic/validation/folds.csv`, executed with
`python -m model_logic.run_ablation` and summarized by
`model_logic.aggregate_folds`. A result is admissible only when its data hash,
registry version and hash, config hash, and git commit match the current clean,
pushed run.

## Primary comparison

The sole primary comparison is:

| Model | Baseline | Metric |
|---|---|---|
| `primary` config, `mkt+cl+lgbm` variant | `market_calibrated` | Per-race log loss |

Gain is calibrated-market loss minus primary-model loss. Positive gain is
better.

## Repeated forward-fold evidence

The repeat rule is fixed before a new forward run:

- A fold is eligible only when it has at least 150 inner-tune races.
- Every eligible fold must have positive gain versus calibrated market.
- Every eligible fold's two-sided 95% paired bootstrap interval for that gain
  must lie entirely above zero.
- Bootstrap resampling is clustered by race day.

This is an all-eligible-fold rule: `k = K`. With the current fold table,
`k1` and `k2` are eligible, so the rule is 2 of 2. `h1` currently has 140
inner-tune races and is watch-only: it is reported, but does not contribute to
freeze eligibility.

## Baseline ladder

Every current-registry forward evaluation must report this ordered ladder:

1. Uniform.
2. Draw-only.
3. Original conditional logit.
4. Raw market.
5. Calibrated market.
6. Conditional-logit-only.
7. LightGBM-only.
8. Market plus conditional logit.
9. Market plus LightGBM.
10. Market plus both models.

Rows unavailable because their implementation or inputs are absent are
incomplete evidence, not negative results. The primary comparison remains
market plus both models versus calibrated market; all other comparisons are
exploratory.

## h_test restriction

`h_test` must not be used for feature selection, feature exclusion, model
selection, hyperparameter selection, early stopping, tree-round selection,
blend fitting, calibration fitting, threshold selection, or freeze decisions.
It may be scored once only after a valid freeze. A prospective score may contain
only races dated after the recorded freeze timestamp.

## Freeze criteria

A final configuration may be frozen only when all of the following hold:

1. Core, leakage, and required evidence checks pass.
2. All required current-registry folds and baseline-ladder outputs are present.
3. The primary repeat and interval rules pass.
4. The run came from a clean, pushed commit with matching data and registry
   provenance.
5. The frozen configuration records the selected conditional-logit features and
   penalty; LightGBM features, parameters, rounds, and seeds; blend settings;
   registry hash; data hash; config hash; git commit; and freeze timestamp.

Before one-shot scoring, the prediction file's SHA-256 must be stored with the
freeze record. Any mismatch, missing provenance, or pre-freeze race in a
prospective score blocks final scoring.
