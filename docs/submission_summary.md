# Submission Summary

## Workload

### Data sourcing and access

The project combines 27,008 Kaggle runner records from the 2017/18--2019/20
Hong Kong seasons with 12,202 HKJC website records scraped for 2025/26 and
2026/27 through 11 October 2026. There is no freely available source for the
2020--2025 gap, and the repository does not redistribute a downloader or
scraper. The source schema, access limitations, and local inputs are documented
in `README.md` and `data/clean data/cleaning_report.md`.

### Data cleaning and preparation

The merged data has 39,210 runner rows across 3,188 races. Preparation
standardises finish status, excludes nine invalid races, preserves valid
starters, and documents ten suspected Kaggle timing inconsistencies. HKJC
profile fields are missing non-randomly for horses that later left Hong Kong
racing; the project neither uses missingness as a feature nor drops those rows.

Feature engineering is strictly pre-race: histories use earlier race days only,
are source-isolated, and are tested with same-day, future-cutoff,
cross-source, and market-perturbation checks. The current raw rebuild passed
all eight reproducibility and leakage checks.

### Model development and iteration

The project evaluated race-normalised conditional logit and listwise LightGBM
models, then market/model blends. It also compared uniform, draw-only,
historical conditional-logit, raw-market, calibrated-market, single-model, and
blend rungs on identical races. Track/distance and trainer-rate ablations were
run alongside the primary configuration. Earlier v3.0, rolling, market-offset,
and in-sample Stage-2 outputs are quarantined and are not used as evidence.

## Reasoning and evidence

### Baselines and accepted fold result

The accepted configuration is the `no_track_distance` market-plus-conditional-
logit blend. On the pooled three-fold validation data (927 races), it achieved
log loss **1.988108**, versus **1.989114** for raw market and **1.989531** for
the calibrated market: gains of **+0.001006** and **+0.001423**, respectively.
Its paired race-day bootstrap interval against calibrated market is
**[-0.000693, +0.003607]**, so the small point-estimate edge is not
statistically clear.

The pooled result masks inconsistent eligible-fold results: k1 has a
calibrated-market gain of **-0.000355**, while k2 has **+0.000774**. The HKJC
h1 result is **+0.006792**, but it is watch-only because its inner-tune period
has only 140 races. The selection therefore supports, at most, a small
exploratory improvement rather than a reliable market edge.

The current freeze policy permits a positive point estimate from any eligible
ladder candidate. This policy and the choice of the best pooled configuration
were made after inspecting the v3.1 fold results. They are operational choices
for this project, not pre-registered confirmatory evidence.

### Retrospective holdout result

The one-shot `h_test` score does not support a market edge. The accepted
configuration's joint model has log loss **2.0593**, compared with **2.0565**
for raw market and **2.0566** for calibrated market; the gain against calibrated
market is **-0.0027** (95% CI **[-0.0102, +0.0051]**). The test races predate
the freeze and were previously exposed during exploration, so this is a
retrospective holdout diagnostic, not prospective validation.

### What worked and what did not

The engineering and evaluation safeguards worked: reproducible feature
construction, source-isolated history, chronological folds, race-day clustered
intervals, a baseline ladder, provenance gates, and quarantine rules make the
result auditable. The modelling claim did not generalise convincingly: the
small pooled fold advantage is uncertain and the retrospective holdout is
negative. The defensible conclusion is that this project built and tested a
rigorous probability-forecasting pipeline, but it has not established a
repeatable improvement over market odds or profitable wagering after takeout.

## Evidence locations

- Fold evidence: regenerate `artifacts/folds/` with
  `python -m model_logic.run_ablation --folds model_logic/validation/folds.csv --out artifacts/folds`;
  current local outputs include `ablation_metrics.csv` and each configuration's
  `fold_metrics.csv` and `ladder.md`.
- Frozen prediction provenance:
  `artifacts/combined/combined_predictions.provenance.json`.
- One-shot retrospective score:
  `analysis/final_scoring_results/h_test/report.md`.
- Freeze record:
  `analysis/final_scoring_results/freeze_record.json`.
