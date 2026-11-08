# HKJC Racing

This repository builds race-level win-probability models for Hong Kong horse
racing and evaluates whether combining them with market odds improves
probability forecasts. The main question is: **on chronologically later races,
does a market-plus-model blend reduce per-race log loss compared with a
calibrated market baseline?

## Contents

- [Project overview](#project-overview)
- [Project strengths](#project-strengths)
- [Repository structure](#repository-structure)
- [Data](#data)
- [Setup](#setup)
- [Quickstart](#quickstart)
- [Reproducing the evidence](#reproducing-the-evidence)
- [Feature registry](#feature-registry)
- [Leakage and validation design](#leakage-and-validation-design)
- [Evaluation protocol](#evaluation-protocol)
- [Findings and limitations](#findings-and-limitations)
- [Freeze and final scoring](#freeze-and-final-scoring)
- [Testing](#testing)
- [Reproducibility notes](#reproducibility-notes)
- [Acknowledgements, data terms, and licence](#acknowledgements-data-terms-and-licence)

## Project overview

The conditional-logit model represents each race as a choice set. It assigns each runner a
linear utility from pre-race features, then applies a softmax within the race so that runner
probabilities sum to one. Feature selection uses an inner chronological split, and race loss
handles dead heats by scoring the mean predicted probability of the winners.

The LightGBM model is a listwise model with a custom race-level softmax objective. It uses
registry-derived features, monotone constraints where specified, an inner date split to choose
the number of boosting rounds, and a small seed bag in fold evaluation. The default fold
runner uses seeds `17` and `118`.

Stage 1 is odds-free: market columns are excluded from conditional-logit and LightGBM feature
construction. Stage 2 may combine Stage 1 probabilities with normalized market probabilities;
its blend settings are fitted on the fold's `s2` period and applied to the later validation
period. The [baseline ladder](#baseline-ladder) separates these comparisons.

```text
raw runner table
      |
      v
feature builder <--- versioned feature registry
      |
      v
chronological folds (s1 -> s2 -> valid)
      |
      +--> Stage 1: conditional logit (odds-free)
      +--> Stage 1: listwise LightGBM (odds-free)
      |
      v
Stage 2: market / calibrated market / blends
      |
      v
fold metrics, baseline ladder, paired race-day intervals
      |
      v
freeze record and one-shot final scoring (separate workflow)
```

## Project strengths

The project emphasizes safeguards that are often absent from small forecasting exercises:

- **Race-aware probabilities:** both Stage 1 models predict a normalized distribution over the
  runners in each race rather than unrelated binary runner labels.
- **Temporal discipline:** histories are source-separated and strictly prior-day, while
  chronological `s1`/`s2`/validation folds separate fitting from scoring.
- **Leakage-focused validation:** 26 dedicated perturbation tests cover same-day outcomes,
  cutoff/future invariance, cross-source contamination, market exclusion, invalid rows, and
  deliberately leaky negative controls. The 62-test core suite adds fold, provenance,
  baseline, and ladder contracts.
- **Transparent comparisons:** the baseline ladder starts with uniform and draw-only
  probabilities, progresses through a frozen historical conditional logit and calibrated market,
  then evaluates the modern models and blends on identical races.
- **Auditability:** a versioned registry records feature inputs, windows, shrinkage, monotonic
  constraints, and leakage notes; fold artifacts record data, registry, configuration, package,
  and commit metadata.

The final market-edge question remains open rather than assumed. The repository is therefore
well suited to demonstrating a disciplined framework for testing incremental predictive value,
with explicit criteria for when a conclusion can be made.

## Repository structure

```text
.
├── analysis/                 Evaluation, provenance, stability, and final-scoring scripts
├── artifacts/                Saved model outputs; ignored for new files, with legacy tracked files
├── data/
│   ├── clean data/           Tracked processed runner tables, dictionary, and cleaning report
│   ├── Horse Racing Results_2017-2020.numbers
│   └── hkjc_results_2025-07-26_to_2026-10-11.csv
├── docs/                     Data rules, protocol, planning, and feature notes
├── leakage test/             Separate feature-leakage suite
├── model_logic/              Feature builder, registry, models, folds, and aggregation
├── modeling/                 Earlier conditional-logit scripts retained for comparison
├── tests/                    Core synthetic and contract tests
├── requirements-modeling.txt Dependency ranges
└── requirements-lock.txt     Freeze of the verified clean-checkout environment
```

Important entry points:

| Path | Purpose |
|---|---|
| [`model_logic/feature_builder.py`](model_logic/feature_builder.py) | Builds active pre-race features. |
| [`model_logic/feature_registry.py`](model_logic/feature_registry.py) | Defines feature metadata, model sets, validation, and registry hash. |
| [`model_logic/run_folds.py`](model_logic/run_folds.py) | Runs a configuration over chronological folds. |
| [`model_logic/run_baselines.py`](model_logic/run_baselines.py) | Runs uniform, draw-only, and frozen historical conditional-logit baselines. |
| [`model_logic/run_ablation.py`](model_logic/run_ablation.py) | Coordinates baselines, all ablations, and aggregation. |
| [`model_logic/aggregate_folds.py`](model_logic/aggregate_folds.py) | Produces fold metrics, evidence status, and ladder outputs. |
| [`analysis/provenance.py`](analysis/provenance.py) | Checks registry version/hash and published commit metadata. |
| [`analysis/final_scoring.py`](analysis/final_scoring.py) | Hashes a frozen predictions file and supports one-shot split-based final scoring. |

## Data

The tracked processed input is
[`data/clean data/runners_merged.parquet`](data/clean%20data/runners_merged.parquet).
At the verification commit it contains **39,210 rows and 51 columns**, measured directly from
that parquet file. The source schema and cleaning notes are
[`data/clean data/data_dictionary.md`](data/clean%20data/data_dictionary.md) and
[`data/clean data/cleaning_report.md`](data/clean%20data/cleaning_report.md).

The documented sources are:

| `source` value | Coverage documented in the repository | Local source file |
|---|---|---|
| `kaggle` | 2017/18 through 2019/20 | `data/Horse Racing Results_2017-2020.numbers` |
| `hkjc` | 2025/26 plus 2026/27 through 2026-10-11 | `data/hkjc_results_2025-07-26_to_2026-10-11.csv` |

Data acquisition is **not reproducible from this repository**: no downloader or scraper command
is present. No licence or terms are documented for the Kaggle data or the team's HKJC scrape,
and redistribution has not been checked. The processed parquet is nevertheless tracked; do not
modify it during modelling runs.

### Processed schema

The following groups list the actual parquet columns and their general role. See the linked data
dictionary for source-field mappings, units, and full definitions.

| Group | Columns | Meaning |
|---|---|---|
| Race identity and declarations | `source`, `race_id`, `race_date`, `season`, `track`, `track_code`, `race_no`, `distance_m`, `surface`, `prize_hkd`, `race_type_raw`, `race_type`, `race_class`, `race_grade`, `is_griffin`, `is_handicap`, `field_size_declared`, `field_size` | Race identity, date, venue, distance, class, and declared field context. |
| Runner declarations | `horse_id`, `jockey_id`, `jockey_name`, `jockey_key`, `trainer_id`, `trainer_name`, `trainer_key`, `draw`, `carried_weight_lb`, `weight_exact`, `horse_country`, `horse_age`, `official_rating`, `profile_missing` | Horse, jockey, trainer, draw, weight, and profile fields. `profile_missing` is never a model feature. |
| Market | `win_odds`, `odds_capped`, `implied_prob`, `market_prob` | At-the-off odds and normalized market probability; benchmark or Stage 2 input only. |
| Outcomes | `finish_code_raw`, `finish_status`, `started`, `finished`, `finish_pos`, `dead_heat`, `is_winner`, `is_top3`, `finish_time_s`, `race_winner_time_s`, `running_positions`, `kaggle_fg_figure`, `kaggle_path` | Labels and post-race fields; only earlier-race outcomes may contribute to history features. |
| Validity | `race_valid`, `race_issue` | Invalid-race marker and reason. |

Data types include pandas `string`, nullable `Int64` and `Float64`, `boolean`, and
`datetime64[ns]`; the exact dtype map was read from the processed parquet during verification.

Core cleaning rules are:

- model only `race_valid & started` rows;
- preserve missing values rather than filling them with zero;
- exclude invalid races and non-starters from history;
- never use outcome or market columns as Stage 1 features;
- treat market-complete races as the common comparison set in the ladder.

[`docs/DATA_RULES.md`](docs/DATA_RULES.md) is the authoritative rule set.

## Setup

The commands below were verified on **Python 3.13.5** only. No claim is made for other Python
versions or operating systems. Verification ran on macOS; LightGBM installation on other
platforms is not verified here.

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-modeling.txt pytest
```

[`requirements-modeling.txt`](requirements-modeling.txt) specifies dependency ranges only.
[`requirements-lock.txt`](requirements-lock.txt) is a `pip freeze` snapshot from the verified
clean-checkout environment; it is not a cross-platform guarantee.

Commands require `PYTHONPATH=.` because this repository has no package metadata or installable
project configuration. The verified tests took 86.95 seconds for the core suite and 18.73
seconds for the leakage suite. The verified `k1` model-run report recorded 5.30 seconds inside
the fold runner; end-to-end memory use and wall time are not documented.

## Quickstart

These are the commands executed successfully in a fresh clone of commit
`b6dcde81dfdb2c2ca4e154e47449e960920a1afb`. Run them from the repository root.

```bash
PYTHONPATH=. .venv/bin/python -m model_logic.feature_registry --validate
```

Validates the registry and prints its version and hash.

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests
PYTHONPATH=. .venv/bin/python -m pytest -q "leakage test/test_leakage.py"
```

Runs the core suite and the separate leakage suite. They are intentionally separate commands.

```bash
PYTHONPATH=. .venv/bin/python -m model_logic.run_folds \
  --folds model_logic/validation/folds.csv \
  --fold-ids k1 \
  --out artifacts/readme-smoke
```

Builds features, runs the primary configuration on `k1`, and writes predictions, selection
traces, coefficients, blend metadata, and `meta.json` under
`artifacts/readme-smoke/primary/k1/`. This is a smoke run, not evidence.

## Reproducing the evidence

### Features and stability

1. Validate the registry with the Quickstart command.
2. Run the separate leakage suite before changing feature logic.
3. The stability script is [`analysis/feature_stability_audit.py`](analysis/feature_stability_audit.py).
   Its default output directory is
   [`analysis/feature_stability_results/`](analysis/feature_stability_results/). Its existing
   report is not a substitute for a fresh run; see [Known issues](#known-issues-and-open-items).

### Forward folds

[`model_logic/validation/folds.csv`](model_logic/validation/folds.csv) defines the fold windows:

| Fold | Era | s1 | s2 | Valid |
|---|---|---|---|---|
| `k1` | Kaggle | 2017-09-01–2019-01-31 | 2019-02-01–2019-07-31 | 2019-09-01–2020-01-31 |
| `k2` | Kaggle | 2017-09-01–2019-07-31 | 2019-09-01–2020-01-31 | 2020-02-01–2020-06-30 |
| `h1` | HKJC | 2017-09-01–2025-10-31 | 2025-11-01–2026-01-31 | 2026-02-01–2026-03-31 |

Within `s1`, the last four months form the inner tune period by default. Stage 1 feature
selection and LightGBM round selection occur only inside this training period. Stage 2 uses
`s2`; `valid` is scored after those choices. `h1` mixes Kaggle and HKJC data in `s1`.

The code-defined full coordinator is `python -m model_logic.run_ablation` with
`--folds`, optional `--data`, `--out`, `--seeds`, `--inner-tune-months`, `--min-gain`,
`--fold-ids`, and `--resume`; its complete execution was **not** part of the clean-checkout
verification, so it is not a verified quickstart command. The implementation is nevertheless
organized as one coordinator that reuses the same folds, feature builder, baseline artifacts,
and aggregation rules across all configurations.

### Ablations

| Config | Exclusion | Interpretation |
|---|---|---|
| `primary` | None | Primary configuration. |
| `no_trainer_rate` | Registered trainer-rate features | Exploratory trainer-rate ablation. |
| `no_track_distance` | `track_distance` group | Exploratory track/distance ablation. |
| `no_trainer_no_track_distance` | Both exclusions | Exploratory combined ablation. |

### Baseline ladder

The aggregation code orders these rungs:

1. uniform;
2. draw-only;
3. frozen historical conditional logit;
4. market;
5. calibrated market;
6. conditional-logit-only;
7. LightGBM-only;
8. market plus conditional logit;
9. market plus LightGBM;
10. market plus both models.

`draw_only` and `original_cl` are fitted once per fold from `s1` without market inputs.
Aggregation joins their valid predictions to each configuration's market-complete valid races
and raises if provenance keys or race counts differ.

### Outputs and intervals

The coordinator writes, per configuration:

| Output | Contents |
|---|---|
| `fold_metrics.csv` | Per-fold model loss, gains versus market and calibrated market, and paired intervals. |
| `feature_selection_frequency.csv` | Conditional-logit feature counts across folds. |
| `ladder.csv` | Ordered rung losses and changes from uniform, market, calibrated market, and the previous rung. |
| `ladder_steps.csv` | Requested paired rung comparisons, gains, and confidence intervals. |
| `ladder.md` | Readable ladder and step tables. |
| `meta.json` | Fold, data, registry, config, package, and runtime provenance. |

With all configurations, aggregation also writes `ablation_metrics.csv`,
`ablation_feature_selection_frequency.csv`, `evidence_summary.json`, and
`evidence_summary.csv` under `artifacts/folds/`. These files are ignored for new paths and
must be regenerated before they are cited.

The interval helper
[`model_logic/validation/evaluation.py`](model_logic/validation/evaluation.py) uses a paired
bootstrap clustered by race day: `n_boot=2000`, `seed=0`. A positive gain means the second
probability column has lower log loss.

## Feature registry

The registry is a versioned description of every active or planned feature. A `FeatureSpec`
records its name, group, description, inputs, history window, shrinkage, race-relative flag,
optional flag, model memberships, monotonic sign, stability status, implementation flag,
version, and leakage note.

The verified registry contains 56 implemented features:

| Group | Count |
|---|---:|
| `draw_weight` | 2 |
| `horse_form` | 13 |
| `interaction` | 5 |
| `jockey_trainer` | 6 |
| `partnership` | 12 |
| `race_context` | 2 |
| `track_distance` | 12 |
| `trajectory` | 4 |

Use the registry CLI:

```bash
PYTHONPATH=. .venv/bin/python -m model_logic.feature_registry --list
PYTHONPATH=. .venv/bin/python -m model_logic.feature_registry --validate
PYTHONPATH=. .venv/bin/python -m model_logic.feature_registry --export /tmp/feature_registry.json
```

`keep`, `watch`, and `review` are stability-audit statuses; `unaudited` indicates no audit
status has been assigned. The generated registry reference
`docs/feature_registry.md` is not present in this repository. The source of truth is
[`model_logic/feature_registry.py`](model_logic/feature_registry.py), and the existing JSON
snapshots are [`model_logic/feature_registry_snapshot.json`](model_logic/feature_registry_snapshot.json)
and [`model_logic/feature_registry_snapshot_v3.1.json`](model_logic/feature_registry_snapshot_v3.1.json).

To add a feature end to end, define its `FeatureSpec`, build it in the shared builder or its
documented helper, include it in the intended model set, validate the registry, extend the
leakage and contract tests, rerun the stability audit, and then regenerate chronological folds.

## Leakage and validation design

The feature rules are deliberately conservative:

- history uses strictly earlier race **days**, never earlier races on the same day;
- histories are separated by `source`, and horse identifiers are never linked across sources;
- outcomes and market values are excluded from Stage 1;
- folds are chronological and race-disjoint;
- Stage 1 selection is confined to the inner part of `s1`, Stage 2 fitting to `s2`, and scoring
  to `valid`;
- `h_test` is prohibited from feature, model, blend, calibration, threshold, and freeze choices.

The separate leakage suite covers same-day perturbations, source isolation, cutoff/future
invariance, market exclusion, invalid rows and non-starters, forbidden names, and negative
controls. Run it with the exact Quickstart command. Core contract tests additionally cover
fold ordering, outcome isolation, resume keys, provenance checks, baseline isolation, and
ladder integrity.

Provenance helpers compare registry version/hash and a commit's membership in `origin/main`.
[`analysis/provenance.py`](analysis/provenance.py) documents the checks and their limitations.

## Evaluation protocol

[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) fixes the primary comparison as
the `primary` configuration's `mkt+cl+lgbm` variant against calibrated market, using per-race
log loss. A gain is calibrated-market loss minus model loss.

“Repeatedly” means all eligible folds must have positive calibrated-market gain and a
two-sided 95% paired race-day bootstrap interval entirely above zero. The current policy
requires at least 150 inner-tune races; `h1` is watch-only below that threshold. All other
ladder comparisons are exploratory. `h_test` cannot influence selection or freezing and may
be scored once only after a valid freeze.

## Findings and limitations

### Results status

Results are generated by the fold/ablation pipeline, with ladder and evidence outputs under
`artifacts/folds/`. Those generated ladder and evidence files are git-ignored. The tracked fold
metadata records registry `v3.0` and commit `fc759fe`, so it is stale and is not evidence.
This section will be completed only after a clean current-registry v3.1 rerun. Until then, the
repository still provides a complete evaluation path: common-race comparisons, race-day
uncertainty intervals, explicit baseline steps, and a pre-specified freeze rule.

### General limitations

Log-loss differences do not establish profitable wagering after takeout, odds timing, and
bankroll constraints. No economic evaluation is implemented in this repository. The original
`h_test` period was exposed during earlier exploration according to
[`docs/report_data.md`](docs/report_data.md); it is not an untouched prospective test.

## Freeze and final scoring

[`analysis/final_scoring.py`](analysis/final_scoring.py) has dry-run, `--freeze`, and
`--final` modes. Freeze records SHA-256 and MD5 hashes of the predictions file, Git state,
split/race counts, model columns, and optional S3 version information. Final scoring requires
the confirmation phrase, a matching SHA-256, passing provenance, and no existing final-output
directory.

No freeze record exists at
`analysis/final_scoring_results/freeze_record.json`. The implementation is one-shot and
split-based (`h_test`), not an enforced rule that all scored races have dates after a recorded
freeze timestamp. Do not treat it as a completed prospective-test implementation.

## Testing

Run the suites separately:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests
PYTHONPATH=. .venv/bin/python -m pytest -q "leakage test/test_leakage.py"
```

On the verified clean checkout, the expected counts were 62 core tests and 26 leakage tests.
The core suite covers folds, ablation coordination, baseline ladder behavior, provenance,
feature context parity, trainer-rate ablation, and v3 feature construction. The leakage suite
tests historical-data isolation and negative controls.

Run an individual core test, for example:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_run_folds.py
```

## Reproducibility notes

- The fold runner's default two seeds are `17` and `118`; seed count defaults to two.
- The paired race-day bootstrap uses `n_boot=2000` and `seed=0`.
- Fold metadata records data hash, registry version/hash, Git commit, config hash, package
  versions, period counts, and runtime. Baseline metadata additionally records a dirty-worktree
  flag and frozen baseline-feature hash.
- Resume reuse checks registry/data/config keys for model folds and registry/data/feature-list
  keys for baselines.
- Provenance detects stale registry versions and unpublished commits, but it does not reject a
  dirty worktree or compare data hashes across all evidence workflows.
- No thread-count setting or cross-platform determinism guarantee is documented for LightGBM.

## Acknowledgements, data terms, and licence

The repository documents Kaggle race results and a team HKJC scrape as its data sources.
No licence file, Kaggle licence, HKJC terms, or redistribution permission is documented here.
