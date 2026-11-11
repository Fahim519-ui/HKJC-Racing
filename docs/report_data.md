# Report data: Inns's sections (data, validation, results so far, limitations)

Prepared 7 Nov 2026 as a working report-data document. Its pre-11-Nov result
sections are historical diagnostics, not the current submission narrative.
Use [`submission_summary.md`](submission_summary.md) and
`analysis/final_scoring_results/h_test/report.md` for the current accepted
fold result and retrospective holdout score.

---

## 1. Data

**Sources** (`data/clean data/cleaning_report.md`)

| Source | Seasons | Meetings | Races | Runner rows |
|---|---|---:|---:|---:|
| Kaggle HK results | 2017/18, 2018/19, 2019/20 | 240 | 2,224 | 27,008 |
| HKJC website (own scraper) | 2025/26, 2026/27 (to 11 Oct 2026) | 98 | 964 | 12,202 |
| **Merged** (`runners_merged.parquet`) | | | **3,188** | **39,210** |

No free source covers 2020-2025, so there is a five-year gap between the two sources.

**Cleaning** (`clean_data.py`, `cleaning_report.md`, `data_dictionary.md`)
- One row per runner; runners of a race share `race_id`.
- Finish status standardised (HKJC: 12,000 finished, 170 withdrawn, 14 void, 7 pulled up,
  7 unseated, 2 did not finish, 2 disqualified).
- 9 races marked invalid (8 Kaggle: placing gaps, no winner, placing/time mismatch; 1 HKJC
  void race). Modelling rows = valid races, started runners: **38,942 rows, 3,179 races**.
- 10 Kaggle races have one finish-time swap; checked against HKJC pages, some are amended
  placings and some are time typos. Placings kept; times flagged as unreliable for speed figures.
- HKJC horse country / age / rating are missing for 34.0% of the 7,040 HKJC runners before the
  test window (Sep 2025-Mar 2026) and the gap is **not random**: those horses won 4.1% of
  races vs 10.1% for the rest (recomputed 7 Nov from `runners_merged.parquet`, valid started
  HKJC runners with `race_date` < 2026-04-01, `profile_missing`; the cleaning report's 28.5% /
  4.0% / 9.7% cover the whole 2025/26 season, which includes test races), because they had left
  Hong Kong racing before the scrape. `profile_missing` is therefore never a feature, and the
  rows are kept to avoid biasing the sample.

**Time splits** (`model_logic/feature_builder.py` `assign_split`, counts from the modelling rows)

| Split | Dates | Races | Rows | Role |
|---|---|---:|---:|---|
| warmup | Sep 2017-Oct 2017, Sep-Oct 2025 | 293 | 3,542 | history only |
| k_train | Nov 2017-Jul 2019 | 1,307 | 15,836 | Stage 1 training |
| k_tune | Sep 2019-Jun 2020 | 756 | 9,299 | feature selection, tuning |
| h_s2 | Nov 2025-Mar 2026 | 423 | 5,287 | Stage 2 (market blend) fitting |
| h_test | Apr-Oct 2026 | 400 | 4,978 | scored once, Week 5 |

**AWS** (Weeks 1-2): private S3 bucket in ap-southeast-1 with budget alerts; SageMaker access
and runbooks for the team. The combined predictions file and its report are on S3 under
`predictions/` and `reports/`.

## 2. Validation methods

**Leakage tests** (`leakage test/test_leakage.py`, `leakage test/result.md`)
- 26 tests over **55 features**, including the 18 v3.1 track/distance (`td_`), trajectory
  (`tr_`) and interaction (`int_`) features. All pass; the core suite in `tests/` (85 tests on
  `main` at `0594916`, 11 Nov) also passes (`analysis/v3_build_results/verification_2026-11-11.md`).
- Method: change the data and watch what moves. (1) Shuffle one race day's results: no
  feature on that day or earlier may change. (2) Shuffle one source: the other source must
  not change. (3) Scale the odds: no Stage 1 feature may change. (4) Cut-off: delete every
  race after a date: no earlier feature may change (catches averages over the whole period).
- 8 negative controls: each check is also run on a deliberately leaky feature and must catch
  it, so a pass is meaningful.
- Found and fixed before reaching a model: a whole-period average in the first version of
  the track/distance features, and a same-day leak in the 6 Nov version of the v3.1
  track/distance features (`leakage test/result.md`, update of 6 Nov: the same-day and cut-off
  tests now pass on every `td_`/`tr_`/`int_` feature).

**Feature stability audit** (`analysis/feature_stability_audit.py`,
`analysis/feature_stability_results/stability_report.md`)
- Compares each feature's distribution (SMD, PSI, blank rate) and its effect on winning
  across Kaggle train, Kaggle tune and HKJC Stage 2.
- The **38 model inputs** of registry v3.1 (LightGBM uses all 38, the logit 32):
  **28 keep, 9 watch, 1 review** (`analysis/feature_stability_results/STABILITY_V3_1.md`, 8 Nov;
  registry and audit agree on every input). The one "review" input is the race-relative
  trainer win rate `jt_trainer_win365_rel`: effect +0.29 ± 0.03 in Kaggle vs +0.09 ± 0.07 in
  HKJC, and the gap holds against a Kaggle window with equally short histories.
- The default audit run covers 42 columns, the 38 inputs plus 4 raw rates no model reads
  directly: 30 keep, 10 watch, 2 review (`stability_report.md`). Its second "review" is the raw
  trainer rate `jt_trainer_win365`, the building block of the input above.

**Scoring** (`analysis/eval_utils.py`, `analysis/README_EVALUATION.md`)
- Per-race log loss (primary), Brier score, top-1 accuracy, pseudo-R², calibration by decile.
- Baselines: raw market, power and Shin normalisation, and a "market refit" (p ∝ market^β,
  β fitted on the fitting window only) that corrects the favourite-longshot bias.
- 95% intervals from a paired bootstrap that resamples **whole race days** (races on one
  day are correlated), 2,000 resamples. Dead heats scored as −log of the mean winner probability.
- A guard refuses h_test rows (by split or date ≥ 2026-04-01) unless the final run is sanctioned.

**Provenance check** (`analysis/provenance.py`, added 7 Nov)
- Freeze and final scoring refuse predictions unless the run's `meta.json` records the live
  registry version and a git commit that is on GitHub `origin/main`.
- Added because all 6 Nov fold runs recorded registry v3.0 and an unpushed commit (`fc759fe`)
  although the registry was v3.1; the scores themselves showed no sign of this.

**Out-of-sample designs**
- **Evidence (8 Nov):** only the pre-registered v3.1 fold re-run (`docs/evaluation_protocol.md`):
  k1 and k2 are eligible (at least 150 inner-tune races), h1 is watch-only; any ladder
  candidate with positive calibrated-market gain in one eligible fold may freeze. The 95%
  race-day intervals are reported but do not block freezing.
- Fahim's fold runner (`model_logic/run_folds.py`): folds k1, k2 (Kaggle) and h1 (HKJC
  Feb-Mar 2026) with a separate Stage 2 window per fold. Both scoring codes give identical
  losses and intervals on the same files (`analysis/FOLD_RUNNER_COMPARISON.md`).
- Rolling validation (`analysis/rolling_validation.py`): 3 chronological Kaggle folds; tree
  count chosen inside training; blend weights fitted on an out-of-fold inner split.
  **Diagnostic, not evidence** (`analysis/quarantine.json`): stand-in models, no run record.

## 3. Results so far (no h_test)

**Status (8 Nov):** every result in this section comes from a run that is now quarantined as
non-evidence (`analysis/quarantine.json`), except the longshot-bias table. They are kept as
diagnostics and for orientation; the report quotes only the v3.1 re-run. See
`docs/report_inns_sections.md` section 4.3 for the evidence placeholders.

**Earlier "residual edge" claim not supported** (diagnostic; `leakage test/result.md` §3): +0.0013
log-loss gain over the market when the scored period also chose the tree count; +0.00005
(−0.0008 to +0.0009) when chosen on training data only; −0.0009 (−0.0036 to +0.0018)
against the bias-corrected market.

**Rolling validation, registry v3.1 features, Kaggle only** (`analysis/rolling_validation_results/report.md`)
*diagnostic, not evidence (stand-in models, no run record; quarantined)*

| Model | Pooled log loss | Gain vs market refit | 95% CI |
|---|---:|---:|---|
| market refit | 1.9523 | 0 | - |
| conditional logit alone | 2.1460 | −0.1937 | −0.2288 to −0.1579 |
| LightGBM alone | 2.1565 | −0.2042 | −0.2413 to −0.1681 |
| market + logit | 1.9519 | +0.0003 | −0.0013 to +0.0020 |
| market + LightGBM | 1.9525 | −0.0003 | −0.0028 to +0.0022 |
| market + both | 1.9533 | −0.0010 | −0.0040 to +0.0021 |

Removing the trainer features changes nothing measurable (market + logit −0.0000,
−0.0025 to +0.0024). This ablation drops the columns whose names start with `jt_trainer` and
keeps `int_jt_strength_x_field_size`, which also uses the trainer rate
(`analysis/rolling_validation.py` `main`), so it is not the same ablation as the fold runner's
`no_trainer_rate` (`model_logic/ablation_configs.py`, `TRAINER_RATE_FEATURES`).

**Fahim's fold runner** (stale v3.0 runs, now `artifacts/quarantine/folds_v3.0_fc759fe/`), pooled k1+k2+h1, 927 races, gain vs calibrated
market: market + logit +0.0016 (−0.0005 to +0.0037); market + both +0.0015 (−0.0007 to
+0.0038). *Stale: registry v3.0, unpushed code. Re-run on v3.1 before quoting.*

**Stage 2 weights** (`analysis/stage2_results/report.md`, in-sample) *diagnostic, not evidence
(quarantined); the report uses the weights in each fold's `blend.json` from the v3.1 re-run
instead. Those are also fitted on the fold's Stage 2 window (`model_logic/run_folds.py`
`add_stage2_variants`), but they are scored on the later validation window, so the blend's
validation loss is out of sample even though the weights themselves are not*. The whole
table is stamped UNVERIFIED: no window passes the provenance check yet. The fold windows come
from the stale v3.0 runs, and the h_s2 window comes from the 5 Nov LightGBM v2 predictions in
the combined file, which have no `meta.json`. Re-run on v3.1 before quoting. As it stands: on
HKJC Stage 2 the logit gets a **negative** weight (α = −0.344,
SE 0.106) with β = 1.237 on the market; on Kaggle windows α is near 0 (k1 +0.050, k2 −0.063,
SE ≈ 0.10). Read: on HKJC the blend gains by leaning against the model, acting as an extra
favourite-longshot correction rather than adding the model's information.

**Favourite-longshot bias** (`artifacts/market_diagnostics/longshot_bias_evidence.md`):
longshots (market deciles 1-2) win less often than the odds imply in every non-test split
(observed/predicted 0.889 Kaggle train, 0.657 Kaggle tune, 0.717 HKJC Stage 2). Favourites:
1.058, 0.978, 1.050. Consistent with the literature (Snowberg & Wolfers 2010).

**Orientation only, not a headline:** none of these diagnostics showed a gain over the
bias-corrected market whose 95% interval excludes zero, and the only pattern that held in every
split is the longshot bias. The headline waits for the v3.1 primary comparison.

## 4. Limitations (paste-ready)

> **The test period was seen before the final run.** It was seen twice. First, in Week 2
> (mid-October), the exploratory analysis (`data/clean data/02_eda.ipynb`) was run on all
> HKJC races, including the later test window: it reported
> the market's per-race log loss, market calibration, the favourite's win rate, draw bias
> and win rate by carried weight over all HKJC races. Second, on 4 Nov, model scripts and a
> market diagnostic committed test-window results to the repository: per-race log loss for
> the market, both models and the blends, and market calibration by probability decile.
> Both were read by team members during development. Nothing was fitted or tuned on the
> test window: the Stage 2 blend, the market-refit β and all feature choices use Kaggle data
> and HKJC Stage 2 (Nov 2025-Mar 2026) only, and the leakage tests and stability audit never
> use test outcomes. Some design choices (for example the draw × Happy Valley feature) may
> have been informed by the Week 2 overview, although the same patterns are well known and
> appear in the Kaggle years. The 4 Nov rows were removed on 7 Nov but remain in the public
> repository's history, so the final test result should be read as a lightly exposed
> hold-out rather than an untouched one. The stability audit also reports the distribution
> of each feature in the test window (inputs only, no outcomes).

Not for the report, but for the team: the scripts that wrote the 4 Nov test-window rows still
compute them when re-run. `model_logic/lightgbm_independent_probability.py` and
`lightgbm_independent_probability_v2.py` write h_test log losses to `metrics.csv` and
`report.md`, and `model_logic/market_diagnostics.py` computes market calibration for every
split including h_test (`calibration_tables`). Do not re-run them before the final scoring.

Points still to confirm before submitting: whether the decision to use the favourite-longshot
(market^β) correction was influenced by the test-window calibration seen in the Week 2 EDA or on 4 Nov. The same
bias appears in Kaggle and HKJC Stage 2, so the method stands either way, but if it was,
the report should say so.

Other limitations:
- **Small HKJC sample.** h_s2 has 423 races; the HKJC fold h1 has 171 races on 17 race days,
  so its intervals are wide. h1's validation window lies inside h_s2, so it is not fresh
  evidence for the main blend.
- **Five-year gap** between Kaggle (to 2020) and HKJC (from Sep 2025); short HKJC histories
  shift several features. The trainer rate behaves differently across eras.
- **Earlier runs are diagnostics only:** the 6 Nov fold runs (registry v3.0, unpushed commit,
  no v3.1 features) and the rolling validation (stand-in models) are quarantined; only the
  v3.1 re-run is evidence. The provenance check blocks quarantined runs from freeze and scoring.
- **Few folds, strict rule:** two eligible folds of about 380 races give wide intervals; the
  pre-registered repeat rule can fail even if a small real edge exists.
- **Code provenance:** until 8 Nov the fold runner recorded the commit even with uncommitted
  edits. The run-gate now refuses a dirty tree or an unpushed commit, so only runs made through
  it prove their code.
- **Standard errors** in the Stage 2 table treat races as independent; same-day races are
  correlated, so true uncertainty is larger.
- **Non-random missing data** in HKJC horse profiles (handled by never using missingness).
- **Better probabilities ≠ profit:** the HKJC takeout is not modelled.
- **Track B** (running-style models) not built; sub-question 1 must be answered from the
  feature-family ablations instead, or Track B dropped explicitly.

## 5. References

- Benter, W. (1994). Computer based horse race handicapping and wagering systems: a report.
  In D. B. Hausch, V. S. Y. Lo & W. T. Ziemba (Eds.), *Efficiency of Racetrack Betting
  Markets* (pp. 183-198). Academic Press.
- Bolton, R. N. & Chapman, R. G. (1986). Searching for positive returns at the track: a
  multinomial logit model for handicapping horse races. *Management Science* 32(8).
- Shin, H. S. (1993). Measuring the incidence of insider trading in a market for
  state-contingent claims. *Economic Journal* 103(420).
- Snowberg, E. & Wolfers, J. (2010). Explaining the favorite-long shot bias: is it risk-love
  or misperceptions? *Journal of Political Economy* 118(4).
