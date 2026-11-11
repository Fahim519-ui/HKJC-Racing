# Report draft: Inns's sections (data, validation, results, limitations)

Draft of 7 Nov 2026, written from `docs/report_data.md`, where every number has a source
file. Updated 11 Nov with the v3.1 fold evidence (`artifacts/folds/`, runs on commit
`1042a3d`, all passing the provenance check), the overfitting checks
(`analysis/overfitting_results/`) and the retrospective h_test diagnostic, consistent with
`docs/submission_summary.md`. Earlier designs are diagnostics (`analysis/quarantine.json`)
and are not quoted as results. Remaining `[TO CONFIRM]` / `[TEAM NOTE]` brackets are for the
team, not the report. Final report due 23:59 Tue 17 Nov.

---

## 2. Data

### 2.1 Sources

We use two sources of Hong Kong Jockey Club (HKJC) race results. The first is a public
Kaggle file covering three seasons, 2017/18 to 2019/20: 240 race meetings, 2,224 races and
27,008 runners. The second we scraped from the HKJC website ourselves: the 2025/26 season
and the start of 2026/27 up to 11 October 2026, which is 98 meetings, 964 races and 12,202
runner rows. No free source covers 2020 to 2025, so there is a five-year gap between the
two. Merged, the data holds 3,188 races and 39,210 runner rows (`runners_merged.parquet`).

Each row is one runner in one race and carries the pre-race information (draw, carried
weight, jockey, trainer, race type and class, distance, track, final win odds) and the result.
The going, the course rail and the horse's body weight are not in either source
(`data/clean data/data_dictionary.md`).
The final win odds give the market's view. The median sum of 1/odds per race is 1.227 for
HKJC and 1.229 for Kaggle, close to the 1.212 implied by the HKJC's 17.5% deduction, and
the favourite won 30.7% of HKJC races before the test window (563 races) and 31.3% of
Kaggle races. Kaggle odds are capped at 99
until 10 March 2019 (1,217 runners), which affects only the longest shots.

### 2.2 Cleaning

`clean_data.py` turns both sources into one schema, documented in `data/clean data/data_dictionary.md`.
Finish status is standardised. In the HKJC data 12,000 runners finished, 170 were withdrawn,
14 were in a void race, 7 pulled up, 7 unseated, 2 did not finish and 2 were disqualified;
all Kaggle runners finished. Nine races are marked invalid and excluded from modelling:
eight Kaggle races with gaps in the placings, no winner, or placings that contradict the
finish times, and one void HKJC race. The modelling sample is every started runner in a
valid race: 38,942 rows in 3,179 races.

Ten Kaggle races contain one pair of runners whose finish times are in the wrong order. We
checked them against the HKJC website. Some are real amended placings after a protest and
some are typing errors in the times. We keep the official placings and do not use the times.

HKJC horse country, age and rating are missing for 34.0% of the 7,040 HKJC runners before the
test window (September 2025 to March 2026), and the gap is not random. Those horses won 4.1%
of their races against 10.1% for the rest, because they had
left Hong Kong racing before we scraped their profiles. Whether a profile is missing
therefore carries information about the horse's future. We never use missingness as a
feature, and we keep those rows, because dropping them would remove mostly weaker horses
and bias the sample.

### 2.3 Time splits

Every split is by race date, never by random rows, and every feature is built only from
earlier race days within the same source.

| Split | Dates | Races | Runner rows | Role |
|---|---|---:|---:|---|
| warm-up | Sep-Oct 2017, Sep-Oct 2025 | 293 | 3,542 | history only, not scored |
| k_train | Nov 2017 - Jul 2019 | 1,307 | 15,836 | Stage 1 model training |
| k_tune | Sep 2019 - Jun 2020 | 756 | 9,299 | feature selection and tuning |
| h_s2 | Nov 2025 - Mar 2026 | 423 | 5,287 | Stage 2 (market blend) fitting |
| h_test | Apr - Oct 2026 | 400 | 4,978 | scored once, after the freeze |

Horse identifiers are never linked across the two sources. A horse's HKJC history starts
in September 2025, so HKJC form features rest on shorter histories than Kaggle ones.

### 2.4 Infrastructure

Data and outputs are stored in a private Amazon S3 bucket in the Singapore region
(ap-southeast-1), with public access blocked and budget alerts on the account. Team members
work through SageMaker with individual access roles. The combined predictions file used
for final scoring and its report are kept on S3 next to the repository copy. When the freeze
is run with the S3 location (`--s3-uri`), the freeze record also stores the S3 version
identifier and whether the uploaded copy's ETag matches the local file's MD5.

## 3. Validation methods

### 3.1 Leakage tests

A model that sees information from after the race start looks better than it is. We test
for this directly instead of relying on code review. The suite (`leakage test/test_leakage.py`,
26 tests) covers 55 features, including the 18 track/distance, trajectory and interaction
features added in registry v3.1. Each test changes the data and checks what moves:

1. **One race day.** Shuffle the results of a single race day. No feature on that day or
   any earlier day may change.
2. **One source.** Shuffle one source's results. Features of the other source may not change.
3. **Odds.** Rescale all the odds. No Stage 1 (fundamental) feature may change.
4. **Cut-off.** Delete every race after a date. No feature before that date may change. This
   catches averages taken over the whole period.

Eight negative controls run the same checks on deliberately leaky features (for example a
jockey win rate that counts his earlier rides on the same day). Each must be caught, so a
pass is meaningful. The suite found two real problems before they reached a model: a
whole-period average in the first version of the track/distance features, and a same-day
leak in the 6 November version of those features. Both were fixed, and all 26 tests pass,
as do the 85 tests of the core suite.

### 3.2 Feature stability across eras

The model is trained on 2017-2020 data and tested on 2025-2026 data, so a feature whose
meaning changed between the eras could mislead it. The stability audit
(`analysis/feature_stability_audit.py`) compares each model feature across Kaggle training,
Kaggle tuning and HKJC Stage 2: its distribution (standardised mean difference, population
stability index, blank rate) and its effect on winning, estimated per standard deviation by
a conditional logit. Because HKJC histories are shorter, every HKJC comparison is repeated
against an early Kaggle window with equally short histories. A gap that also appears
against that window is a real change, not a history-length artefact.

Of the 38 model inputs, 28 are marked keep, 9 watch and 1 review
(`analysis/feature_stability_results/STABILITY_V3_1.md`). The review input is the trainer's
race-relative 365-day win rate. Its effect is +0.29 (± 0.03) per standard deviation in Kaggle
but +0.09 (± 0.07) in HKJC, and the gap holds against the history-matched window. (The raw
trainer rate it is built from is flagged too; it is not itself a model input.) The audit is a
warning list, not a selection rule: features are chosen on k_tune only.

### 3.3 Scoring

The race is the unit of analysis. The primary metric is per-race log loss, the negative log
probability the model gave the winner, averaged over races; we also report Brier score,
top-1 accuracy, pseudo-R² against a uniform guess, and calibration by probability decile.
In a dead heat the loss is the negative log of the mean probability of the winners. All
models are compared on exactly the same races.

The benchmark is the betting market. Besides the raw market probabilities we use power and
Shin normalisations of the odds and a market refit, p ∝ market^β, with β fitted only on a
fitting window. The refit corrects the favourite-longshot bias (section 4.2), so a model
that only rediscovers that bias does not count as beating the market. Because the refit can
also miss on a later period, we report gains against the raw market as well (section 4.3).

Differences between two models are tested with a paired bootstrap, 2,000 resamples. It
resamples whole race days, not single races, because races on the same day share the
going and track bias and their results are correlated. The intervals are reported for
context; a positive point-estimate gain is sufficient for the freeze rule.

### 3.4 Out-of-sample designs

The evidence comes from one forward-fold design (`docs/evaluation_protocol.md`). Its fold
table and primary comparison were fixed before the v3.1 re-run; its freeze rule was relaxed
afterwards (see "How the rule changed" below).

- **Fold runner** (`model_logic/run_folds.py`, run through `model_logic/run_ablation.py`):
  folds k1 and k2 on Kaggle and h1 on HKJC (Feb-Mar 2026), each with its own Stage 1
  training, Stage 2 and validation windows (`model_logic/validation/folds.csv`). The Stage 2
  blend is fitted on each fold's Stage 2 window and scored on the later validation window.
- **Primary comparison:** the `primary` configuration's market + logit + LightGBM blend
  against the calibrated market (the same idea as the market refit, p ∝ market^β with β fitted
  on the Stage 2 window), by per-race log loss.
- **Freeze rule:** a fold counts only with at least 150 inner-tune races, which makes k1 and
  k2 eligible and h1 watch-only. Any ladder candidate with positive calibrated-market gain in
  one eligible fold may freeze; its 95% race-day interval is reported but does not block the
  decision.
- Both scoring codes, the fold runner's and `analysis/eval_utils.py`, give identical losses
  and intervals on the same prediction files (`analysis/FOLD_RUNNER_COMPARISON.md`).

**How the rule changed.** Before the v3.1 re-run the protocol required every eligible fold
(here 2 of 2) to show a positive gain with its 95% race-day interval entirely above zero. No
configuration met that rule. After the fold results had been seen, the group relaxed it to
the freeze rule above and froze the best pooled candidate. The relaxed rule and that choice
are therefore operational decisions for this project, not pre-registered confirmatory
evidence, and section 4.3 reports how much a best-of-N choice can gain by chance alone.

Earlier designs are kept as **diagnostics, not evidence** (`analysis/quarantine.json`): the
rolling validation (`analysis/rolling_validation.py`, three Kaggle folds with simplified
stand-in models and no run record), the market-offset LightGBM runs, the in-sample Stage 2
table and the 3-5 November model runs. They helped find problems but are not quoted as results.

### 3.5 Run provenance and the one-shot test

A result is only as good as our knowledge of the code and features that made it. Every
model run records the feature registry version and the git commit in a `meta.json`. Freeze
and final scoring (`analysis/final_scoring.py`) refuse predictions unless that run used the
current registry and a commit published on the shared repository
(`analysis/provenance.py`). This check was added after we found that all fold runs of
6 November had used an older registry and unpublished code, so they lacked the v3.1
features. Nothing in their scores showed this. Since then an evidence run also refuses to
start unless the working tree is clean and its commit is on the shared repository, and it
records the data, registry and configuration hashes; those 6 November runs are quarantined.

The test window is scored once. The freeze step records the SHA-256 of the predictions
file, the git commit, the freeze timestamp, each split's race dates and, if given, the S3
version. The final run refuses to start without that
record, if the file has changed, or if a result already exists, and it refits nothing on
the test window: the market-refit β and the Stage 2 blend carry their h_s2 fits.

## 4. Results

### 4.1 An earlier "edge" did not survive a fair check (diagnostic)

An earlier experiment reported that a LightGBM model starting from the market odds improved
log loss by +0.0013 per race on four Kaggle walk-forward periods. In that setup the scored
period also chose the number of trees. Choosing it on training data only, the gain fell to
+0.00005 (95% CI −0.0008 to +0.0009), and against the bias-corrected market it was −0.0009
(−0.0036 to +0.0018). How the comparison is set up mattered more than the model itself. That
model starts from the odds, so it is a residual diagnostic, not an independent comparator,
and its runs are quarantined; the point here is the method lesson, not the number.

### 4.2 The favourite-longshot bias

In every non-test split, longshots (the two lowest market-probability deciles) win less
often than their odds imply: observed / predicted is 0.889 in Kaggle training, 0.657 in
Kaggle tuning and 0.717 in HKJC Stage 2. Favourites (top two deciles) win about as often as
implied or slightly more: 1.058, 0.978 and 1.050. Each longshot group holds only 7 to 26
winners, so each ratio is noisy, but the direction is the same in all three splits. This
matches the literature (Snowberg & Wolfers, 2010) and is why the market refit is the
baseline.

### 4.3 Validation results

All numbers in this section come from the v3.1 fold runs (`artifacts/folds/`), made on the
published commit `1042a3d` from a clean tree through the run-gate; every run passes
`analysis/provenance.py`.

**Primary comparison** (market + logit + LightGBM vs calibrated market, per-race log loss;
`artifacts/folds/primary/fold_metrics.csv`; figure `analysis/overfitting_results/primary_gain_by_fold.png`):

| Fold | Role | Validation window | Races | Race days | Gain | 95% race-day CI | Above zero? |
|---|---|---|---:|---:|---:|---|---|
| k1 | eligible | Sep 2019 - Jan 2020 | 379 | 40 | −0.00041 | −0.00274 to +0.00171 | no |
| k2 | eligible | Feb - Jun 2020 | 377 | 40 | +0.00065 | −0.00104 to +0.00230 | no |
| h1 | watch-only | Feb - Mar 2026 | 171 | 17 | +0.00506 | −0.00581 to +0.01464 | reported only |
| pooled | descriptive | | 927 | 97 | +0.00103 | −0.00126 to +0.00331 | descriptive |

The original rule (2 of 2 eligible folds positive with intervals above zero) is not met: k1 is
negative and neither eligible interval excludes zero. Under the relaxed freeze rule (a
positive point estimate in one eligible fold), 18 candidates qualify, including the raw market
itself with no model, which shows that the relaxed rule does not separate a model from noise.

**Baseline ladder** (`artifacts/folds/primary/ladder.csv`, pooled, 927 races; gain vs
calibrated market in per-race log loss): uniform −0.510, draw-only −0.484, original
conditional logit −0.208, raw market +0.0004, calibrated market 0, logit alone −0.205,
LightGBM alone −0.204, market + logit +0.0011, market + LightGBM +0.0008, market + both
+0.0010. On their own, the models are far behind the market (about −0.20 per race); blended
with the market they move it by about one thousandth, inside the noise.

Out of sample the calibrated market is not better than the raw odds: against calibrated
market the raw market scores −0.0006 (k1), +0.0018 (k2), −0.0004 (h1) and +0.0004 pooled.
The β fitted on each Stage 2 window does not transfer reliably to the next period. Against
the raw market the gains are therefore smaller: primary market + both +0.0006 pooled
(k2 −0.0012), and the frozen no-track-distance market + logit +0.0010 pooled (k2 −0.0010);
every interval contains zero (`fold_metrics.csv`, `gain_vs_market`).

**Feature-group ablations** (`artifacts/folds/ablation_metrics.csv`; market + both vs
calibrated market):

| Configuration | k1 | k2 | h1 | Pooled (95% CI) |
|---|---:|---:|---:|---|
| primary (all v3.1 features) | −0.00041 | +0.00065 | +0.00506 | +0.00103 (−0.00126 to +0.00331) |
| without trainer rate | −0.00067 | +0.00037 | +0.00446 | +0.00070 (−0.00137 to +0.00281) |
| without track/distance | −0.00023 | +0.00094 | +0.00598 | +0.00139 (−0.00104 to +0.00387) |
| without both | −0.00054 | +0.00076 | +0.00656 | +0.00130 (−0.00088 to +0.00358) |

Every configuration is negative in k1 and every interval contains zero. Removing a feature
group changes the pooled gain by less than 0.0004 per race, so no group is shown to help or hurt.

**Stage 2 weights out of sample** (each fold's `blend.json`; weights fitted on the fold's
Stage 2 window, scored on its later validation window). The blend chose the strongest penalty
in the grid (λ = 0.1 of 0, 0.003, 0.01, 0.03, 0.1) in every fold. The penalty pulls the
blend towards the raw market (β = 1, model weights 0), and the month-by-month forward loss
inside the Stage 2 window was still falling at 0.1 in all 12 fold runs, so the best penalty
may lie beyond the grid: the data favour moving even closer to the raw odds. The calibrated-market weight β is 1.106 (k1), 1.060 (k2) and 1.005 (h1).
The model weights are small and change sign: in market + logit, α_cl = +0.034, −0.006 and
−0.119; in market + both, (α_cl, α_lgbm) = (+0.028, +0.011), (+0.005, −0.020) and
(−0.080, −0.073). On HKJC (h1) both model weights are negative: the blend moves away from
the models' probabilities, so h1's gain is not evidence that the models carry information the
market lacks. This matches the earlier in-sample diagnostic.

**Overfitting and selection checks** (`analysis/overfitting_check.py`,
`analysis/overfitting_results/report.md`; folds only, no model refitted):

1. *In-sample vs out-of-sample.* Over 36 config × blend × fold cases the mean gain is +0.0017
   on the Stage 2 window where the blend is fitted and +0.0018 on the later validation window;
   in-sample is larger in only 15 of 36. The Stage 2 blend is not overfitting. This is a weak
   test: the blend has at most three weights, all pulled towards the raw market by the
   penalty, so it has little room to overfit, and the two windows are different periods, so part of the difference is the
   period, not the fit.
2. *Stability.* Of 16 conditional-logit features ever selected, only 6 are selected in all
   three folds (draw, draw × Happy Valley, carried weight, recent form, days since the last
   run, jockey win rate); fold overlap (Jaccard) 0.40-0.64. LightGBM's best iteration ranges
   from 310 to 508. The stable features are standard handicapping factors that the odds
   are likely to reflect already.
3. *Best of N vs chance.* We compared 12 candidates (4 feature configurations × 3 blends). A
   race-day bootstrap with every candidate centred at zero gives the size of the best gain
   under no real edge. The best observed candidate is within that range: on the eligible
   folds, no-track-distance market + LightGBM +0.00065 against a chance 95th percentile of
   +0.00144 (selection-adjusted p = 0.31); on all three folds, no-track-distance market +
   logit +0.00142 against +0.00203 (p = 0.17). Figure
   `analysis/overfitting_results/best_of_n_vs_chance.png`.

In short: the blend does not overfit, but the models add almost no information beyond the
market, and the best configuration is what picking the winner of 12 produces by
chance.

Two design points. First, h1 has 171 races on 17 race days, so its intervals are wide.
Second, h1's validation window lies inside h_s2, so it is not independent of the HKJC Stage 2
fit.

### 4.4 Retrospective h_test diagnostic

After the fold results were seen, the group froze the best pooled candidate,
no-track-distance market + logit (pooled fold gain +0.0014, 95% CI −0.0007 to +0.0036; k1
−0.0004, k2 +0.0008), and scored it once on h_test (1 Apr-11 Oct 2026). Its gain against the
calibrated market there is −0.0027 (95% CI −0.0102 to +0.0051), and about −0.003 against
the raw market (log loss 2.0593 vs 2.0565): no edge
(`docs/submission_summary.md`, `analysis/final_scoring_results/h_test/report.md`). Because the
test races were seen during exploration and the model was chosen after the fold results, this
is a retrospective diagnostic, not prospective evidence.

**Headline:** no model or blend shows a reliable gain over the bias-corrected market. The
pre-registered comparison fails, the best of 12 configurations is within chance, and the
retrospective test window is negative. The pattern that held in every split is the
favourite-longshot bias, which the calibrated market already corrects.

## 5. Limitations

**The test period was seen before the final run.** It was seen twice. First, in Week 2
(mid-October), the exploratory analysis (`data/clean data/02_eda.ipynb`) was run on all
HKJC races, including the later test window: it reported
the market's per-race log loss, market calibration, the favourite's win rate, draw bias
and win rate by carried weight over all HKJC races. Second, on 4 Nov, model scripts and a
market diagnostic committed test-window results to the repository: per-race log loss for
the market, both models and the blends, and market calibration by probability decile.
Both were read by team members during development. Nothing was fitted or tuned on the
test window: the Stage 2 blend, the market-refit β and all feature choices use Kaggle data
and HKJC Stage 2 (Nov 2025-Mar 2026) only, and the leakage tests and stability audit never
use test outcomes. Some design choices (for example the draw × Happy Valley feature) may
have been informed by the Week 2 overview, although the same patterns are well known and
appear in the Kaggle years. The 4 Nov rows were removed on 7 Nov but remain in the public
repository's history, so the final test result should be read as a lightly exposed
hold-out rather than an untouched one. The stability audit also reports the distribution
of each feature in the test window (inputs only, no outcomes).
[TEAM NOTE, not for the report: the scripts that wrote the 4 Nov rows are now under
`legacy_code/` and still compute test-window numbers if re-run. Do not re-run them.]
[TO CONFIRM with Fahim: whether the choice of the market^β correction was influenced by the
test-window calibration seen in the Week 2 overview or on 4 November. The same bias appears in Kaggle and HKJC
Stage 2, so the method stands either way; if it was influenced, say so here.]

Other limitations:

- **Small HKJC sample.** h_s2 has 423 races and the HKJC fold h1 171 races on 17 race days,
  so intervals on HKJC data are wide. h1 lies inside h_s2, so it is not fresh evidence for
  the main blend.
- **Five-year gap between the sources.** Short HKJC histories shift several features, and
  the trainer win rate behaves differently across the eras.
- **Few folds, low power.** Two eligible folds of about 380 races each give wide
  intervals, so the original rule could fail even if a small real edge exists, and the
  relaxed rule cannot separate a small edge from noise.
- **Earlier runs are diagnostics only.** The rolling validation used simplified models, and
  the 6 November fold runs used an older registry and unpublished code. Both are quarantined
  and only the v3.1 re-run is evidence.
- **Code provenance.** Until 8 November the fold runner recorded the commit even if the
  working tree had uncommitted edits. The run-gate now refuses such runs; earlier runs cannot
  prove their code.
- **Rule changed after the results.** The freeze rule was relaxed after the fold results were
  seen, and the frozen model is the best of 12 candidates; its selection-adjusted p is 0.17
  (0.31 on k1 and k2 only).
- **Two seeds.** LightGBM was bagged over 2 seeds per fold, so part of the fold-to-fold
  variation may be seed noise.
- **Non-random missing data** in HKJC horse profiles, handled by never using missingness.
- **Probability is not profit.** Better log loss than the market does not imply a profit
  after the HKJC's 17.5% deduction, which we do not model.
- **Track B** (HKJC-only running-style and rating models) was not built. Sub-question 1 (which
  feature groups matter) is answered from the feature-group ablations in section 4.3.

## References

- Benter, W. (1994). Computer based horse race handicapping and wagering systems: a report.
  In D. B. Hausch, V. S. Y. Lo & W. T. Ziemba (Eds.), *Efficiency of Racetrack Betting
  Markets* (pp. 183-198). Academic Press.
- Bolton, R. N. & Chapman, R. G. (1986). Searching for positive returns at the track: a
  multinomial logit model for handicapping horse races. *Management Science*, 32(8),
  1040-1060.
- Shin, H. S. (1993). Measuring the incidence of insider trading in a market for
  state-contingent claims. *The Economic Journal*, 103(420), 1141-1153.
- Snowberg, E. & Wolfers, J. (2010). Explaining the favorite-long shot bias: is it risk-love
  or misperceptions? *Journal of Political Economy*, 118(4), 723-746.
