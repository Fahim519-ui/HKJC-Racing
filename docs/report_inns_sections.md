# Report draft: Inns's sections (data, validation, results, limitations)

Draft of 7 Nov 2026, written from `docs/report_data.md`, where every number has a source
file. Two kinds of gap are left on purpose:

- `[PROVISIONAL: ...]` marks a number that will change when the models are re-run on
  registry v3.1 before the freeze (23:59 Wed 12 Nov). The bracket says where the current
  value lives and what replaces it.
- `[FINAL: ...]` marks a number that exists only after the one-shot test scoring in Week 5.

No test-window (h_test) result appears in this draft. Final report due 23:59 Tue 17 Nov.

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
weight, jockey, trainer, class, distance, track, going, final win odds) and the result.
The final win odds give the market's view. The median sum of 1/odds per race is 1.227 for
HKJC and 1.229 for Kaggle, close to the 1.212 implied by the HKJC's 17.5% deduction, and
the favourite won 30.7% of HKJC races before the test window (563 races) and 31.3% of
Kaggle races. Kaggle odds are capped at 99
until 10 March 2019 (1,217 runners), which affects only the longest shots.

### 2.2 Cleaning

`clean_data.py` turns both sources into one schema, documented in `docs/data_dictionary.md`.
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
for final scoring and its report are kept on S3 next to the repository copy, and the freeze
record stores the S3 version identifier.

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
leak in 15 of them in the 6 November builder. Both were fixed, and all 26 tests pass.

### 3.2 Feature stability across eras

The model is trained on 2017-2020 data and tested on 2025-2026 data, so a feature whose
meaning changed between the eras could mislead it. The stability audit
(`analysis/feature_stability_audit.py`) compares each model feature across Kaggle training,
Kaggle tuning and HKJC Stage 2: its distribution (standardised mean difference, population
stability index, blank rate) and its effect on winning, estimated per standard deviation by
a conditional logit. Because HKJC histories are shorter, every HKJC comparison is repeated
against an early Kaggle window with equally short histories. A gap that also appears
against that window is a real change, not a history-length artefact.

Of 42 features, 30 are marked keep, 10 watch and 2 review. The two review features are the
trainer's 365-day win rate and its race-relative version. Its effect is +0.29 (± 0.03) per
standard deviation in Kaggle but +0.09 (± 0.07) in HKJC, and the gap holds against the
history-matched window. The audit is a warning list, not a selection rule: features are
chosen on k_tune only.

### 3.3 Scoring

The race is the unit of analysis. The primary metric is per-race log loss, the negative log
probability the model gave the winner, averaged over races; we also report Brier score,
top-1 accuracy, pseudo-R² against a uniform guess, and calibration by probability decile.
In a dead heat the loss is the negative log of the mean probability of the winners. All
models are compared on exactly the same races.

The benchmark is the betting market. Besides the raw market probabilities we use power and
Shin normalisations of the odds and a market refit, p ∝ market^β, with β fitted only on a
fitting window. The refit corrects the favourite-longshot bias (section 4.2), so a model
that only rediscovers that bias does not count as beating the market.

Differences between two models are tested with a paired bootstrap, 2,000 resamples. It
resamples whole race days, not single races, because races on the same day share the
going and track bias and their results are correlated. Resampling single races made one
interval look significant that was not (section 4.3). A model is said to beat a baseline
only if the whole 95% interval of its gain lies above zero.

### 3.4 Out-of-sample designs

Two chronological designs check whether any gain carries over to later periods.

- **Rolling validation** (`analysis/rolling_validation.py`): three Kaggle folds with
  validation windows Feb-Jul 2019, Aug 2019-Jan 2020 and Feb-Jul 2020. Each fold trains
  only on earlier races, chooses the LightGBM tree count on an inner split inside training,
  and fits the market blend on out-of-fold Stage 1 predictions, so the scored window is
  used once.
- **Fold runner** (`model_logic/run_folds.py`, Fahim): folds k1 and k2 on Kaggle and h1 on
  HKJC (Feb-Mar 2026), each with its own Stage 2 window. Both scoring codes give identical
  losses and intervals on the same prediction files (`analysis/FOLD_RUNNER_COMPARISON.md`).

### 3.5 Run provenance and the one-shot test

A result is only as good as our knowledge of the code and features that made it. Every
model run records the feature registry version and the git commit in a `meta.json`. Freeze
and final scoring (`analysis/final_scoring.py`) refuse predictions unless that run used the
current registry and a commit published on the shared repository
(`analysis/provenance.py`). This check was added after we found that all fold runs of
6 November had used an older registry and unpublished code, so they lacked the v3.1
features. Nothing in their scores showed this.

The test window is scored once. The freeze step records the SHA-256 of the predictions
file, the git commit and the S3 version. The final run refuses to start without that
record, if the file has changed, or if a result already exists, and it refits nothing on
the test window: the market-refit β and the Stage 2 blend carry their h_s2 fits.

## 4. Results

### 4.1 An earlier "edge" did not survive a fair check

An earlier experiment reported that a LightGBM model starting from the market odds improved
log loss by +0.0013 per race on four Kaggle walk-forward periods. In that setup the scored
period also chose the number of trees. Choosing it on training data only, the gain fell to
+0.00005 (95% CI −0.0008 to +0.0009), and against the bias-corrected market it was −0.0009
(−0.0036 to +0.0018). How the comparison is set up mattered more than the model itself.

### 4.2 The favourite-longshot bias

In every non-test split, longshots (the two lowest market-probability deciles) win less
often than their odds imply: observed / predicted is 0.889 in Kaggle training, 0.657 in
Kaggle tuning and 0.717 in HKJC Stage 2. Favourites (top two deciles) win about as often as
implied or slightly more: 1.058, 0.978 and 1.050. Each longshot group holds only 7 to 26
winners, so each ratio is noisy, but the direction is the same in all three splits. This
matches the literature (Snowberg & Wolfers, 2010) and is why the market refit is the
baseline.

### 4.3 Validation results

**Rolling validation (Kaggle).** Pooled gain over the market refit, 95% race-day interval:

| Model | Pooled log loss | Gain vs market refit | 95% CI |
|---|---:|---:|---|
| market refit | [PROVISIONAL: rolling_validation_results/pooled_summary.csv] | 0 | - |
| conditional logit alone | [PROVISIONAL] | [PROVISIONAL] | [PROVISIONAL] |
| LightGBM alone | [PROVISIONAL] | [PROVISIONAL] | [PROVISIONAL] |
| market + logit | [PROVISIONAL] | [PROVISIONAL] | [PROVISIONAL] |
| market + LightGBM | [PROVISIONAL] | [PROVISIONAL] | [PROVISIONAL] |
| market + both | [PROVISIONAL] | [PROVISIONAL] | [PROVISIONAL] |

[PROVISIONAL: replace once the final model scripts and features are fixed for v3.1; re-run
`python analysis/rolling_validation.py`. Current file shows no blend beating the refit.]
Removing the trainer features: [PROVISIONAL: `ablation_no_trainer.csv`].

**Fold runner (k1, k2, h1 pooled).** Gain over the calibrated market: market + logit
[PROVISIONAL], market + LightGBM [PROVISIONAL], market + both [PROVISIONAL]. [PROVISIONAL:
the 6 Nov runs used registry v3.0 and unpublished code; take the numbers from
`artifacts/folds/primary/fold_metrics.csv` (pooled rows) after Fahim's v3.1 re-run, and
check `python analysis/provenance.py` passes first.]

Two design points do not depend on the re-run. First, the HKJC fold h1 (171 races on 17
race days) has wide intervals; in the 6 November runs its market + logit gain looked
significant with single-race resampling and was not with race-day resampling. Second, h1's
window lies inside h_s2, so it is not independent of the main Stage 2 blend.

**Stage 2 weights.** Refitting the blend score = β·log p_market + α·log p_model without a
penalty gives each weight a standard error. On HKJC Stage 2 the logit weight is
α = [PROVISIONAL] (SE [PROVISIONAL]) with β = [PROVISIONAL]; on the Kaggle Stage 2 windows
α = [PROVISIONAL]. [PROVISIONAL: `analysis/stage2_results/alpha_beta.csv`, re-run
`stage2_alpha_beta.py` on the v3.1 predictions. The current, unverified table has a negative
α on HKJC and α near zero on Kaggle. If that holds, the reading is that on HKJC the blend
gains by leaning against the model, which acts as an extra favourite-longshot correction
rather than adding the model's own information.]

### 4.4 Test window (one shot)

[FINAL: `analysis/final_scoring_results/h_test/report.md`. Per model: log loss, pseudo-R²,
Brier, top-1, and gain over the market refit and the raw market with 95% race-day
intervals; races and race days scored; calibration table. State the headline model chosen
before scoring.]

**Headline so far:** no model or blend shows a gain over the bias-corrected market whose
95% interval excludes zero, in any design. The only stable pattern is the longshot bias.
[FINAL: confirm or revise after the test scoring.]

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
[TO CONFIRM with Fahim: whether the choice of the market^β correction was influenced by the
test-window calibration seen in the Week 2 overview or on 4 November. The same bias appears in Kaggle and HKJC
Stage 2, so the method stands either way; if it was influenced, say so here.]

Other limitations:

- **Small HKJC sample.** h_s2 has 423 races and the HKJC fold h1 171 races on 17 race days,
  so intervals on HKJC data are wide. h1 lies inside h_s2, so it is not fresh evidence for
  the main blend.
- **Five-year gap between the sources.** Short HKJC histories shift several features, and
  the trainer win rate behaves differently across the eras.
- **Validation models.** The rolling validation uses simplified versions of the models
  (binary LightGBM, unpenalised logit), not the tuned production models.
- **Code provenance.** The fold runner records the commit even if the working tree had
  uncommitted edits, so only a run from a clean checkout fully proves the code.
- **Stage 2 standard errors** treat races as independent; same-day races are correlated,
  so the true uncertainty is larger.
- **Non-random missing data** in HKJC horse profiles, handled by never using missingness.
- **Probability is not profit.** Better log loss than the market does not imply a profit
  after the HKJC's 17.5% deduction, which we do not model.
- **Track B** (running-style models): [DECIDE: built, or dropped explicitly; if dropped,
  sub-question 1 is answered from the feature-family ablations.]

## References

- Benter, W. (1994). Computer based horse race handicapping and wagering systems: a report.
  In D. B. Hausch, V. S. Y. Lo & W. T. Ziemba (Eds.), *Efficiency of Racetrack Betting
  Markets*. Academic Press.
- Bolton, R. N. & Chapman, R. G. (1986). Searching for positive returns at the track: a
  multinomial logit model for handicapping horse races. *Management Science*, 32(8),
  1040-1060.
- Shin, H. S. (1993). Measuring the incidence of insider trading in a market for
  state-contingent claims. *The Economic Journal*, 103(420), 1141-1153.
- Snowberg, E. & Wolfers, J. (2010). Explaining the favorite-long shot bias: is it risk-love
  or misperceptions? *Journal of Political Economy*, 118(4), 723-746.
