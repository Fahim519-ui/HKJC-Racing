# Leakage tests: what they check and how to use them

**File:** `leakage test/test_leakage.py` (33 tests, about 25 seconds).
**Owner:** Ek-Anankul Wirith (Inns). Roadmap of 4 Nov, execution step 2.
**Status (5 Nov 2026):** all 33 pass on the current `model_logic/` code. Not committed.

Every structural check runs against **both** feature builders a live model could use:
- `cl_v2` = `conditional_logit_tuning_v2.build_features` - what both models run today
- `fb` = `model_logic/feature_builder.build_features` - Fahim's newer builder (the parity
  test proves it is numerically identical; the models will switch to it)

So whichever builder the models use, it is covered. When the models move fully to
`feature_builder.py`, drop the `cl_v2` entry from `BUILDER_NAMES` in the test file.

---

## 1. Why these tests exist

A model "leaks" when a feature contains information that would not be known before the race starts. A leaky model looks great in testing and is useless in real life. In this project the rules are in `DATA_RULES.md`:

1. **Earlier race days only.** A feature may use results from previous days, never from the same day. A jockey rides many races a day, so even his earlier rides that day are off limits.
2. **Same source only.** Kaggle (2017-20) and HKJC (2025-26) histories never mix.
3. **No post-race or market columns.** No `finish_*`, `is_winner`, `running_positions`, etc. No odds in Stage 1.
4. **No peeking at the future through an average.** A feature must not shrink or normalise toward a mean taken over the whole period (which includes the unseen test races). Nayeon's planned track/distance features currently do exactly this.

Reading the code can miss a leak. These tests check the rules by **changing the data and watching what moves**.

## 2. The main idea in one picture

```
Take the real data  ->  build features  ->  "before" table
Swap the results of every race on ONE day (who won, placings, times, running positions)
                    ->  build features  ->  "after" table
Compare:
   features on that day or earlier   must be IDENTICAL   (nothing knew the result)
   features on later days            must CHANGE          (history really is used)
   features in the other source      must be IDENTICAL   (sources never mix)
```

If an earlier-day feature changes, it was reading the result it is not allowed to see.

How the swap works: inside each chosen race, the results are shuffled between the horses, as a block. So each race still has one winner and the same placings, but different horses get them. Pre-race facts (horse, jockey, draw, weight, date) are never touched.

**The swap test has a blind spot, so there is a second idea: the cut-off test.** Shuffling results *within* a race leaves the overall average for that source unchanged. So a feature that leans on a whole-period average (rule 4) would not move under the swap, and would pass by luck. The cut-off test instead **removes every race after a mid-season date and rebuilds**. Any feature dated on or before the cut-off that uses only earlier data must be byte-for-byte identical. A feature built from a whole-period mean changes, because dropping the later races changes that mean, and it is caught.

```
Build on the full data            ->  "before" table
Build again with every race after a mid-season cut-off removed  ->  "after" table
Every feature dated on/before the cut-off must be IDENTICAL in both.
```

## 3. Setup (once)

```powershell
cd "C:\Users\Lenovo\Documents\HKU3\ELEC3544\Group project\6"
.\.venv\Scripts\Activate.ps1
python -m pip install pytest==9.2.0      # already installed on 4 Nov
```

The modelling packages (scipy, statsmodels, scikit-learn, lightgbm) must be installed too, because the tests load Fahim's scripts. They already are in this `.venv`.

**Data:** the tests use `data/clean data/runners_merged.parquet` inside the repo. To use another file:

```powershell
$env:HKJC_DATA = "C:\path\to\runners_merged.parquet"
```

## 4. Run them

```powershell
cd github\HKJC-Racing
python -m pytest "leakage test\test_leakage.py" -v
```

Good output ends with:

```
============================= 33 passed in 23.36s =============================
```

Each structural test runs twice, once per builder, shown as `[...-cl_v2]` and `[...-fb]` in the test name. To run one test only, add `-k` and part of its name, e.g. `-k cutoff` or `-k same_day`.

## 5. What each test checks

### Group 1: same-day exclusion (rule 1)

| Test | What it does | Passes when |
|---|---|---|
| `test_one_days_results_only_move_later_days` | Swaps all results on one middle day of HKJC Stage 2, and separately one middle day of Kaggle train | No feature on that day or earlier changes. Some later features do change. Nothing in the other source changes. |
| `test_person_rate_is_equal_for_all_rides_on_a_day` | Looks at the jockey and trainer 365-day win rates | Every ride of one person on one day has the same value. (If his 3rd ride counted his 1st, they would differ.) |
| `test_horse_runs_at_most_once_a_day` | Checks no horse appears twice on a day | Always. The horse features use "previous start", which is only safe if this holds. |

### Group 2: source separation (rule 2)

| Test | What it does | Passes when |
|---|---|---|
| `test_one_source_never_moves_the_other` | Swaps every Kaggle result, then separately every HKJC result | The other source's features do not move at all |
| `test_first_day_in_each_source_gets_the_prior` | Looks at each jockey and trainer on their first day in each source | The rate is exactly the prior, 1/12, so nothing was carried over |
| `test_people_shared_across_sources_exist` | Counts jockeys found in both sources | At least 10 (there are 27). Without shared people, the separation test could not fail. |

### Group 3: no post-race or market columns (rule 3)

| Test | What it does | Passes when |
|---|---|---|
| `test_no_forbidden_column_is_a_model_feature` | Checks every feature name any of the three scripts can use | No name is on the forbidden list (`finish_*`, `is_winner`, `is_top3`, `running_positions`, `race_winner_time_s`, `kaggle_*`, `dead_heat`, `finished`, odds columns, `mkt_*`, `profile_missing`, Nayeon's `early_position` / `position_change`) |
| `test_saved_artifact_features_are_clean_and_covered` | Reads the feature lists Fahim saved with his results (`selected_features.json`, `stage1_features.json`) | They are clean, and every one is checked by the tests above |
| `test_odds_never_move_a_stage1_feature` | Multiplies all odds by random numbers | No feature moves |
| `test_invalid_races_and_non_starters_never_feed_history` | Swaps the results of the 9 invalid races and the non-starters | No feature moves (history is built from valid, started runners only) |

### Group 4: cut-off stability (rule 4)

| Test | What it does | Passes when |
|---|---|---|
| `test_cutoff_earlier_features_are_stable` | Drops every race after a mid-season cut-off (Kaggle: Feb 2019; HKJC: mid-Jan 2026), rebuilds, and compares | Every feature dated on or before the cut-off is identical. This is the only check that catches a whole-period-mean leak. |

### Group 5: negative controls (do the tests actually work?)

A test that can never fail proves nothing. So each check is also run on a **deliberately leaky feature**, and must catch it:

| Control | The planted leak | Caught by |
|---|---|---|
| `test_control_same_race_leak_is_caught` | A feature that is simply this race's `is_winner` | the one-day test |
| `test_control_same_day_leak_is_caught` | A jockey win rate that counts his earlier rides the same day | the one-day test |
| `test_control_look_ahead_leak_is_caught` | A feature that peeks at the horse's next race | the one-day test (earlier days change) |
| `test_control_cross_source_leak_is_caught` | A career jockey rate with no source split (how Nayeon's Week 3 rate is built) | the source test |
| `test_control_odds_leak_is_caught` | A feature that is the market log-odds | the odds test |
| `test_control_excluded_rows_leak_is_caught` | A jockey win count that includes invalid races and non-starters | the invalid-rows test |
| `test_control_whole_period_mean_leak_is_caught` | A rate shrunk toward the whole-source mean (Nayeon's `td_*` pattern) | the cut-off test |
| `test_control_forbidden_name_is_caught` | Names like `finish_pos`, `mkt_logit` | the name check |

All 8 controls are caught. So when the real features pass, it means something.

**Checked against Nayeon's integrated builder.** Her `integrated_feature_builder.py` (not yet used by any model) builds the `td_*` track/distance features with whole-source-mean shrinkage. Running the cut-off test against it flags all 18 `td_*` columns (about 12,400 pre-cutoff rows move). That is the proof the cut-off test will catch her leak once her features are merged into a live builder.

## 6. What a failure looks like and what to do

Example message:

```
AssertionError: 2026-01-14: {'same_source_on_or_before_day': {'jt_pair_win365': 152}, ...}
```

Read it as: "after swapping the results of 14 Jan 2026, the feature `jt_pair_win365` changed for 152 rows on or before that day". That feature can see same-day results.

Steps:
1. Find where that feature is built in `build_features()`.
2. Look for: a groupby without `"source"`; a rolling window that includes today (needs `closed="left"` or "strictly earlier day" logic); a `shift(-1)` (looks ahead); a column from the forbidden list.
3. Fix it and run the tests again. Never "fix" a failure by removing the test.

| Failing part | Usual cause |
|---|---|
| `same_source_on_or_before_day` | Same-day or same-race result used, or a look-ahead |
| `other_source` | History grouped without `source` |
| `later_rows_changed` is 0 | The test did not see the swap: the feature builder ignored the results completely, or the day picked had no races |
| Odds test | An odds or market column fed a feature |
| `offenders_on_or_before_cutoff` (cut-off test) | A feature shrinks or normalises toward a whole-period mean. Pull toward a fixed prior (like 1/12) or an earlier-days-only mean instead. |
| Artifact test, "not covered" | Fahim saved a feature that the tests do not know about (see section 7) |

## 7. When Fahim or Nayeon add features (roadmap steps 1 and 2)

New features are tested **automatically** if they are registered in `model_logic/feature_registry.py` (any model set) or added to `ALL_FEATURES` / the LightGBM feature lists. `model_features()` in the test file collects features from the registry joined with those lists.

If a new feature lives in a brand-new list or module the registry doesn't know about, add it to `model_features()`. The artifact test warns you if a saved model uses something the tests don't cover.

Run the tests **after every change to a feature builder**, and before any results are saved.

The roadmap features need special care, and the tests are already set up for them:
- **Jockey-trainer pairs, "last-N" version:** last-N has no time window, so it would cross the five-year gap between Kaggle and HKJC. It must be grouped by `source`. The cross-source control proves the test would catch a miss. (Already merged and passing.)
- **Track/distance form (Nayeon):** must use starts strictly before today, grouped by `source` and horse, and **must not shrink toward a whole-source mean**. Her current version does, so when it is merged into a live builder the cut-off test (Group 4) will fail on the `td_*` features until the shrinkage is changed to a fixed prior or an earlier-days-only mean.

**A note on which builder.** Both models still call `conditional_logit_tuning_v2.build_features`. Fahim's `feature_builder.py` exists and is identical, but no model imports it yet, and Nayeon's `integrated_feature_builder.py` is a copy nothing imports. So her features are in **no** live builder right now, which is why every test passes. The moment someone adds her features to the builder the models actually run, re-run these tests.

## 8. Limits (what these tests do not prove)

- They test the code as it is now. They cannot prove that an old saved result was made by this code. The scripts in the repo match the copies reviewed on 3 Nov, so this is likely but not proven.
- Source separation passes partly because of the data: the jockey/trainer rates look back only 365 days, the gap between datasets is five years, and no horse ID appears in both. The test matters most for future features without a time window.
- Only two days are swapped (one per source). A leak that only shows on certain days (for example, only in meetings with a dead heat) could be missed.
- The cut-off test uses one cut-off date per source. A leak that only appears with a different split could be missed, though a whole-period-mean leak shows at any cut-off.
- They check features only. They do not check how models are fitted or which window is scored (for example, they do not stop a script from computing test-window metrics).
- They compute no metrics and never read test-window results.
