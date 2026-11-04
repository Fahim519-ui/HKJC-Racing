# Leakage tests: what they check and how to use them

**File:** `leakage test/test_leakage.py` (22 tests, about 20 seconds).
**Owner:** Ek-Anankul Wirith (Inns). Roadmap of 4 Nov, execution step 2.
**Status (4 Nov 2026):** all 22 pass on the current `model_logic/` code. Not committed.

---

## 1. Why these tests exist

A model "leaks" when a feature contains information that would not be known before the race starts. A leaky model looks great in testing and is useless in real life. In this project the rules are in `DATA_RULES.md`:

1. **Earlier race days only.** A feature may use results from previous days, never from the same day. A jockey rides many races a day, so even his earlier rides that day are off limits.
2. **Same source only.** Kaggle (2017-20) and HKJC (2025-26) histories never mix.
3. **No post-race or market columns.** No `finish_*`, `is_winner`, `running_positions`, etc. No odds in Stage 1.

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
============================= 22 passed in 15.65s =============================
```

To run one test only, add `-k` and part of its name, e.g. `-k same_day`.

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

### Group 4: negative controls (do the tests actually work?)

A test that can never fail proves nothing. So each check is also run on a **deliberately leaky feature**, and must catch it:

| Control | The planted leak | Caught by |
|---|---|---|
| `test_control_same_race_leak_is_caught` | A feature that is simply this race's `is_winner` | the one-day test |
| `test_control_same_day_leak_is_caught` | A jockey win rate that counts his earlier rides the same day | the one-day test |
| `test_control_look_ahead_leak_is_caught` | A feature that peeks at the horse's next race | the one-day test (earlier days change) |
| `test_control_cross_source_leak_is_caught` | A career jockey rate with no source split (how Nayeon's Week 3 rate is built) | the source test |
| `test_control_odds_leak_is_caught` | A feature that is the market log-odds | the odds test |
| `test_control_excluded_rows_leak_is_caught` | A jockey win count that includes invalid races and non-starters | the invalid-rows test |
| `test_control_forbidden_name_is_caught` | Names like `finish_pos`, `mkt_logit` | the name check |

All 7 controls are caught. So when the real features pass, it means something.

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
| Artifact test, "not covered" | Fahim saved a feature that the tests do not know about (see section 7) |

## 7. When Fahim or Nayeon add features (roadmap steps 1 and 2)

New features are tested **automatically** if they are added to `ALL_FEATURES` in `conditional_logit_tuning_v2.py` (or the LightGBM v1/v2 feature lists). The tests collect features from those lists.

If a new feature lives in a new list or a new file, add that list to `model_features()` in `test_leakage.py`. The artifact test will warn you if a saved model uses something the tests don't cover.

Run the tests **after every change to the feature builder**, and before any results are saved.

The new roadmap features need special care:
- **Jockey-trainer pairs, "last-N" version:** last-N has no time window, so it would cross the five-year gap between Kaggle and HKJC. It must be grouped by `source`. The cross-source control proves the test would catch this.
- **Track/distance form:** must use starts strictly before today, grouped by `source` and horse.

## 8. Limits (what these tests do not prove)

- They test the code as it is now. They cannot prove that an old saved result was made by this code. The scripts in the repo match the copies reviewed on 3 Nov, so this is likely but not proven.
- Source separation passes partly because of the data: the jockey/trainer rates look back only 365 days, the gap between datasets is five years, and no horse ID appears in both. The test matters most for future features without a time window.
- Only two days are swapped (one per source). A leak that only shows on certain days (for example, only in meetings with a dead heat) could be missed.
- They check features only. They do not check how models are fitted or which window is scored (for example, they do not stop a script from computing test-window metrics).
- They compute no metrics and never read test-window results.
