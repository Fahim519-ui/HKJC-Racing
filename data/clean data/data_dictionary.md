# Data dictionary - `data/processed/runners_merged`

One row per horse per race, from two sources mapped into the same columns by `clean_data.py`.

| Source | File | Period | Rows | Races |
|---|---|---|---|---|
| `kaggle` | `data/Horse_Racing_Results_2017-2020.csv` (Kaggle: bogdandoicin/horse-racing-results-2017-2020; identical to the .numbers/.xls copies) | 3 Sep 2017 - 21 Jun 2020 | 27,008 | 2,224 (2,216 valid) |
| `hkjc` | `data/hkjc_results_2025-07-26_to_2026-10-11.csv` (our scraper) | 7 Sep 2025 - 11 Oct 2026 | 12,202 | 964 (963 valid) |

**When known** decides whether a column may be a model input:

- **Before** = known at declarations, so it can be a feature.
- **At the off** = final odds, settled when betting closes. Use it as the benchmark, or as a Stage 2 input, never as a "fundamental" feature.
- **After** = the result. Use it only as a label, or as history from *earlier* races.

**Missing values stay missing (NaN).** Nothing is filled with 0. Any imputation happens in modelling, fitted on the training period only.

## Race columns

| Column | Type | Values / unit | HKJC source | Kaggle source | When known | Notes |
|---|---|---|---|---|---|---|
| `source` | text | `hkjc` / `kaggle` | - | - | - | |
| `race_id` | text | `2025-09-07_ST_01` | built | built | Before | date_track_raceNo |
| `race_date` | date | | `Dato` (yyyy/mm/dd) | `Dato` (dd.mm.yyyy) | Before | |
| `season` | text | `2025/26` | built | built | Before | HK season runs Sep-Jul |
| `track`, `track_code` | text | Sha Tin / Happy Valley, ST / HV | `Track` | `Track` | Before | |
| `race_no` | int | 1-11 | `Race Number` | `Race Number` | Before | |
| `distance_m` | int | metres | `Distance` | `Distance` | Before | |
| `surface` | text | Turf / All Weather | `Surface` | `Surface` (Gress -> Turf, Dirt -> All Weather) | Before | |
| `prize_hkd` | int | HK$ | `Prize money` | `Prize money` | Before | |
| `race_type_raw` | text | original text | `RaceType` | `RaceType` (Norwegian) | Before | kept for checking |
| `race_type` | text | Class 1-5, Group G1-G3, Griffin, 4YO Series / Handicap, Non-handicap, Griffin, Unclassified | parsed | parsed | Before | **The two sources use different systems** |
| `race_class` | int | 1-5 | from `RaceType` | **missing** | Before | NaN for Group/Griffin races |
| `race_grade` | text | G1 / G2 / G3 | from `RaceType` | **missing** | Before | |
| `is_griffin` | bool | | from `RaceType` | from `RaceType` | Before | Griffin races are for unraced newcomers |
| `is_handicap` | bool | | **missing** | from `RaceType` | Before | |
| `field_size_declared` | int | rows in race | built | built | Before | includes withdrawn horses (HKJC) |
| `field_size` | int | starters | built | built | Before* | *late withdrawals can change it. Kaggle lists finishers only |
| `race_winner_time_s` | float | seconds | last value of `Race time` | fastest `Race time` | After | label / history only |
| `race_valid` | bool | | built | built | - | **False = do not model** (void, no winner, missing rows) |
| `race_issue` | text | `void_race`, `no_winner`, `placing_gap`, `fewer_than_2_starters`, `duplicate_horse`, `placing_time_mismatch` | built | built | - | why `race_valid` is False. `placing_time_mismatch` = placings disagree with the horses' own times in 2+ places (Kaggle 2018-02-10 ST R7: the real winner, at 1.3, is listed 5th) |

## Runner columns

| Column | Type | Values / unit | HKJC source | Kaggle source | When known | Notes |
|---|---|---|---|---|---|---|
| `horse_id` | text | `HK_2019_D090` / `KAGGLE_1736` | `HorseId` | `HorseId` | Before | **cannot be linked across sources** |
| `jockey_id`, `trainer_id` | text | `HKJC_AA` / `KAGGLE_8656` | `JockeyId`, `TrainerID` | `JockeyId`, `TrainerID` | Before | IDs differ between sources |
| `jockey_name`, `trainer_name` | text | as written | `Jockey`, `TrainerName` | `Jockey`, `TrainerName` | Before | "J Moreira" vs "Joao Moreira" |
| `jockey_key`, `trainer_key` | text | `J MOREIRA`, `CH YIP` | built | built | Before | **use these to link people across sources** (see report for the few links made by first initial + surname) |
| `draw` | int | 1-14 | `Starting position` | `Starting position` | Before | NaN for withdrawn horses |
| `carried_weight_lb` | float | lb | `Jockey weight` (lb) | `Jockey weight` (kg x 2.20462) | Before | Kaggle kg was rounded, so +/- about 1 lb |
| `weight_exact` | bool | | True | False | - | |
| `horse_country` | text | AUS, NZ, IRE, GB, FR, USA, SAF, GER, JPN, BRZ, ARG ... | `Country` | `Country` (Norwegian -> code) | Before | HKJC: blank for about 26%, almost all in 2025/26 (horses that had left HK by the scrape). **Not random** - see `profile_missing` |
| `horse_age` | int | years | `Horse age` | `Horse age` | Before | HKJC value is estimated by the scraper; blank like country. Kaggle has some age 2 and 12 |
| `official_rating` | float | HKJC rating points | `FGrating` | **missing** | Before | **pre-race** rating the horse ran off (checked: rises about 6.7 after a win, sits inside class bands). HKJC blank for about 27% |
| `profile_missing` | bool | | country and age blank | False | - | **Never a model feature.** The gap marks horses that left HK racing before the scrape (Oct 2026), i.e. future information: in 2025/26 they win 4.0% of races vs 9.7%. Use it only to check results. Don't drop these rows (see report) |

## Market columns

| Column | Type | Unit | HKJC source | Kaggle source | When known | Notes |
|---|---|---|---|---|---|---|
| `win_odds` | float | decimal (HK$ return per HK$1) | `Odds` | `Odds` | At the off | NaN for withdrawn |
| `odds_capped` | bool | | always False | True when odds = 99 before 13 Mar 2019 | - | Kaggle capped odds at 99 until then, so "99" means "99 or longer" (1,217 runners). Treat as censored |
| `implied_prob` | float | 1 / odds | built | built | At the off | starters in valid races only |
| `market_prob` | float | implied_prob / race total | built | built | At the off | **the baseline to beat**; sums to 1 per valid race |

## Outcome columns (labels / history only)

| Column | Type | Values | HKJC source | Kaggle source | Notes |
|---|---|---|---|---|---|
| `finish_code_raw` | text | `1`, `1 DH`, `WV`, `PU` ... | `Final place` | `Final place` | as published |
| `finish_status` | text | FINISHED, WITHDRAWN, PULLED_UP, UNSEATED, FELL, DID_NOT_FINISH, DISQUALIFIED, TOOK_NO_PART, VOID | parsed | always FINISHED | Kaggle has no non-finishers |
| `started` | bool | | parsed | True | withdrawn = False |
| `finished` | bool | | parsed | True | |
| `finish_pos` | int | 1-14 | number part of `Final place` | `Final place` | NaN if not finished |
| `dead_heat` | bool | | `n DH` | same place twice in a race | |
| `is_winner`, `is_top3` | bool | | built | built | **targets** |
| `finish_time_s` | float | seconds | **missing** (not in the scraped file) | `Race time` | horse's own time. Kaggle has a few typos (e.g. 2017-12-10 ST R4 winner 2:26.63, HKJC says 2:26.23); the report lists races where a time is out of order |
| `running_positions` | text | `2 3 13` | `Path` | **missing** | last number = finishing place |
| `kaggle_fg_figure` | float | | **missing** | `FGrating` | **Not the HKJC rating.** Behaves like a post-race performance figure (0.77 rank correlation with the result). Never use it for the same race |
| `kaggle_path` | int | 0-11 | **missing** | `Path` | **Meaning unknown** - not running positions. Do not use |

## Not available in either file

Going, course rail (A/B/C...), lengths behind the winner, declared horse body weight, horse name, gear, class rating band, each horse's finish time (HKJC).
