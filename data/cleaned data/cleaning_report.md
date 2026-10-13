# Cleaning report

## Inputs
- hkjc: `data/hkjc_results_2025-07-26_to_2026-10-11.csv`
- kaggle: `data\Horse_Racing_Results_2017-2020.csv`

## Rows and races by season

| source | season | first | last | meetings | races | runners | valid races |
|---|---|---|---|---|---|---|---|
| hkjc | 2025/26 | 2025-09-07 | 2026-07-15 | 88 | 866 | 10962 | 865 |
| hkjc | 2026/27 | 2026-09-06 | 2026-10-11 | 10 | 98 | 1240 | 98 |
| kaggle | 2017/18 | 2017-09-03 | 2018-05-16 | 72 | 656 | 7960 | 653 |
| kaggle | 2018/19 | 2018-09-02 | 2019-07-14 | 88 | 810 | 9723 | 807 |
| kaggle | 2019/20 | 2019-09-01 | 2020-06-21 | 80 | 758 | 9325 | 756 |

## Finish status

| source | status | rows |
|---|---|---|
| hkjc | DID_NOT_FINISH | 2 |
| hkjc | DISQUALIFIED | 2 |
| hkjc | FINISHED | 12000 |
| hkjc | PULLED_UP | 7 |
| hkjc | UNSEATED | 7 |
| hkjc | VOID | 14 |
| hkjc | WITHDRAWN | 170 |
| kaggle | FINISHED | 27008 |

## Race issues (race_valid = False)

| source | issue | races |
|---|---|---|
| hkjc | void_race;no_winner;fewer_than_2_starters | 1 |
| kaggle | no_winner | 2 |
| kaggle | placing_gap | 5 |
| kaggle | placing_time_mismatch | 1 |

Races:

- kaggle `2017-10-25_HV_01`: placing_gap
- kaggle `2018-01-17_HV_03`: placing_gap
- kaggle `2018-02-10_ST_07`: placing_time_mismatch
- kaggle `2019-01-27_ST_02`: placing_gap
- kaggle `2019-04-07_ST_03`: no_winner
- kaggle `2019-07-07_ST_04`: placing_gap
- kaggle `2019-09-08_ST_10`: no_winner
- kaggle `2020-02-23_ST_06`: placing_gap
- hkjc `2025-11-15_ST_08`: void_race;no_winner;fewer_than_2_starters

## Finish-time order (valid races with exactly one swap)

A horse placed behind another has a faster own time. Checked on HKJC pages: some are real amended placings (protests, e.g. 2018-10-07 ST R7); others are time typos in the Kaggle file (2017-12-10 ST R4 winner 2:26.63 there, 2:26.23 on HKJC; 2019-04-14 ST R5 10th 1:51.76 there, 1:50.76 on HKJC). Placings are kept. Treat these times with care if used for speed figures.

- `2017-11-01_ST_06`
- `2017-12-10_ST_04`
- `2018-10-07_ST_07`
- `2019-03-24_ST_06`
- `2019-04-14_ST_05`
- `2019-04-28_ST_02`
- `2019-10-09_HV_08`
- `2019-12-18_HV_02`
- `2020-01-08_HV_02`
- `2020-05-09_ST_06`

## Missing values (share of modelling rows: valid races, started runners)

| column | hkjc | kaggle |
|---|---|---|
| race_class | 3.6% | 100.0% |
| race_grade | 97.1% | 100.0% |
| is_handicap | 100.0% | 0.5% |
| horse_country | 25.9% | 0.0% |
| horse_age | 25.9% | 0.0% |
| official_rating | 27.5% | 100.0% |
| finish_pos | 0.1% | 0.0% |
| finish_time_s | 100.0% | 0.0% |
| running_positions | 0.0% | 100.0% |
| kaggle_fg_figure | 100.0% | 0.0% |
| kaggle_path | 100.0% | 0.0% |

### HKJC missing country / age / rating is NOT random

| season | runners | profile missing | win rate (profile missing) | win rate (profile present) | median odds (missing) | median odds (present) |
|---|---|---|---|---|---|---|
| 2025/26 | 10,797 | 28.5% | 4.0% | 9.7% | 22 | 14 |
| 2026/27 | 1,221 | 2.9% | 0.0% | 8.3% | 23 | 15 |

The gaps are almost all in 2025/26 and belong to weaker horses that had left Hong Kong racing by the time the scraper ran. A gap therefore says something about the horse's FUTURE.

- Do not use `profile_missing`, or the fact that a value is missing, as a model feature (leakage).
- Do not drop these rows: that would remove mostly weaker horses and bias the sample.
- If `horse_age` / `horse_country` / `official_rating` are used for hkjc rows, impute them (fitted on training dates only) and check the result with and without these columns.

## Jockey / trainer name keys shared by both sources

- jockey_key: hkjc 47, kaggle 89, shared 27 (e.g. A ATZENI, A BADEL, B AVDULLA, C DEMURO, C LEMAIRE, C SOUMILLON, C WILLIAMS, CL CHAU)
- trainer_key: hkjc 45, kaggle 86, shared 22 (e.g. A FABRE, AM BALDING, AP O'BRIEN, AS CRUZ, C FOWNES, CH YIP, CS SHUM, CW CHANG)
  - linked by first initial + surname (please check): kaggle 'D HALL' -> hkjc 'DJ HALL'
  - linked by first initial + surname (please check): kaggle 'A O'BRIEN' -> hkjc 'AP O'BRIEN'
  - linked by first initial + surname (please check): kaggle 'A BALDING' -> hkjc 'AM BALDING'
  - linked by first initial + surname (please check): kaggle 'D WHYTE' -> hkjc 'DJ WHYTE'

## Market check

- kaggle odds are capped at 99 from 2017-09-03 to 2019-03-10: 1,217 runners have `odds_capped = True` (true odds were 99 or longer)
- hkjc: median sum of 1/odds per race = 1.227 (about 1/(1-0.175) = 1.212 with HKJC's 17.5% deduction); favourite won 29.2% of valid races
- kaggle: median sum of 1/odds per race = 1.229 (about 1/(1-0.175) = 1.212 with HKJC's 17.5% deduction); favourite won 31.3% of valid races

## Reminders

- `finish_time_s` is empty for hkjc: the scraped file has no per-horse finish time.
- `kaggle_fg_figure` and `kaggle_path` are NOT the HKJC rating / running positions; never use them as features for the same race.
- Kaggle weights were rounded to whole kg in the source, so `carried_weight_lb` is +/- about 1 lb.
- Kaggle odds before 13 Mar 2019 are capped at 99 (`odds_capped`); long shots' market_prob is slightly too high there.
- Filter `race_valid & started` for modelling.