# Progress summary: data cleaning, AWS and modelling (Inns, Group 17)

**Updated:** Sat 31 Oct 2026. Weeks 1-2 were due Fri 16 Oct 2026.

**Status:**

- **Weeks 1-2: done and handed over (Fri 16 Oct).**
  - The cleaned data is checked and was shared with the group on 13 Oct.
  - The AWS pipeline (S3 + SageMaker) works: SageMaker re-ran the cleaning and produced the same table as the laptop.
  - The EDA notebook ran with every number checked; findings saved, EDA PDF made, hand-over message sent.
- **Weeks 3-4: plan and starter pipeline ready.** Owners agreed 31 Oct: **Nayeon** = Week 3 features (laptop), **Fahim** = Week 4 models in his own SageMaker space, **Inns** = data, AWS, Fahim's access and costs. Group target: **Weeks 3-4 done by Wed 11 Nov.**
- **Deadlines (course):** report **Tue 17 Nov 11:59 PM**; presentations **Wed 18 / Wed 25 Nov** (about 10 min per group); a Group Project checkpoint (Homework 3, individual PDF, max 2 pages) - due date to be confirmed.

**Still to do:**

1. **Tonight (31 Oct): give Fahim AWS access** - `docs/FAHIM_ACCESS_RUNBOOK.md` Steps 0-7, almost all with the AWS CLI (incl. the US$20 SageMaker budget and testing as yourself first).
2. Checkpoint PDF (each member's own version) - due date to be confirmed.
3. Optional: run the starter in your own space to have your own reference (`docs/WEEK3-4_RUNBOOK.md`).
4. `eval_utils.py` for Week 5 (owner not agreed yet).
5. Small check: the EDA notebook's findings cell (cell 2) was a **code** cell on 16 Oct. If it still is, switch it to Markdown (Esc, M), save and re-upload.
6. Clean up the accidental Unified Studio setup. Around **Mon 30 Nov**: remove Fahim's access (runbook "Removing his access"; a reminder is set).

---

## 1. My part

From the proposal:

- **Week 1:** inspect both datasets, decide how to clean them, plan the AWS setup.
- **Week 2:** deploy AWS (S3, SageMaker), merge the datasets, run EDA, clean the HKJC fields.

Week 3 (features) comes next. The Weeks 3-4 plan (features + model training) is in `docs/WEEK3-4_PLAN.md`.

| Task | Status |
|---|---|
| Inspect both datasets and decide the cleaning rules | Done |
| Clean and merge the two datasets (`clean_data.py`) | Done, 13 Oct |
| Check the output (automatic checks + HKJC results pages) | Done, 13 Oct |
| Share the cleaned data with the group | Sent, 13 Oct |
| AWS: S3 bucket + SageMaker JupyterLab | Done, 14-15 Oct |
| EDA notebook + findings | Run and checked 15 Oct; findings saved and PDF made 16 Oct |
| Hand-over | **Sent, Fri 16 Oct** (message in section 12) |
| Weeks 3-4 plan, runbook and starter pipeline | Written 16 / 28 Oct, re-checked and fixed 31 Oct (section 13) |
| Weeks 3-4 run in SageMaker | Fahim (Week 4), by Wed 11 Nov |
| Fahim's AWS access (own profile + private space, scoped role, files copied in, US$20 budget) | Files written and tested 31 Oct; to set up 31 Oct (`docs/FAHIM_ACCESS_RUNBOOK.md`) |

---

## 2. Data used (final, no more scraping)

| Source | File | Period | Rows | Races |
|---|---|---|---|---|
| Kaggle (bogdandoicin/horse-racing-results-2017-2020) | `Horse_Racing_Results_2017-2020.csv` | 3 Sep 2017 - 21 Jun 2020 | 27,008 | 2,224 |
| Our HKJC scrape | `hkjc_results_2025-07-26_to_2026-10-11.csv` | 7 Sep 2025 - 11 Oct 2026 | 12,202 | 964 |

The Kaggle file is my CSV conversion of the Numbers file. It gives exactly the same cleaned output as Kaggle's original CSV.

---

## 3. Setup

| Item | Details |
|---|---|
| Machine and editor | My laptop, Kiro |
| Project folder | `C:\Users\Lenovo\Documents\HKU3\ELEC3544\Group project\4` |
| Python | 3.14.7, in a virtual environment (`.venv`). `Activate.ps1` was checked: valid Python Software Foundation signature. |
| Packages | From `requirements-cleaning.txt`: pandas 3.0.6, pyarrow 26.0.0, matplotlib 3.11.2, jupyterlab 4.6.4. `pip check`: no broken requirements. |
| Kiro rules | `.kiro/steering/data-rules.md` |

---

## 4. What the cleaning does

One command, `python clean_data.py`. It never changes the raw files.

1. **Reads both files.** It handles `;` or `,` separators and decimal commas.
2. **Maps both sources into one table** with 51 columns, using the same names and units:
   - dates and track codes;
   - Kaggle's Norwegian values translated (Gress -> Turf, Dirt -> All Weather, country names -> codes);
   - Kaggle weights converted from kg to lb;
   - finishing codes (WV, PU, UR, DISQ, VOID ...) turned into `finish_status`, `started` and `finished`;
   - jockey and trainer **name keys** (e.g. `J MOREIRA`), so the same person can be linked across the two sources.
3. **Adds race-level columns:**
   - `race_id`;
   - field size;
   - winner time;
   - **market probability** (1/odds, normalised so each race sums to 1). This is the baseline the models must beat.
4. **Flags problems instead of deleting rows:**
   - `race_valid` / `race_issue`;
   - `odds_capped`;
   - `profile_missing`;
   - `weight_exact`.
5. **Keeps missing values blank (NaN)**, never 0. Any imputation is left to modelling, fitted on training dates only.
6. **Runs hard checks** and stops with `CHECKS FAILED` if any fails:
   - a horse appearing twice in a race;
   - market probabilities not summing to 1;
   - a valid race without a winner;
   - values out of range.
7. **Writes the outputs** as Parquet and CSV, plus `reports/cleaning_report.md`.

---

## 5. Results

| | Kaggle 2017-20 | HKJC 2025-26 (+ 98 races of 2026/27) | Total |
|---|---|---|---|
| Rows (horse x race) | 27,008 | 12,202 | 39,210 |
| Races | 2,224 | 964 | 3,188 |
| Valid races | 2,216 | 963 | 3,179 |
| **Modelling rows** (`race_valid & started`) | 26,924 | 12,018 | **38,942** |

Nine races are flagged `race_valid = False`, with the reason in `race_issue`:

- 1 HKJC void race;
- 7 Kaggle races with missing rows or no winner;
- 1 Kaggle race with wrong placings.

---

## 6. Problems found and how they were handled

| Problem | Evidence | Handling |
|---|---|---|
| HKJC race 2025-11-15 ST R8 was declared void | HKJC results page | Flagged `void_race` |
| 7 Kaggle races have missing rows or no winner | gaps in the placings | Flagged `placing_gap` / `no_winner` |
| **Kaggle 2018-02-10 ST R7 has wrong placings.** The real winner (CONTE, odds 1.3, 1:34.85) is listed 5th. | HKJC results page | Flagged `placing_time_mismatch` |
| 10 Kaggle races where one horse's time is out of order with its placing. Some are real protests (2018-10-07 ST R7). Some are typos (2017-12-10 ST R4: 2:26.63 vs HKJC's 2:26.23; 2019-04-14 ST R5). | HKJC results pages | Kept and listed in the report. Times only matter for Week 3 speed features. |
| Kaggle odds are capped at 99 until 13 Mar 2019 (1,217 runners) | first odds above 99 appear on 13 Mar 2019 | `odds_capped` flag. Read "99" as "99 or longer". |
| Kaggle weights are rounded to whole kg | compared with HKJC pages | Converted to lb (within about 1 lb). `weight_exact = False`. |
| Kaggle `FGrating` and `Path` are not the HKJC rating or running positions. `FGrating` behaves like a post-race figure (0.77 rank correlation with the result). | data checks | Renamed `kaggle_fg_figure` / `kaggle_path`. Never features. |
| HKJC `Race time` holds the race's sectional times (the same for every horse) | file vs HKJC pages | Only the winner's time is used. Per-horse finish time is not available for HKJC. |
| HKJC `FGrating` is the **pre-race** official rating | it rises about 6.7 after a win and sits inside each race's rating band | Used as `official_rating` |
| **HKJC country/age/rating are blank for 515 of 1,506 horses (about 26% of rows)** | see below | Kept blank. `profile_missing` flag. **Never a feature; don't drop.** |
| Kaggle has no class or rating. HKJC has no handicap flag. Neither has going, course, LBW or declared weight. | both files | Documented in `data_dictionary.md` |

**The missing HKJC profile data is not random.**

- 28.5% of 2025/26 starters have no profile, against 2.9% in 2026/27.
- Those horses win 4.0% of the time vs 9.7% for the rest, at median odds of 22 vs 14.
- They are mostly horses that left Hong Kong racing before the scrape (Oct 2026), so a blank carries **future** information. Using it as a feature would leak; dropping the rows would bias the sample towards stronger horses.

---

## 7. How the output was checked

- **All hard checks pass:** `All checks passed`.
- **Same result everywhere.** My laptop's output (Windows, pandas 3.0.6) is identical, value for value, to test runs on Linux with pandas 3.0.6 and with pandas 2.2 on Python 3.12 (similar to SageMaker). The CSVs differ only in Windows line endings.
- **Checked against the official HKJC results pages:**

  | Race | What was checked |
  |---|---|
  | 2018-01-01 ST R1 | Kaggle draw, odds and finish times exact; weight within about 1 lb |
  | 2018-02-10 ST R7 | Found the wrong placings |
  | 2018-10-07 ST R7 | Protest confirmed |
  | 2017-12-10 ST R4 and 2019-04-14 ST R5 | Time typos confirmed |
  | 2025-11-15 ST R8 | Void confirmed |
  | 2026-03-08 ST R5 | All 14 runners match; protest confirmed |
  | 2026-10-11 ST R9 | All 12 runners match exactly: placing, draw, weight, jockey, trainer, odds, running positions, winning time 1:08.53 |

- **Ages** go up on 1 January in both sources and never go down for a horse, so the two sources use the same rule.
- **Market:** the sum of 1/odds is about 1.23 per race, which matches HKJC's 17.5% deduction. The favourite wins 31.3% (Kaggle) and 29.2% (HKJC) of races.
- **Market baseline** (per-race log loss, lower is better), computed from the cleaned data; the EDA notebook shows it:

  | | Market | Uniform (1/field size) |
  |---|---|---|
  | Kaggle | **1.987** | 2.486 |
  | HKJC | **2.035** | 2.516 |

---

## 8. Shared with the group (13 Oct)

| File | What it is |
|---|---|
| `runners_merged.parquet` (1.1 MB) | The cleaned dataset: 39,210 rows x 51 columns. For modelling (`pd.read_parquet`). |
| `runners_merged.csv` (12 MB) | The same table, for viewing in Excel |
| `data_dictionary.md` | Meaning, unit, source and timing (before the race / at the off / after) of every column |
| `cleaning_report.md` | Counts, flagged races, missing values, name links, market check |

Key points in the message:

- model only on `race_valid & started`;
- the target is `is_winner`, and the baseline to beat is `market_prob`;
- split train and test by date;
- don't use `profile_missing` or missingness as a feature;
- `odds_capped`;
- never use post-race columns as features.

---

## 9. AWS pipeline (done 14-15 Oct)

All in Asia Pacific (Singapore), `ap-southeast-1`, on my own AWS account. Every resource has the tag `Project = ELEC3544 DSFM Group Project`.

| Resource | Details |
|---|---|
| **S3 bucket** `inns-elec3544-dsfm-sagemaker` | Private, versioning on. Holds the 2 raw CSVs (`raw/`), the per-source clean tables (`interim/`), the merged table (`processed/`), the cleaning report and 5 EDA charts (`reports/`), and the script and notebook (`code/`). |
| **SageMaker AI domain** `QuickSetupDomain-20261014T163783` | Single-user Quick setup (IAM sign-in). Idle shutdown 120 min. |
| **JupyterLab space** `inns-elec3544-dsfm-datacleaning` | Private, `ml.t3.medium`, 5 GB, SageMaker Distribution 4.5.0 (Python 3.12.14, pandas 2.3.3, pyarrow 21.0.0). Stopped after use. |

**Reproduced in SageMaker (15 Oct).** The space downloaded the raw data and code from S3 and re-ran `clean_data.py`:

- `hkjc: 12,202 rows, 964 races (963 valid)`
- `kaggle: 27,008 rows, 2224 races (2216 valid)`
- `All checks passed`
- **`same table: True`**: identical to the laptop's output (Windows, Python 3.14.7, pandas 3.0.6). Screenshot taken.

**Added for Fahim (31 Oct, see `docs/FAHIM_ACCESS_RUNBOOK.md`):** Identity Center user `fahim.ahmed` + permission set `ELEC3544-DSFM-Fahim` (12 h sessions); SageMaker user profile `fahim` with role `inns-elec3544-dsfm-fahim-role` (project bucket only, own private spaces, `ml.t3.medium`-`ml.t3.xlarge`); lifecycle config `inns-elec3544-dsfm-fahim-sync` (copies project files into his space); budget `ELEC3544-DSFM-SageMaker-Monthly` (US$20). Remove around 30 Nov.

**Decisions:**

- **SageMaker AI (Studio + JupyterLab), not SageMaker Unified Studio.** Unified Studio's default role is much broader (`s3:*` on every bucket in the account, plus `sagemaker:*`, Redshift, Bedrock and RDS).
- **The space is private, not shared.** Only my user profile can open it. A separate shared space can be created later for collaboration.
- **No Data Wrangler / Canvas report** (not needed this week; US$1.90/h while logged in).

**Still to do (as of 31 Oct):**

- SageMaker-only budget: US$20/month, created with Fahim's access (`docs/FAHIM_ACCESS_RUNBOOK.md` Step 3).
- Remove the Unified Studio setup started by mistake on 13 Oct: domain `SageMakerUnifiedStudio-6nx6o6rfru3dsx-...` and project `admin-project-...`. It's unused.

---

## 10. EDA findings (15 Oct, `notebooks/02_eda.ipynb`)

All 11 code cells ran without errors on the laptop, and every number below was checked. The 5 charts are in `reports/figures/` and S3. This list was pasted into the notebook (cell 2) on 16 Oct; make sure that cell is Markdown, not code.

1. **Coverage.** 3,179 valid races and 38,942 starters.
   - Kaggle: 2,216 races, 2017/18-2019/20. No data after 16 May 2018 or after 21 Jun 2020.
   - HKJC: 963 races, all of 2025/26 plus 98 races of 2026/27.
2. **The baseline to beat** (per-race log loss, lower is better):

   | | Market | Guessing 1/field size | Favourite wins |
   |---|---|---|---|
   | Kaggle | **1.987** | 2.486 | 31.3% |
   | HKJC | **2.035** | 2.516 | 29.2% |

   The market is close to well calibrated in both periods.
3. **Kaggle odds are capped at 99 before 13 Mar 2019** (8.7% of 2017/18 and 5.4% of 2018/19 runners; `odds_capped`).
4. **HKJC blanks are not random.**
   - 28.5% of 2025/26 starters have no country, age or rating, against 2.9% in 2026/27.
   - They won 4.0% of the time vs 9.7%, at median odds of 22 vs 14.
   - So a blank carries future information: never use it as a feature, and don't drop the rows.
5. **The two periods look alike.** Median odds 15 vs 16; median field size 12, age 5, distance 1,400 m. Two differences:
   - HKJC fields are full (14 runners) more often: 37.1% vs 28.7% of races.
   - Kaggle has more 3-year-olds: 11.1% vs 2.9%.
6. **Draw bias at Happy Valley, steady across both periods.**
   - Draws 1-2 win 13.9-14.7% of the time, vs 3.9-5.2% for draws 10+. The no-bias rate is about 8.5%.
   - Weaker at Sha Tin: 8.9-11.8% vs 6.0-7.0%.
7. **Weight.** The win rate rises with carried weight: Kaggle 5.7% to 9.7%, HKJC 6.6% to 8.7%. The HKJC rating relative to the race shows only a weak, noisy rise.
8. **Winning times agree** within about 0.6 s at every common distance except 2400 m (only 3 HKJC races).
9. **People carry over.** 27 of 47 HKJC jockeys and 22 of 45 trainers also appear in 2017-20, covering 68% / 67% of 2025-26 rides.

---

## 11. Next

| When | What |
|---|---|
| ~~Fri 16 Oct~~ | ~~Findings, PDF, hand-over message~~ - done |
| **Tonight (31 Oct)** | Fahim's AWS access (`docs/FAHIM_ACCESS_RUNBOOK.md`) |
| **By Wed 11 Nov** | Week 3 features (Nayeon), Week 4 models in SageMaker (Fahim) |
| **~12-14 Nov** | Week 5: score every model once on the test window (HKJC Apr-Oct 2026) with `eval_utils.py` |
| **Tue 17 Nov 11:59 PM** | Report due. Use the real data dates (section 13, "Proposal vs data") |
| **Wed 18 / 25 Nov** | Presentation (about 10 min) |
| **TBC** | Checkpoint PDF (individual) |
| **Any time** | Remove the accidental Unified Studio setup; optional GitHub upload (`.gitignore` included) |
| **~Mon 30 Nov** | Remove Fahim's access and his space (reminder set) |

Step-by-step instructions are in `docs/WEEK1-2_RUNBOOK.md`.

---

## 12. Hand-over message (sent Fri 16 Oct)

> Hi both, my Weeks 1-2 part (data cleaning + AWS) is done.
>
> **AWS (my account, Singapore):**
> - S3 bucket `inns-elec3544-dsfm-sagemaker` holds the raw data, the cleaned tables (`processed/runners_merged.parquet`), the cleaning report, the EDA charts, and the code.
> - SageMaker Studio: I re-ran the cleaning in a SageMaker JupyterLab space and got exactly the same table as on my laptop (screenshot attached). So the pipeline is reproducible on AWS.
> - For now I'll keep sharing files directly. If you need SageMaker access for modelling, tell me and I'll set up a login.
>
> **EDA (report attached), key points for modelling:**
> 1. Baseline to beat: market per-race log loss 1.987 (2017-20) and 2.035 (2025-26). The favourite wins about 30%.
> 2. Strong draw bias at Happy Valley (draws 1-2 win about 14-15% vs about 4-5% for 10+), so draw is a useful feature. It's weaker at Sha Tin.
> 3. Win rate rises with carried weight. The 2017-20 and 2025-26 data look alike, so training on both should be fine.
> 4. HKJC country/age/rating blanks are mostly horses that left HK before the scrape. Don't use "missing" as a feature, and don't drop those rows.
> 5. Kaggle odds before 13 Mar 2019 are capped at 99 (`odds_capped`).
> 6. About two-thirds of 2025-26 rides are by jockeys/trainers who also appear in 2017-20, so history features will cover most rides.
>
> Model on `race_valid & started`, target `is_winner`, and split by date. Column details are in `data_dictionary.md`.

---

## 13. Weeks 3-4: features and models (status 31 Oct)

**Files:** `docs/WEEK3-4_PLAN.md` (why and what, 16 Oct), `docs/WEEK3-4_RUNBOOK.md` (how, in SageMaker, 28 Oct), `week34_starter.py` (the pipeline, 28 Oct, fixed 31 Oct).

**The models are classical**, as the proposal says: odds baselines (market, power, Shin, market-only refit), conditional logit (statsmodels), LightGBM, Stage 2 conditional logit on log(model p) + log(market p). Claude (a foundation model) is used for ideas, code and checking.

**Splits (by date, fixed):**

| Window | Data | Races |
|---|---|---|
| Stage 1 train / tune | Kaggle Nov 2017 - Jul 2019 / 2019-20 | 1,307 / 756 |
| Stage 1 final fit | Kaggle Nov 2017 - Jun 2020 | 2,063 |
| Stage 2 fit | HKJC Nov 2025 - Mar 2026 | 423 |
| Test (scored once, Week 5) | HKJC Apr - Oct 2026 | 400 |

**Starter results (19 features; same package versions as SageMaker):**

- Kaggle tuning window (2019/20), per-race log loss: logit 2.1804, LightGBM 2.1837, market 1.9947.
- Stage 2 window: market 2.0384, guessing 2.5173, logit 2.3154, LightGBM 2.3167.
- Stage 2 α = -0.35 ± 0.11 (logit), -0.40 ± 0.12 (LightGBM); β ≈ 1.24. Market-only refit β = 1.04. The simple models run against the market, so **Week 3 effort goes into better features**.
- Test window: predictions saved, not scored.

**31 Oct checks and fix:**

- Re-run reproduced the 28 Oct reference exactly before the fix.
- Leak checks pass: changing a race's result changes only later race days' features; jockey/trainer values are the same for all rides on one day.
- Fixed: `rc_weight_rel` and `rc_weight_x_dist` were 0.992 correlated (both SEs ±0.12). Distance is now centred at 1,400 m (correlation -0.02); logit predictions unchanged, LightGBM 2.1790 → 2.1837.
- Finding for the report: the centred weight × distance term is -0.006 ± 0.023, so the logit shows no sign that weight matters more over long distances.

**Proposal vs data** (fix in the final report):

- Scrape period: the proposal says Jan 2025 - Sep 2026; the data is **7 Sep 2025 - 11 Oct 2026**.
- Gap between datasets: the proposal says four years; it's **Jun 2020 - Sep 2025, about five years**.

**Who does what (agreed 31 Oct - plan section 4):**

| Item | Owner | Plan section |
|---|---|---|
| Fahim's AWS access: own user profile `fahim`, private space, scoped role (project bucket only, `ml.t3.medium`-`xlarge`), files copied in automatically, US$20 budget | Inns (31 Oct) | `FAHIM_ACCESS_RUNBOOK.md` |
| Better features (distance/track record, jockey-trainer pair, running style, rating/age fills) + feature checks 5.5 #1-8 + `feature_dictionary.md` | Nayeon | 5.3-5.5 |
| Baselines, logit, LightGBM, Stage 2, Track B, predictions + models in S3 | Fahim (guide `WEEK4_FAHIM.md`) | 6.3-6.8 |
| `eval_utils.py` (log loss, Brier, top-1, pseudo-R², calibration, paired bootstrap) | not agreed yet | 6.2 |
