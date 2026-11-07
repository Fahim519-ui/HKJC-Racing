# Weeks 3-4 plan: features and model training (Group 17)

> Restored to `docs/` on 7 Nov 2026 from `update_2026-10-31_fahim_cli/docs/`. Plan as of 31 Oct;
> bucket and SageMaker space names replaced by placeholders because the repository is public.

**Written:** Fri 16 Oct 2026. **Updated:** Sat 31 Oct 2026.

> **Status 31 Oct (evening):** owners agreed - **Week 3 features: Nayeon** (on her laptop; no AWS needed), **Week 4 models: Fahim** in his own SageMaker space (access: `docs/FAHIM_ACCESS_RUNBOOK.md`), **Inns:** data, AWS, access and costs. The real deadlines replace the week dates below: **Weeks 3-4 done by Wed 11 Nov** (group target), report **Tue 17 Nov 11:59 PM**, presentations **18 / 25 Nov** (about 10 min). The starter pipeline (`week34_starter.py`, fixed 31 Oct) does a first version of Weeks 3-4 in one run. Still to write: better features + checks (5.3-5.5, Nayeon), `eval_utils.py` (6.2), the tuned models and Track B (6.4-6.7, Fahim). Changes on 31 Oct: `rc_weight_x_dist` centred (5.3), starter results (7.1), proposal mismatches (8).

**Covers:**

- Week 3: Sat 17 - Fri 23 Oct
- Week 4: Sat 24 - Fri 30 Oct

These dates assume each week ends on a Friday, like Weeks 1-2.

**Based on:**

- the proposal, `DataScience_Project-v4.pdf`;
- the cleaned data, `data/processed/runners_merged.parquet`;
- checks run on that data on 16 Oct;
- the sources listed at the end.

---

## The short version

**Week 3: features.**

- Build one feature table, `data/features/features_v1.parquet`.
- Use history from earlier race days only.
- Write the checks the proposal asks for.

**Week 4: models.** Train these in SageMaker:

- the odds baselines;
- the conditional logit;
- LightGBM;
- the Stage 2 model.

Then save every model's predictions to one file. **The test races are scored once, in Week 5.**

**Main design.** The proposal trains on 2017-20 and tests on the newest races. That works, but only with features that exist in both datasets.

- Rating, class and running positions exist only in the HKJC data.
- So they get a second, HKJC-only experiment ("Track B").
- Track B answers sub-question 1: which feature groups matter.

**The splits are fixed now (section 3):**

| Stage | Data | Races |
|---|---|---|
| Stage 1 train | Kaggle Nov 2017 - Jun 2020 | 2,063 |
| Stage 2 fit | HKJC Nov 2025 - Mar 2026 | 423 |
| Test | HKJC Apr - Oct 2026 | 400 |

**Missing HKJC rating and age can be filled without leaking the future:**

- **Rating:** fill from the carried weight in the same race. This is within 2 points 93% of the time.
- **Age:** fill from the year in `horse_id`. This is within 1 year 99.9% of the time.

**Expect small differences.**

- Today's market already has a pseudo-R² of 0.19-0.20.
- The public had 0.12 in Benter's 1986-93 Hong Kong data.
- So use bootstrap confidence intervals, and report "no clear difference" if that's the result.

**A toy run (section 7) checked the tools.**

- Both models run in seconds.
- A weak model is far behind the market.
- A weak model's Stage 2 weight can even come out negative.

---

## 1. What the proposal asks for in Weeks 3-4

| Proposal | Where in this plan |
|---|---|
| Wk 3: running-style metrics from HKJC running positions | 5.3, group `rs_` |
| Wk 3: form, jockey and trainer features from past races only | 5.3, groups `hf_` and `jt_` |
| Wk 3: correlation and target-leakage checks | 5.5 |
| Wk 4: odds baseline (1/odds normalised); Shin and power as a sensitivity check | 6.3 |
| Wk 4: conditional logit and LightGBM, trained in SageMaker | 6.4, 6.5 |
| Wk 4: Stage 2 conditional logit on log(model p) and log(market p) | 6.6 |
| Metrics: per-race log loss (main), Brier, calibration, top-1, bootstrap CIs | 6.2: written in Wk 4, used in Wk 5 |
| Sub-question 1: which feature groups help most | groups 5.3 + Track B 6.7, compared in Wk 5 |
| Sub-question 2: does LightGBM add more beyond the odds than the logit? | Stage 2 weights for each, 6.6 |

---

## 2. Seven facts from our data that shape the plan

All counts are on modelling rows (`race_valid & started`).

**1. Some columns exist in only one dataset.**

| Column | HKJC 2025-26 | Kaggle 2017-20 |
|---|---|---|
| `official_rating` | 70% filled (2025/26), 97% (2026/27) | none |
| `race_class` | 96-98% | none: 2,141 of 2,216 races are just "Handicap" |
| `running_positions` | 100% | none |
| `finish_time_s` (each horse's own time) | none | 100% |
| `kaggle_fg_figure` | none | 100%, but it's post-race |
| draw, weight, distance, prize, jockey, trainer, odds | 100% | 100% |

- **Track A (main):** trained on Kaggle, so it uses only the shared columns.
- **Track B:** adds rating and running style, using HKJC only.
- Speed figures from finish times can't be used: HKJC has no per-horse times.

**2. Prize money stands in for class in both datasets.** Each class has a fixed prize within a season.

| Class | 2025/26 prize (HK$) |
|---|---|
| Class 5 | 875k |
| Class 4 | 1.17m |
| Class 3 | 1.86-2.05m |
| Class 2 | 2.84-3.12m |

Kaggle has the same kind of fixed tiers, e.g. 2019/20: 725k, 967k, 1.45m, 2.1m. So:

- use `log(prize / season median prize)` as the class level;
- use `log(prize / prize of the horse's last race)` as "up or down in class".

Both are comparable across the 5-year gap.

**3. Horses can't be linked across datasets, and history needs a warm-up.** Share of runners with at least one earlier start in the data:

| Month | HKJC | Kaggle |
|---|---|---|
| 1st month (Sep) | 18% | 32% |
| 2nd month (Oct) | 71% | 77% |
| 3rd month (Nov) | 87% | 87% |
| Later | 90-98% | 89-97% |

- The first two months of each dataset are **history only**: used to build features, not modelled or scored.
- Use counts over a window ("starts in the last 365 days", "last 5 starts"), not career totals. The data starts in the middle of every horse's career.

**4. People link across datasets, but five years apart.**

- 27 of 47 HKJC jockeys and 22 of 45 trainer
s also appear in 2017-20.
- Their stats from 2017-20 are stale, so compute jockey and trainer stats within each dataset.






**5. The HKJC blanks can be filled without leaking the future.** The blanks mark horses that left HK, so the gap itself must never become a feature.

- **Rating from weight.** In Class races, rating minus carried weight is almost constant within a race.
  - Its spread within a race (median SD) is 0.9 lb, against 6.5 in Group, Griffin and 4YO races, which aren't weighted by rating.







  - Estimating a horse's rating from the other runners in its race: average error 0.66 points, within 2 points for 93%.
  - This covers 95% of the missing ratings.
  - The other 160 rows, in Group, Griffin and 4YO races, get the field average.
- **Age from `horse_id`.** `HK_2019_D090` holds the year the horse was registered.
  - The formula is: age = race year - that year + 3.
  - This is within 1 year for 99.9% of the known ages, and exact for 66%.
  - For Track B, `years in HK` (race year - id year) has no blanks at all, so it can replace age.
-
 **No missing flags,** and round the filled rating to a whole number, like the real ones. That way filled rows don't stand out.




**6. Race-level columns cancel out in a conditional logit.**

- Distance, class, field size and surface are the same for every horse in a race.
- So they only work in an interaction. An example is the proposal's own idea: weight × distance.
- LightGBM can use them directly. But because it scores one horse at a time, it also needs within-race relative versions (value minus race average).

**7. The market is a tough benchmark.**

Per-race log loss on the Stage 2 window:

| | Log loss |
|---|---|
| Market | 2.038 |
| Guessing | 2.517 |

That's a pseudo-R² of 0.19, where pseudo-R² = 1 - model / guessing.

- **Benter's era:** his public estimate was 0.1218, and his combined model added 0.0178.
- **A later study:** a conditional logit beat the public by only 0.0036.
- **Takeout:** the win pool averages 1.227 in 1/odds per race, about 18.5% taken out.
- **Odds of 1.0:** 12 runners have odds of 1.0, and all of them won. Clip odds at 1.01 before the power and Shin methods (6.3).
- **Kaggle's 99 cap:** it doesn't affect the test, which uses HKJC only.

---

## 3. Splits (fixed now, before any modelling)

| Window | Data | Races / rows | Used for |
|---|---|---|---|
| Warm-up | Kaggle Sep-Oct 2017; HKJC Sep-Oct 2025 | 153 / 140 races | building history only |
| Stage 1 train | Kaggle Nov 2017 - Jul 2019 | 1,307 / 15,836 | fitting while choosing features and settings |
| Stage 1 tuning | Kaggle 2019/20 (Sep 2019 - Jun 2020) | 756 / 9,299 | choosing features and settings, early stopping |
| Stage 1 final fit | Kaggle Nov 2017 - Jun 2020 | 2,063 / 25,135 | refit once the choices are made |
| **Stage 2 fit** | HKJC Nov 2025 - Mar 2026 | 423 / 5,287 | fitting α and β (6.6), the market-only refit, Track B training |
| **Test** | HKJC Apr - Oct 2026 (includes 98 races of 2026/27) | 400 / 4,978 | **Week 5 only, scored once** |

**Rules:**

- Split by date only. Never split randomly by row.
- Keep a `split` column in the feature table: `warmup`, `k_train`, `k_tune`, `h_s2`, `h_test`.
- Predictions on the test window can be saved in Week 4, but nobody computes test metrics until Week 5.
- If anything is changed after looking at the test, say so in the report.

---

## 4. Who does what (agreed 31 Oct)

| | Week 3 (features) | Week 4 (models) | Target |
|---|---|---|---|
| **Nayeon** | Feature table `features_v1.parquet` (5.1-5.4, incl. running style `rs_` and the rating/age fills), `feature_dictionary.md`, checks 5.5 → `feature_checks.md`. Laptop is fine; send the files to Inns or Fahim to upload to `s3://<project-bucket>/features/` | - | as early as possible before Wed 11 Nov, so Fahim can use them |
| **Fahim** | - | Baselines, conditional logit, LightGBM, Stage 2, Track B (6.3-6.7); `predictions_v1.parquet`, `models/`, `week4_models.md` in S3 (6.8). In his own SageMaker space, guide `docs/WEEK4_FAHIM.md` | Wed 11 Nov |
| **Inns** | Data questions; upload Nayeon's features if needed | Fahim's AWS access (31 Oct), costs/budget; `eval_utils.py` (6.2) for Week 5 (suggested - not agreed yet) | access tonight; `eval_utils.py` before Week 5 |
| **All** | | | Week 5 scoring ~12-14 Nov; report Tue 17 Nov; presentation 18/25 Nov |

The plan's Week 3 and Week 4 dates (17-30 Oct) are out of date; use the targets above.

---

## 5. Week 3: features

### 5.1 Output

| File | What |
|---|---|
| `build_features.py` | Reads `data/processed/runners_merged.parquet` and writes the features. It stops with "CHECKS FAILED", like `clean_data.py`. |
| `data/features/features_v1.parquet` | One row per modelling runner (38,942). Keys + `split` + `is_winner` + `market_prob` (for Stage 2 only) + feature columns. |
| `docs/feature_dictionary.md` | Every feature: group, definition, Track A/B, how blanks are handled |
| `reports/feature_checks.md` | Results of the checks in 5.5 |

### 5.2 Rules

1. **History = earlier race days, same dataset.**
   - A horse runs at most once a day, so `groupby(horse).shift(1)` is safe.
   - Jockeys and trainers ride many races a day, so use totals from earlier days only (snippet below).
2. **Build history only from rows that are valid and started.** Leave out void races, withdrawn horses and flagged races.
3. **Never a feature:**
   - anything from the same race's result: `finish_*`, `is_winner`, `running_positions`, `race_winner_time_s`, `kaggle_fg_figure`;
   - market columns, in Stage 1;
   - `profile_missing`, or whether a value is blank;
   - `kaggle_path`.
4. **Windows, not careers:** last 365 days, last 3 or 5 starts.
5. **Smooth small samples toward the average** (empirical Bayes): `(wins + a × prior) / (rides + a)`.
   - Prior: 1/12 for win rate, 3/12 for top-3 rate.
   - Weight `a`: about 30 rides for jockeys and trainers, 5 runs for horses.
6. **Prefix every feature with its group** (`rc_`, `hf_`, `jt_`, `rs_`, `rt_`), so the group comparison in Week 5 is one filter.

Jockey win rate from earlier days only (tested on our data):

- every jockey's first day gets exactly the prior;
- all of a jockey's rides on the same day get the same value.

```python
daily = (m.groupby(["source", "jockey_key", "race_date"], as_index=False)
           .agg(wins=("is_winner", "sum"), rides=("is_winner", "size"))
           .sort_values(["source", "jockey_key", "race_date"]))
roll = (daily.set_index("race_date")
             .groupby(["source", "jockey_key"])[["wins", "rides"]]
             .rolling("365D", closed="left").sum()        # closed="left" = before today
             .fillna(0).reset_index())
roll["jt_jockey_win365"] = (roll["wins"] + 30 / 12) / (roll["rides"] + 30)
m = m.merge(roll[["source", "jockey_key", "race_date", "jt_jockey_win365"]],
            on=["source", "jockey_key", "race_date"], how="left")
```

### 5.3 Feature list

Columns marked "A+B" are used in both tracks; "B" means HKJC only (Track B). "rel" means value minus the race average.

**`rc_` race conditions (today's race, known at declarations):**

| Feature | Definition | Track |
|---|---|---|
| `rc_draw_pct` | (draw - 1) / (field size - 1) | A+B |
| `rc_draw_x_hv` | `rc_draw_pct` × Happy Valley. The EDA found a strong HV draw bias in both periods. | A+B |
| `rc_draw_x_sprint` | `rc_draw_pct` × (distance ≤ 1,200 m) | A+B |
| `rc_weight_rel` | carried weight minus race average | A+B |
| `rc_weight_x_dist` | `rc_weight_rel` × (distance − 1,400) in km (the proposal's example). **Centred** (31 Oct): uncentred it was 0.99 correlated with `rc_weight_rel` | A+B |
| `rc_age`, `rc_age_rel` | age (HKJC blanks filled, 5.4) | A+B |
| `rc_field_size`, `rc_distance_m`, `rc_track_hv`, `rc_surface_aw` | race-level (LightGBM only) | A+B |
| `rc_prize_level` | log(prize / season median prize), i.e. the class level (fact 2) | A+B |

**`hf_` horse form (earlier starts, same dataset):**

| Feature | Definition | Track |
|---|---|---|
| `hf_days_since` | days since last run, log, capped at 365 | A+B |
| `hf_first_start` | no earlier start in the data (after warm-up = a real debut or return) | A+B |
| `hf_starts_365` | starts in the last 365 days | A+B |
| `hf_last_fin_pct`, `hf_avg3_fin_pct`, `hf_best5_fin_pct` | finish percentile: (place - 1) / (field - 1), so 0 = won | A+B, + rel |
| `hf_win5_s`, `hf_top3_5_s` | smoothed win and top-3 rate, last 5 starts | A+B |
| `hf_dist_top3_s`, `hf_track_top3_s` | smoothed top-3 rate at this distance (±200 m) and this track | A+B |
| `hf_weight_change`, `hf_class_change`, `hf_dist_change` | vs last start (class change = log prize ratio) | A+B |

**`jt_` jockey and trainer (earlier days, same dataset):**

| Feature | Definition | Track |
|---|---|---|
| `jt_jockey_win365`, `jt_jockey_top3_365` | smoothed rates, last 365 days | A+B, + rel |
| `jt_trainer_win365`, `jt_trainer_top3_365` | same for trainers | A+B, + rel |
| `jt_combo_win365` | jockey + trainer pair, smoothed with weight 50 | A+B |
| `jt_jockey_rides365` | rides in the last 365 days (log): busy jockeys get better rides | A+B |

**`rs_` running style (HKJC only, from earlier races' `running_positions`):**

`running_positions` has 3-6 calls depending on distance (3 at 1,000-1,200 m, up to 6 at 2,200-2,400 m), and the last number is the finish.

| Feature | Definition | Track |
|---|---|---|
| `rs_early_avg3` | first call / field size, average of the last 3 starts | B |
| `rs_mid_avg3` | second call / field size, same | B |
| `rs_gain_avg3` | (first call - finish) / field size, i.e. the late run | B |
| `rs_lead_share5` | share of the last 5 starts in the first 2 at the first call | B |
| `rs_pace_others` | number of *other* runners today with `rs_early_avg3` ≤ 0.25 (pace pressure) | B |
| `rs_early_x_pace`, `rs_early_x_draw_hv` | interactions for the logit | B |

**`rt_` rating (HKJC only):**

| Feature | Definition | Track |
|---|---|---|
| `rt_rating`, `rt_rating_rel` | official rating, with blanks filled from weight (5.4), rounded | B |
| `rt_rating_change` | vs the rating at the last start (both pre-race) | B |
| `rt_years_in_hk` | race year minus the year in `horse_id` (no blanks) | B |

### 5.4 Filling HKJC rating and age (in `build_features.py`)

**Rating:**

1. For Class 1-5 races, for each runner with a known rating, compute `rating - carried_weight_lb`.
2. A runner with a blank gets the race median of that value + its own weight, rounded.
3. Any remaining blank (Group, Griffin and 4YO races) gets `rt_rating_rel = 0`, the field average.

**Age:**

1. Blank age = race year - `horse_id` year + offset.
2. Learn the offset (it's 3) from known ages in the HKJC warm-up and Stage 2 windows only, i.e. the training dates (data rule).

**Check:** after filling, the features' blank rate is the same (0%) for `profile_missing` rows and other rows.

### 5.5 Checks (proposal: "correlation and target-leakage checks")

| # | Check | Pass when |
|---|---|---|
| 1 | Recompute 5 features by hand for 10 random runners | all match |
| 2 | **Future test:** change one race's result in a copy, rebuild | only features of *later* race days change |
| 3 | Same-day test | jockey/trainer features equal for all of one jockey's rides that day |
| 4 | Single-feature strength: a one-feature logit per feature, on `k_train` | no single feature beats the market; anything close is a leak until explained |
| 5 | Correlation (Spearman) between features | pairs above 0.9: keep one in the logit |
| 6 | Blank pattern vs `profile_missing` (HKJC) | no difference after filling |
| 7 | Era drift: standardised mean difference, `k_train` vs `h_s2` | list features above 0.25; they may not carry over from 2017-20 |
| 8 | Counts | 38,942 rows, keys unique, every row has a `split` |

### 5.6 Week 3 is done when

- [ ] `features_v1.parquet`, `feature_dictionary.md` and `feature_checks.md` exist, and all checks pass;
- [ ] they're uploaded to `s3://<project-bucket>/features/`;
- [ ] the group has agreed the splits (section 3) and the feature list.

Upload (laptop, venv active, signed in with `aws sso login --profile cloudpc`):

```powershell
$B = "<project-bucket>"
$R = "--region", "ap-southeast-1", "--profile", "cloudpc"
aws s3 cp data\features "s3://$B/features/" --recursive @R
aws s3 cp build_features.py "s3://$B/code/" @R
aws s3 cp docs\feature_dictionary.md "s3://$B/code/docs/" @R
aws s3 cp reports\feature_checks.md "s3://$B/reports/" @R
```

---

## 6. Week 4: models

### 6.1 Where to run

**SageMaker.** Use the space `<data-cleaning space>` (ml.t3.medium).

- Its image (SageMaker Distribution 4.5.0) already has lightgbm 4.6.0, statsmodels 0.15.0 and scikit-learn 1.7.2, so nothing needs installing.
- Take the final numbers from SageMaker, so everyone's numbers match.
- Stop the space after each session (idle shutdown is 120 min).

**Laptop, for drafts:**

```powershell
python -m pip install lightgbm==4.6.0 statsmodels==0.15.0 scikit-learn
```

- Windows wheels for Python 3.14 exist for all three (checked).
- Use `python -m pip`, because `pip.exe` is blocked by Application Control.

**Fahim:**

- Either give him access through IAM Identity Center (`WEEK1-2_RUNBOOK.md`, Fahim section);
- or he drafts on his own machine with `features_v1.parquet`, and the final run happens in SageMaker.

### 6.2 Shared evaluation helper first: `eval_utils.py`

Every model is scored by the same code on the same races.

| Function | What |
|---|---|
| `normalise(p, race_id)` | make probabilities sum to 1 per race |
| `race_softmax(score, race_id)` | turn logit scores into probabilities per race |
| `per_race_log_loss` | mean over races of -log p(winner). Dead heats: average over the winners. **Main metric.** |
| `brier` | per race, sum of (p - won)², averaged over races |
| `top1` | share of races where the highest-p horse won |
| `pseudo_r2` | 1 - log loss / log loss of guessing 1/field size (Benter's measure) |
| `calibration_table` | bins of p (0-2, 2-5, 5-10, 10-20, 20-35, 35%+): mean p vs actual win rate |
| `paired_bootstrap` | resample *races* 2,000 times (fixed seed): 95% CI of (model log loss - market log loss). "Better" only if the whole CI is below 0. |

### 6.3 Odds baselines (proposal: the benchmark)

| Baseline | How |
|---|---|
| Guessing | 1 / field size |
| **Market (main)** | `market_prob` = 1/odds normalised per race (already in the data) |
| Power | p = (1/odds)^k, with k solved per race so the race sums to 1 |
| Shin | Shin's method per race |
| Market-only refit | p ∝ market^β, with β fitted on `h_s2`. This is the fair comparison for Stage 2 (6.6). |

**Note:** our odds come from the tote (a betting pool). So 1/odds normalised is each horse's share of the win pool. Power and Shin therefore test the favourite-longshot bias rather than "remove a bookmaker's margin".

Tested code:

- it matches the `shin` package to 2×10⁻¹⁶ on all 963 HKJC races;
- every race sums to 1.

We use our own code because `shin` has no wheel for Windows Python 3.14.

```python
import numpy as np
from scipy.optimize import brentq

def power_probs(odds):
    q = 1.0 / np.clip(np.asarray(odds, dtype=float), 1.01, None)   # 12 runners have odds 1.0
    k = brentq(lambda k: (q ** k).sum() - 1.0, 0.1, 10.0)
    return q ** k

def shin_probs(odds, tol=1e-12, max_iter=1000):
    q = 1.0 / np.clip(np.asarray(odds, dtype=float), 1.01, None)
    n, B = len(q), q.sum()
    z = 0.0
    for _ in range(max_iter):
        z_new = (np.sqrt(z**2 + 4 * (1 - z) * q**2 / B).sum() - 2) / (n - 2)
        if abs(z_new - z) < tol:
            z = z_new
            break
        z = z_new
    return (np.sqrt(z**2 + 4 * (1 - z) * q**2 / B) - z) / (2 * (1 - z))
```

### 6.4 Stage 1a: conditional logit (Benter-style)

1. **Standardise** each feature with the `k_train` mean and SD. Then fill any remaining blank with 0, i.e. the training mean.
   - `hf_first_start` carries the "no history" information.
   - Never add blank flags for rating or age.
2. **Choose the features.**
   - Start from about 6 strong ones: jockey rate, `hf_avg3_fin_pct_rel`, `rc_draw_pct`, `rc_draw_x_hv`, `rc_weight_rel`, `hf_days_since`.
   - Add a feature only if the per-race log loss on `k_tune` improves by at least 0.001.
   - Stop at about 15 features.
3. **Fit.** Use statsmodels `ConditionalLogit`.
   - Don't include an intercept: it cancels within a race.
   - Convert to plain floats first. With the pandas nullable types in our parquet, statsmodels fails with `TypeError: ufunc 'isfinite' not supported`.

   ```python
   from statsmodels.discrete.conditional_models import ConditionalLogit
   X = d[feats].astype(float).to_numpy()
   res = ConditionalLogit(d["is_winner"].astype(float).to_numpy(), X,
                          groups=d["race_id"].to_numpy()).fit(disp=0)
   p = race_softmax(X @ res.params, d["race_id"])      # statsmodels has no per-race predict
   ```

4. **Check the coefficients.** Each sign should make sense: more weight is good, an outside draw at HV is bad. Save the coefficient table with its SEs for the report.
5. **Refit** on the Stage 1 final fit window, then predict every HKJC window.
6. **If coefficients blow up** (features that move together), drop one of the pair. statsmodels' `fit_regularized` for this model is poorly documented, so avoid it.

### 6.5 Stage 1b: LightGBM (binary, then normalised per race)

**Setup:**

- `LGBMClassifier(objective="binary")` on `is_winner`.
- Use all Track A features: race-level + within-race relative.
- Normalise the output per race.

**Settings to search (20 combinations at most), scored by per-race log loss on `k_tune`:**

- `learning_rate` 0.02-0.05
- `num_leaves` 7-31
- `min_child_samples` 50-200
- `subsample=0.8` (with `subsample_freq=1`), `colsample_bytree=0.8`
- `reg_lambda` 1-10
- early stopping on `k_tune`

**Reproducible runs:** `deterministic=True`, `force_row_wise=True`, `random_state=17`.

**Refit** on the Stage 1 final fit window with the chosen number of trees.

**Save for Week 5:**

- gain importance;
- per-feature contributions from `model.predict(X, pred_contrib=True)` (SHAP values, no extra package).

### 6.6 Stage 2: model + market (the proposal's key experiment)

1. **Fit on `h_s2`** (423 races): `ConditionalLogit(y, [log p_stage1, log p_market])`, which gives **α** (the model's weight) and **β** (the market's weight), with SEs.
2. **Fit the market-only refit** on the same races: `ConditionalLogit(y, [log p_market])`, which gives β alone.
   - Stage 2 has to beat *this*, not just the raw market.
   - Otherwise a gain could just be the favourite-longshot correction.
3. **Do this twice:** with the logit's p and with LightGBM's p. That answers **sub-question 2**: which model has the larger α and the larger gain?
4. **Reading α:**
   - α > 0, more than 2 SE from 0: the model adds information beyond the odds.
   - α ≈ 0: nothing beyond the odds.
   - α < 0: the model's view runs against the market's. It happened in the toy run (section 7).
   - With 423 races, don't over-read any of these.
5. **Predict the test window and save,** but don't score it until Week 5.

### 6.7 Track B: HKJC only (for sub-question 1)

**Training:**

- Train a small logit (≤ 10 features) and a shallow LightGBM on `h_s2` (Nov 2025 - Mar 2026).
- Tune on Nov-Jan → Feb-Mar, then refit on Nov-Mar.
- 423 races is under Benter's "500 to 1,000 races" guide, so keep the models small.

**Four feature sets, all scored on the same test races in Week 5:**

1. base: the Track A features;
2. base + `rt_` (rating);
3. base + `rs_` (running style);
4. all.

**The Track A groups** (form, jockey/trainer, race conditions) are compared by dropping one group at a time from the Kaggle-trained model.

**Stage 2 for Track B is optional.** It needs walk-forward predictions, because Track B is trained on the Stage 2 window.

### 6.8 Save and upload

**Predictions:** `data/predictions/predictions_v1.parquet`, with one row per HKJC runner (Stage 2 + test windows). Columns:

- keys: `race_id`, `horse_id`, `split`, `is_winner`;
- baselines: `p_uniform`, `p_market`, `p_power`, `p_shin`, `p_market_refit`;
- Stage 1 and 2: `p_cl`, `p_lgbm`, `p_s2_cl`, `p_s2_lgbm`;
- Track B: `p_b_base`, `p_b_rating`, `p_b_running`, `p_b_all`.

**Models and notes:**

- `models/`: the fitted logit coefficients (CSV) and the LightGBM model (`booster_.save_model("models/lgbm_v1.txt")`);
- `reports/week4_models.md`: chosen features, settings, coefficient table, α/β table.

**Upload,** in a SageMaker terminal:

```bash
B=<project-bucket>
aws s3 cp data/predictions "s3://$B/predictions/" --recursive
aws s3 cp models "s3://$B/models/" --recursive
aws s3 cp reports/week4_models.md "s3://$B/reports/"
```

### 6.9 Week 4 is done when

- [ ] the baselines, logit, LightGBM, Stage 2 (both) and Track B are fitted in SageMaker;
- [ ] `predictions_v1.parquet` + `models/` + `week4_models.md` are in S3;
- [ ] nobody has computed metrics on the test window yet;
- [ ] Week 5 can start straight away: one notebook reads `predictions_v1.parquet` and calls `eval_utils.py`.

---

## 7. What a toy run showed (16 Oct, feasibility only)

**Setup:**

- 8 simple features: draw, weight, jockey and trainer 365-day rates, last and average-of-3 finish, days since, number of starts.
- Trained on Kaggle Nov 2017 - Jun 2020.
- Run in a Python 3.12 environment with the same package versions as SageMaker.

**Speed:**

- ConditionalLogit fitted 2,063 races (25,135 rows) in about 21 s.
- LightGBM took under 1 s.
- An ml.t3.medium is plenty.

**Stage 2 window (HKJC Nov-Mar, 423 races), per-race log loss:**

| Model | Log loss |
|---|---|
| Market | 2.038 |
| Toy logit | 2.321 |
| Toy LightGBM | 2.344 |
| Guessing | 2.517 |

**Stage 2 weights:**

| Stage 1 input | α | β |
|---|---|---|
| Logit | -0.37 ± 0.11 | 1.25 |
| LightGBM | -0.27 ± 0.09 | 1.21 |
| Market-only refit | - | 1.04 |

So the toy model's opinions ran *against* the market. The crowd already uses these simple form facts, and may over-use them. The real feature set has to carry information the market under-uses.

**Pitfall found:** statsmodels fails on pandas nullable types. Convert with `.astype(float).to_numpy()` (6.4).

The toy also ran on the test window, only to check the code. Those numbers aren't used for any decision.

### 7.1 Starter pipeline (`week34_starter.py`, 28 Oct; re-checked and fixed 31 Oct)

19 features (12 in the logit), same splits as section 3, same package versions as SageMaker. Runs in about 25 s.

| | Per-race log loss |
|---|---|
| Kaggle tuning window 2019/20: logit / LightGBM / market | 2.1804 / 2.1837 / 1.9947 |
| Stage 2 window: logit / LightGBM / market / guessing | 2.3154 / 2.3167 / 2.0384 / 2.5173 |

| Stage 2 (on `h_s2`) | α | β |
|---|---|---|
| Logit | -0.35 ± 0.11 | 1.24 |
| LightGBM | -0.40 ± 0.12 | 1.24 |
| Market-only refit | - | 1.04 ± 0.06 |

**31 Oct checks:**

- **Leaks (5.5 #2, #3): pass.** Changing one race's result (one HKJC and one Kaggle race) changed no feature on or before that race day, only later ones. Jockey/trainer values are equal across all of a person's rides on one day.
- **Correlation (5.5 #5): one failure, fixed.** `rc_weight_rel` vs `rc_weight_x_dist` was 0.992 (SEs ±0.12 for both). Centring distance at 1,400 m drops it to -0.02, with identical logit predictions. Remaining highest pair: `hf_avg3_fin_pct_rel` / `hf_last_fin_pct_rel`, 0.74.
- **Result for the report:** centred, weight × distance is -0.006 ± 0.023 in the logit, i.e. no evidence for the proposal's "weight matters more over long distances" in a linear model. LightGBM's version is tested in Week 5.
- Same conclusion as the toy: the market already uses these simple facts, so **Week 3 effort goes into better features** (5.3), not a fancier model.

---

## 8. Risks and what to do

| Risk | What to do |
|---|---|
| 2017-20 patterns don't carry over to 2025-26 (new people, prize inflation, 5-year gap) | Use relative and windowed features, and run the era-drift check (5.5 #7). If Track A is clearly worse than Track B on the Stage 2 window, say so; Track B becomes the main result. Decide with the Stage 2 window only. |
| Small samples (423 / 400 races) | Use bootstrap CIs, keep the models small, and report "no clear difference" honestly |
| A leak makes a model look great | Run checks 5.5 #2-4. Treat anything better than the market as a leak until it's explained. |
| Peeking at the test | Section 3 rules: test metrics only in Week 5, computed once |
| Laptop and SageMaker numbers differ slightly | Take the final numbers from SageMaker only. Fixed seeds. |
| AWS cost | Stop the space after each session (idle shutdown 120 min). US$20/month SageMaker budget alert (`FAHIM_ACCESS_RUNBOOK.md` Step 3); Fahim limited to `ml.t3.medium`-`xlarge`. |
| Behind schedule (31 Oct: Weeks 3-4 not yet run in SageMaker) | Run the starter now (runbook Steps 1-6), then spend the time on features + `eval_utils.py`; Track B can be cut to "base vs base + rating" if time runs short |

**Proposal text vs the data we actually used** (fix in the final report):

- The proposal says the scrape covers "Jan 2025 to Sep 2026". It actually covers **7 Sep 2025 - 11 Oct 2026** (2025/26 + 98 races of 2026/27).
- The proposal says "a four-year gap". The gap between the datasets is **Jun 2020 - Sep 2025, about five years**.

---

## 9. Sources

- Group 17 proposal, `DataScience_Project-v4.pdf`: research question, stages, metrics, six-week timeline.
- Benter, W. (1994), *Computer Based Horse Race Handicapping and Wagering Systems: A Report*. Two-stage model, separate data for Stage 2, pseudo-R² figures, "500 to 1000 races": https://www.Gwern.net/doc/statistics/decision/1994-benter.pdf
- Silverman, N. & Suchard, M. (2013), *Predicting Horse Race Winners Through a Regularized Conditional Logistic Regression With Frailty*, Journal of Prediction Markets 7(1). Hong Kong data, trained on 4 years, tested on a hold-out year: https://ideas.repec.org/a/buc/jpredm/v7y2013i1p43-52.html
- Lessmann, S., Sung, M.-C. & Johnson, J. (2007), *Adapting Least-Square Support Vector Regression Models to Forecast the Outcome of Horseraces*, Journal of Prediction Markets 1(3): https://ideas.repec.org/a/buc/jpredm/v1y2007i3p169-187.html
- Deza, A., Huang, K. & Metel, M. (2017), *Managing losses in exotic horse race wagering*, JORS. Conditional logit vs public, McFadden R² 0.2181 vs 0.2145: https://link.springer.com/content/pdf/10.1057%2Fs41274-017-0213-8.pdf
- statsmodels `ConditionalLogit` (no intercept; groups required): https://www.statsmodels.org/devel/generated/statsmodels.discrete.conditional_models.ConditionalLogit.html
- `shin` package (Shin's method; used to check our own code): https://pypi.org/project/shin
- SageMaker Distribution 4.5.0 package list (lightgbm, statsmodels, scikit-learn included): https://github.com/aws/sagemaker-distribution (`build_artifacts/v4/v4.5/v4.5.0/cpu.env.out`)
