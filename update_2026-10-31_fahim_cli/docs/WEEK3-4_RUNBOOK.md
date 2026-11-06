# Weeks 3-4 runbook: features and model training in AWS (SageMaker)

**Written:** Wed 28 Oct 2026. **Updated:** Sat 31 Oct 2026. **Why and what:** `docs/WEEK3-4_PLAN.md`. This file is the **how**, click by click.

## Where things stand (Sat 31 Oct)

- **Owners (agreed 31 Oct):** Week 3 features = **Nayeon** (laptop), Week 4 models = **Fahim** in **his own** SageMaker space - his guide is `docs/WEEK4_FAHIM.md`, and his access is set up with `docs/FAHIM_ACCESS_RUNBOOK.md`. This runbook stays as the reference for running the starter in **your** space (optional, e.g. to double-check his numbers). Deadlines: Weeks 3-4 by Wed 11 Nov, report Tue 17 Nov, presentations 18/25 Nov.
- **Not run in SageMaker yet** (as far as these notes go). On the plan's calendar Week 4 ended Fri 30 Oct, so this is now the first job.
- **31 Oct re-check:** the script was re-run with the same package versions as the SageMaker space (Python 3.12, pandas 2.3.3, LightGBM 4.6.0, statsmodels 0.15.0, scikit-learn 1.7.2). It runs in about 25 s, and the leak checks pass: changing one race's result only changes features of *later* race days, and jockey/trainer values are equal for all of a person's rides on one day.
- **31 Oct fix in `week34_starter.py`:** `rc_weight_x_dist` now uses distance centred at 1,400 m. Uncentred, it was 0.99 correlated with `rc_weight_rel` (plan rule: keep one of any pair above 0.9), which blew up both standard errors (±0.12). Logit predictions are unchanged; LightGBM moves slightly. **The Step 5 check numbers below are for the 31 Oct version.**
- **Finding for the report:** with the centred term, weight × distance adds nothing in the logit (−0.006 ± 0.023). The proposal's example "weight mattering more over long distances" isn't supported by the logit on this data. LightGBM can still test it in Week 5.

**Order from here:** Steps 1-6 (about 20 min) → Step 7 (better features + checks) → Step 8 upload → Step 9 stop → Step 10 message. Then "Before Week 5" at the end of this file.

**What you'll do:** upload one script to S3 → open your SageMaker JupyterLab space → copy the data and script into it → run the script → look at the results → upload the outputs to S3 → stop the space.

**The models are classical** (conditional logit with statsmodels, LightGBM), not a foundation model - see the proposal: "The predictive models themselves are classical." They train in under a minute on your `ml.t3.medium` space.

| Step | What | Where | Time |
|---|---|---|---|
| 1 | Put the files in the project folder | Laptop | 2 min |
| 2 | Upload the script to S3 | Laptop terminal | 2 min |
| 3 | Start the SageMaker space | Browser (AWS console) | 3-5 min |
| 4 | Copy data + script into the space | SageMaker terminal | 1 min |
| 5 | Run the starter pipeline | SageMaker terminal | 1 min |
| 6 | Read the results | JupyterLab | 10 min |
| 7 | Improve the features, re-run (Week 3 work) | JupyterLab | most of Week 3 |
| 8 | Final run + upload outputs to S3 | SageMaker terminal | 2 min |
| 9 | **Stop the space** | Browser | 1 min |
| 10 | Tell the group | Group chat | - |

---

## Step 1 - Put the files in the project folder (laptop)

Extract the update zip into `C:\Users\Lenovo\Documents\HKU3\ELEC3544\Group project\4` and choose **Replace**. You should now have:

```
4\
  week34_starter.py          <- new: the whole Weeks 3-4 pipeline
  clean_data.py
  data\processed\runners_merged.parquet
  docs\WEEK3-4_PLAN.md
  docs\WEEK3-4_RUNBOOK.md    <- this file
```

## Step 2 - Upload the script to S3 (laptop, Kiro terminal)

```powershell
cd "C:\Users\Lenovo\Documents\HKU3\ELEC3544\Group project\4"
.\.venv\Scripts\Activate.ps1
aws sso login --profile cloudpc                 # not "aws login"
$B = "inns-elec3544-dsfm-sagemaker"
$R = "--region", "ap-southeast-1", "--profile", "cloudpc"
aws s3 cp week34_starter.py "s3://$B/code/" @R
aws s3 cp docs\WEEK3-4_PLAN.md "s3://$B/code/docs/" @R
aws s3 ls "s3://$B/code/" @R
```

**Check:** the list shows `week34_starter.py`. The cleaned data is already in S3 (`processed/runners_merged.parquet`, 1,122,677 bytes) from Week 2 - no need to upload it again.

## Step 3 - Start the SageMaker space (browser)

1. AWS console → region **Asia Pacific (Singapore) ap-southeast-1** (top right).
2. **Amazon SageMaker AI** → **Domains** → `QuickSetupDomain-...` → **Launch** → **Studio**.
   (Not "Amazon SageMaker" / Unified Studio - its address contains `/datazone/`.)
3. In Studio: **JupyterLab** → space `inns-elec3544-dsfm-datacleaning` → **Run space** → wait until **Running** → **Open JupyterLab**.

Same space as Week 2 (`ml.t3.medium`, 5 GB). Its image already has **lightgbm 4.6.0, statsmodels 0.15.0, scikit-learn 1.7.2** - nothing to install.

## Step 4 - Copy data + script into the space (SageMaker terminal)

In JupyterLab: **File → New → Terminal**. Then:

```bash
B=inns-elec3544-dsfm-sagemaker
mkdir -p ~/dsfm && cd ~/dsfm
aws s3 cp s3://$B/code/week34_starter.py .
aws s3 cp s3://$B/processed/runners_merged.parquet data/processed/
ls -l week34_starter.py data/processed/
python -c "import lightgbm, statsmodels, sklearn, pandas; print(lightgbm.__version__, statsmodels.__version__, sklearn.__version__, pandas.__version__)"
```

**Check:** the parquet is `1122677` bytes; versions print `4.6.0 0.15.0 1.7.2 2.3.3`.

(No `--profile` here: inside SageMaker the space's own role signs the requests. Its policy allows buckets with `sagemaker` in the name - that's why the bucket is named that way.)

## Step 5 - Run the starter pipeline (SageMaker terminal)

```bash
cd ~/dsfm
python week34_starter.py
```

It takes about a minute. What it does, in order:

| Part | What happens | Plan section |
|---|---|---|
| Features | Builds 19 features from **earlier race days only** (draw, weight, last finishes, jockey/trainer 365-day win rate...) and the `split` column | 5 |
| Baselines | Market (1/odds normalised), power, Shin, guessing, market-only refit | 6.3 |
| Stage 1a | Conditional logit: tuned on Kaggle 2017-19 → 2019/20, refit on all Kaggle | 6.4 |
| Stage 1b | LightGBM: early stopping on 2019/20, refit on all Kaggle | 6.5 |
| Stage 2 | α and β fitted on HKJC Nov 2025 - Mar 2026, for both models | 6.6 |
| Save | predictions for HKJC Nov 2025 - Oct 2026; **test window not scored** | 6.8 |

**Check - the end of the output should look like this** (numbers from my 31 Oct test run on the same data, same package versions):

```
features: 38,942 rows, 19 features
k_train 15836 / k_tune 9299 / h_s2 5287 / h_test 4978 / warmup 3542
conditional logit per-race log loss: 2.1804      (tuning window)
LightGBM per-race log loss: 2.1837 (trees: 102)
market on the same races: 1.9947
Stage 2 alpha: p_cl -0.3453, p_lgbm -0.3956
predictions: 10,265 rows
```

If yours match, the AWS run reproduces the reference - take a screenshot for the report.

(The 28 Oct version of the script gave `LightGBM 2.1790 (trees: 107)` and `p_lgbm -0.3951`; the logit and market lines are the same in both versions. If you see that, you're running the old script - re-do Steps 1-2 with the 31 Oct zip.)

**Files it writes** (inside `~/dsfm`):

| File | What |
|---|---|
| `data/features/features_v1.parquet` | one row per starter: keys, `split`, `is_winner`, odds, 19 features |
| `data/predictions/predictions_v1.parquet` | one row per HKJC runner (Stage 2 + test windows), one `p_...` column per model |
| `models/cl_coefficients.csv` | conditional logit weights ± SE |
| `models/lgbm_v1.txt` | the LightGBM model |
| `reports/week4_models.md` | the summary printed above |

## Step 6 - Read the results (JupyterLab)

Open `reports/week4_models.md` (double-click in the file browser, right-click → Open With → Markdown Preview).

**What the starter run shows (honest reading):**

- **The simple models are well behind the market** on its own: 2.18 vs 1.99 per-race log loss on 2019/20. Expected - the market already knows these simple facts.
- **The logit's signs make sense:** outside draws are bad (more so at Happy Valley), better recent finishes are good, a strong jockey is the biggest single factor. Weight and weight × distance are both about 0 once the market-free model has form and jockey.
- **Stage 2 α is negative** (about -0.35 ± 0.11 for the logit, -0.40 ± 0.12 for LightGBM) and β about 1.24: where the simple model disagrees with the market, the market is right *and then some*. The Stage 2 log loss on Nov-Mar looks lower (2.024 vs 2.038) but that's the same races it was fitted on - **it means nothing until the test in Week 5.**
- So the Week 3 job is clear: **better features**, not a fancier model.

## Step 7 - Improve the features (the real Week 3 work)

Edit `build_features()` in `week34_starter.py` (in JupyterLab: double-click the file). Add features from plan section 5.3, one group at a time, then re-run Step 5 and compare the **tuning-window log loss** (lower = better). Never look at the test window.

Good next additions, in order:

1. `hf_dist_top3_s`, `hf_track_top3_s` - horse's record at this distance / track.
2. `hf_starts_365`, `hf_best5_fin_pct`, `hf_top3_5_s`.
3. `jt_trainer_top3_365`, `jt_combo_win365` (jockey + trainer pair).
4. Add each new feature's name to `FEATS_CL` (if it varies between horses in a race) and `FEATS_LGB`.
5. HKJC-only experiment (Track B, plan 6.7: rating filled from weight, running style) - a second script once Track A is settled.

**Rules while editing** (also in `.kiro/steering/data-rules.md`):

- history from **earlier race days**, same `source`; `groupby(...).shift(1)` for horses, the `closed="left"` rolling for jockeys/trainers;
- never use the same race's result, the odds (Stage 1), or `profile_missing` / blank flags;
- keep features that work on Kaggle **and** HKJC for Track A;
- **interactions:** centre the continuous part first (e.g. distance − 1,400), or the product moves with the main feature;
- **after each new group, check correlations** on `k_train` (Spearman): for pairs above 0.9, keep only one in `FEATS_CL`. Today the only pair above 0.7 is `hf_avg3_fin_pct_rel` / `hf_last_fin_pct_rel` (0.74), which is fine.

Tip: you can also edit on the laptop in Kiro, test there (`python -m pip install lightgbm==4.6.0 statsmodels==0.15.0 scikit-learn scipy`), then re-upload with Step 2. Final numbers come from SageMaker.

## Step 8 - Final run + upload outputs to S3 (SageMaker terminal)

```bash
cd ~/dsfm
python week34_starter.py
B=inns-elec3544-dsfm-sagemaker
aws s3 cp week34_starter.py s3://$B/code/
aws s3 cp data/predictions s3://$B/predictions/ --recursive
aws s3 cp models s3://$B/models/ --recursive
aws s3 cp reports/week4_models.md s3://$B/reports/
aws s3 ls s3://$B/ --recursive | grep -E "features|predictions|models|week4"
```

Don't upload the starter's own `data/features/` - S3 `features/` is for Nayeon's Week 3 files.

**Check:** the last command lists `predictions_v1.parquet`, `cl_coefficients.csv`, `lgbm_v1.txt`, `week4_models.md`.

Optional - copy them to the laptop too (laptop terminal):

```powershell
aws s3 cp "s3://$B/predictions/" data\predictions\ --recursive @R
aws s3 cp "s3://$B/reports/week4_models.md" reports\ @R
```

## Step 9 - Stop the space (browser) - every time

Studio → **Compute → Running instances** (or the JupyterLab space list) → `inns-elec3544-dsfm-datacleaning` → **Stop**. Wait until it shows **Stopped**.

Files in `~/dsfm` stay on the space's 5 GB disk, and the outputs are in S3 anyway. The 120-minute idle shutdown is only a backstop.

Check from the laptop that nothing is running (prints nothing = all off):

```powershell
aws sagemaker list-apps --region ap-southeast-1 --profile cloudpc --query "Apps[?Status!='Deleted'].[DomainId,SpaceName,AppType,Status]" --output table
```

## Step 10 - Tell the group

> Weeks 3-4 pipeline runs in SageMaker (`week34_starter.py`). Predictions and models are in our S3 bucket (`predictions/`, `models/`); Nayeon's features go in `features/`. Splits: train Kaggle Nov 2017-Jun 2020, Stage 2 HKJC Nov 2025-Mar 2026, test HKJC Apr-Oct 2026 (not scored yet - Week 5). First result: the simple models are behind the market and Stage 2 α is negative, so Week 3 is about better features. Plan: `docs/WEEK3-4_PLAN.md`.

---

## Before Week 5 (still to write)

| Item | Plan section | Why it's needed |
|---|---|---|
| `eval_utils.py`: log loss, Brier, top-1, pseudo-R², calibration table, paired bootstrap | 6.2 | Week 5 scores every model with the same code, once |
| Feature checks 5.5 #1-8 written to `reports/feature_checks.md` (#2 and #3 already pass on the starter features, 31 Oct) | 5.5 | the proposal's "correlation and target-leakage checks" |
| `docs/feature_dictionary.md` | 5.1 | every feature: group, definition, Track A/B |
| Track B (HKJC only: rating filled from weight, running style) → `p_b_*` columns | 5.4, 6.7 | sub-question 1 |
| LightGBM gain importance + `pred_contrib` | 6.5 | Week 5 interpretation |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `aws s3 cp` in SageMaker: AccessDenied | Check the bucket name has no typo; it must contain `sagemaker`. |
| `ModuleNotFoundError: lightgbm` | You're in the wrong image/space - use `inns-elec3544-dsfm-datacleaning` (SageMaker Distribution). On the laptop: `python -m pip install lightgbm==4.6.0 statsmodels==0.15.0 scikit-learn scipy`. |
| `TypeError: ufunc 'isfinite' not supported` | A new feature is a pandas nullable type - add `.astype(float)` (the script already does this for features in the lists). |
| `FileNotFoundError: runners_merged.parquet` | Run from `~/dsfm` and repeat Step 4. |
| Numbers differ slightly from Step 5's check | Different package versions (e.g. the laptop's newer LightGBM). Use SageMaker's numbers. |
| Laptop: `pip` blocked by Application Control | Use `python -m pip ...`. Don't turn the protection off. |
| `aws` says the token expired (laptop) | `aws sso login --profile cloudpc` |
| Space won't start / stuck "Pending" | Wait 5 min, refresh. Still stuck: Stop, then Run again. |

## Costs

- `ml.t3.medium` space: a few US cents per running hour - **stop it after each session**.
- S3 for the new files: a few MB, well under US$0.01 a month.
- Don't open Canvas / Data Wrangler (US$1.90 an hour while logged in).
