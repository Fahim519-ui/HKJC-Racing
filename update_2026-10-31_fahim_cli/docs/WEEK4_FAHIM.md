# Week 4 in SageMaker - guide for Fahim (Group 17)

**Written:** Sat 31 Oct 2026 by Inns. **Your part:** Week 4 - the models (odds baselines, conditional logit, LightGBM, Stage 2, Track B). **Group target:** models done by **Wed 11 Nov**, so Week 5 (scoring the test races once) and the report (**due Tue 17 Nov, 11:59 PM**) have time. Presentations: 18 / 25 Nov.

The full why-and-what is in `docs/WEEK3-4_PLAN.md` (sections 3 and 6). This file is the how.

## Your environment

- Sign in at the AWS access portal → our account → **ELEC3544-DSFM-Fahim** → Management console (opens SageMaker in Singapore). Sign-in lasts 12 h; then sign in again.
- **Applications and IDEs → Studio →** user **fahim → Open Studio → JupyterLab.** First time: **Create JupyterLab space** (Private), instance `ml.t3.medium` → Run → Open JupyterLab.
- Instances allowed: `ml.t3.medium` (default, enough for everything here), `ml.t3.large`, `ml.t3.xlarge`. Others are blocked.
- Python 3.12 with pandas 2.3.3, lightgbm 4.6.0, statsmodels 0.15.0, scikit-learn 1.7.2 - **nothing to install.**
- **Stop the space when you finish** (Studio → Running instances → Stop). It auto-stops after 2 h idle, but please don't rely on that - it runs on Inns's account.
- You can read/write only our bucket `inns-elec3544-dsfm-sagemaker`. Inside SageMaker you don't need `--profile`.

## What's already in `~/dsfm`

Copied from S3 every time JupyterLab starts (log: `~/.dsfm_sync.log`):

| Path | What | Refreshed |
|---|---|---|
| `data/processed/runners_merged.parquet` | the cleaned table (39,210 rows; model on `race_valid & started` = 38,942) | every start |
| `data/features_week3/` | Nayeon's Week 3 features, once she uploads them to `s3://.../features/` | every start |
| `docs/` | this file, `WEEK3-4_PLAN.md`, `data_dictionary.md`, `DATA_RULES.md` | every start |
| `week34_starter.py` | working pipeline: 19 simple features, all baselines, logit, LightGBM, Stage 2 | first start only |
| `reports/cleaning_report.md`, `reports/figures/`, `notebooks/02_eda.ipynb` | Week 2 background (data quality, EDA) | first start only |

**Don't edit files in `data/` or `docs/`** - they're overwritten at the next start. Your own scripts and notebooks anywhere else in `~/dsfm` are never touched.

## Step 1 - Run the starter (2 min)

File → New → Terminal:

```bash
cd ~/dsfm
python week34_starter.py
```

The end of the output should match:

```
features: 38,942 rows, 19 features
conditional logit per-race log loss: 2.1804
LightGBM per-race log loss: 2.1837 (trees: 102)
market on the same races: 1.9947
predictions: 10,265 rows
```

It writes `data/features/features_v1.parquet` (its own simple features), `data/predictions/predictions_v1.parquet`, `models/` and `reports/week4_models.md`.

## Step 2 - What it shows

- **Splits (fixed, by date):** Stage 1 train Kaggle Nov 2017 - Jul 2019, tune Kaggle 2019/20, final fit Kaggle Nov 2017 - Jun 2020; **Stage 2 fit HKJC Nov 2025 - Mar 2026 (423 races); test HKJC Apr - Oct 2026 (400 races) - not scored until Week 5.**
- Simple models are well behind the market (2.18 vs 1.99 per-race log loss on 2019/20; lower is better).
- Stage 2 α is negative (-0.35 ± 0.11 logit, -0.40 ± 0.12 LightGBM), β ≈ 1.24; the market-only refit β = 1.04. Where the simple model disagrees with the odds, the odds are right. **Better features matter more than a fancier model.**

## Step 3 - Your Week 4 work (plan section 6)

| # | Task | Plan |
|---|---|---|
| 1 | Use Nayeon's features when they arrive (`data/features_week3/`): join on `race_id` + `horse_id`, keep her `split` column. Until then, work on the starter's `build_features()` | 5.3 |
| 2 | Conditional logit: start from about 6 strong features, add one only if `k_tune` log loss improves ≥ 0.001; stop at about 15; check signs; save coefficients ± SE | 6.4 |
| 3 | LightGBM: ≤ 20 settings (learning rate 0.02-0.05, leaves 7-31, min child 50-200, lambda 1-10), early stopping on `k_tune`, refit on all Kaggle; save gain importance and `predict(X, pred_contrib=True)` | 6.5 |
| 4 | Stage 2 for both models + market-only refit on `h_s2`; report α, β with SEs | 6.6 |
| 5 | Track B (HKJC only): small logit + shallow LightGBM on `h_s2` with rating / running style (base, + rating, + running, all) → `p_b_*` columns | 6.7 |
| 6 | One predictions file with every model's `p_...` column for the Stage 2 + test windows | 6.8 |

Choose features and settings on `k_tune` only. **Never compute metrics on `h_test`** - Week 5 does that once, for all models.

## Rules (full list: `docs/DATA_RULES.md`)

- Features use history from **earlier race days** only, same `source`. Never the same race's result, never the odds (`win_odds`, `market_prob`) in Stage 1, never `profile_missing` or "was blank".
- Convert to float before statsmodels: `.astype(float).to_numpy()`.
- Centre interactions (e.g. `weight_rel * (distance_m - 1400)`); for feature pairs with |Spearman| > 0.9, keep one in the logit.

## Step 4 - Save outputs to S3 (SageMaker terminal)

```bash
cd ~/dsfm
B=inns-elec3544-dsfm-sagemaker
aws s3 cp data/predictions s3://$B/predictions/ --recursive
aws s3 cp models s3://$B/models/ --recursive
aws s3 cp reports/week4_models.md s3://$B/reports/
aws s3 cp week34_starter.py s3://$B/code/fahim/            # and any other scripts/notebooks you write
aws s3 ls s3://$B/ --recursive | grep -E "predictions|models|week4|code/fahim"
```

Please **don't write to** `processed/`, `raw/`, `interim/` or `code/docs/`. `features/` is for Nayeon's Week 3 files only - if she sends them to you, upload them there exactly as she sent them: `aws s3 cp <file> s3://$B/features/` (they then appear in `~/dsfm/data/features_week3/` at your next start, or copy them there yourself). Never upload the starter's own `data/features/` to S3. The bucket keeps old versions, so an accidental overwrite or delete can be undone - tell Inns.

## Step 5 - Stop the space

Studio → **Running instances** → Stop. Your files in `~/dsfm` stay on the space's disk.

## Troubleshooting

| Problem | Fix |
|---|---|
| `~/dsfm` is empty or missing files | `cat ~/.dsfm_sync.log`. Copy by hand: `mkdir -p ~/dsfm/data/processed && cd ~/dsfm && aws s3 cp s3://inns-elec3544-dsfm-sagemaker/processed/runners_merged.parquet data/processed/ && aws s3 cp s3://inns-elec3544-dsfm-sagemaker/code/week34_starter.py .` |
| "not authorized" when starting a space | instance not allowed - use `ml.t3.medium` / `large` / `xlarge` |
| `AccessDenied` on S3 | only our bucket works; check the name |
| `TypeError: ufunc 'isfinite' not supported` | a feature is a pandas nullable type - add `.astype(float)` |
| Numbers differ from Step 1 | you edited the starter before running it; the reference is the unedited file (`s3://.../code/week34_starter.py`) |
| Anything else | message Inns with the error text |
