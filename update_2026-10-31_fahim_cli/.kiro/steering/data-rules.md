# Data rules for this project

- Never change the raw files in `data/` (the two input CSVs). Write outputs only to `data/interim`, `data/processed` and `reports`.
- `clean_data.py` and `docs/data_dictionary.md` are the source of truth for column names and meanings.
- Missing values stay NaN. Never fill them with 0. Any imputation happens in modelling, fitted on the training dates only.
- Never use post-race columns as features for the same race: `finish_*`, `is_winner`, `is_top3`, `running_positions`, `race_winner_time_s`, `kaggle_fg_figure`, `kaggle_path`.
- `win_odds`, `implied_prob` and `market_prob` are the market baseline (or a Stage 2 input), never a "fundamental" feature.
- Never use `profile_missing`, or whether HKJC country/age/rating is missing, as a feature. The blanks mark horses that left HK racing before the scrape (future information). Don't drop those rows either.
- Model only on `race_valid & started` rows. Split by race date, never randomly by row.
- Never put AWS keys, tokens or passwords in any file.

# Features and models (Weeks 3-4, see `docs/WEEK3-4_PLAN.md`)

- Features use history from earlier race days only, within the same source (`kaggle` / `hkjc`). Horse IDs never link across sources.
- Build history only from `race_valid & started` rows.
- Use the splits in the plan (section 3) and keep them in a `split` column. Never compute metrics on the test window (`h_test`) before Week 5.
- HKJC blanks: fill rating from carried weight in the same race and age from the `horse_id` year (plan 5.4). Never add a "was missing" flag.
- Prefix features by group: `rc_`, `hf_`, `jt_`, `rs_`, `rt_`.
- Convert pandas nullable types to float (`.astype(float).to_numpy()`) before statsmodels.
- Centre the continuous part of an interaction before multiplying (e.g. `rc_weight_rel * (distance_m - 1400) / 1000`), or it moves with the main feature.
- Check Spearman correlations between logit features on `k_train`; for pairs above 0.9, keep only one in `FEATS_CL`.
- Choose features and settings on the Kaggle tuning window (`k_tune`) only. Stage 2 is fitted on `h_s2`; its log loss there is in-sample.

# Environment and AWS

- Python 3.14 virtual environment in `.venv`. Activate it (`.\.venv\Scripts\Activate.ps1`) before running anything. Don't edit files inside `.venv`.
- AWS for this project: region `ap-southeast-1` (Singapore) only, AWS CLI profile `cloudpc` (sign in with `aws sso login --profile cloudpc`). Always pass `--region ap-southeast-1`.
- The S3 bucket name contains `sagemaker`. Keep it private: no public access, no public links.
