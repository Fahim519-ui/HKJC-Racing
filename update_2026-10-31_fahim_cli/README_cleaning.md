# Data cleaning (Weeks 1-2)

- **What's been done, and the results:** `docs/PROGRESS_SUMMARY.md` (cleaning done, checked and shared on 13 Oct 2026)
- **Step-by-step guide and next steps:** `docs/WEEK1-2_RUNBOOK.md`
- **Column meanings:** `docs/data_dictionary.md`
- **Weeks 3-4 plan (features + model training):** `docs/WEEK3-4_PLAN.md`; how to run it in SageMaker: `docs/WEEK3-4_RUNBOOK.md` (`week34_starter.py`)

**Status (Sat 31 Oct 2026):** Weeks 1-2 done and handed over (16 Oct). Weeks 3-4: Nayeon = features, Fahim = models in his own SageMaker space (target Wed 11 Nov); report due Tue 17 Nov. Fahim's AWS access: `docs/FAHIM_ACCESS_RUNBOOK.md` (files in `aws/fahim/`); his guide: `docs/WEEK4_FAHIM.md`. Group data rules for everyone: `docs/DATA_RULES.md`.

## Run it (Windows, Kiro terminal, this folder)

```powershell
.\.venv\Scripts\Activate.ps1          # every new terminal; the prompt then starts with (.venv)
python clean_data.py
```

First time only: `py -3.14 -m venv .venv`, then activate, then `pip install -r requirements-cleaning.txt`.

Inputs, from `data\`:

- `hkjc_results_2025-07-26_to_2026-10-11.csv` - the scraped HKJC data
- `Horse_Racing_Results_2017-2020.csv` - the Kaggle data
  - Your Numbers-exported CSV is identical, value for value, to Kaggle's original. Rename it to this name.
  - The script finds the Kaggle file by "2017" and "2020" in its name, and prefers the CSV.

## Outputs

| File | Type | Use it for |
|---|---|---|
| `data\processed\runners_merged.parquet` | Parquet | **Main output** - both sources in one table. Share this with Fahim (`pd.read_parquet`) |
| `data\processed\runners_merged.csv` | CSV | the same table, for Excel. Look, don't re-save |
| `data\interim\hkjc_2025_26_clean.*` | Parquet + CSV | HKJC rows only |
| `data\interim\kaggle_2017_2020_clean.*` | Parquet + CSV | Kaggle rows only |
| `reports\cleaning_report.md` | Markdown | counts, missing values, flagged races, name links |
| `reports\figures\*.png` | PNG | written by `notebooks\02_eda.ipynb` (Week 2) |

Parquet keeps the column types (dates, whole numbers, True/False, blanks) and is about 10x smaller than CSV.

## Rules the script follows

- Raw files are never changed.
- Missing values stay blank (NaN), never 0.
- Problem races are flagged (`race_valid = False`, reason in `race_issue`), not deleted. This includes placings that disagree with the horses' own finish times. For modelling, filter `race_valid & started`.
- If the HKJC data is ever re-scraped with each horse's own `Finish Time`, the script picks it up with no code change (`FINISH_TIME_COLS` in `clean_data.py`).
- The script stops with "CHECKS FAILED" if any of these fail:
  - a horse appears twice in a race
  - market probabilities don't sum to 1
  - a valid race has no winner
  - a draw, weight, distance, odds, age or rating value is out of range
