# Evaluation tools (Inns)

Four scripts share one scoring module. All of them follow `docs/DATA_RULES.md`: the race is the
unit, models are compared on the same races, nothing is fitted on the window being scored
unless that window is a fitting window, and h_test is scored once. The decision criteria for
forward-fold evidence, freezing, and final scoring are fixed in
[`docs/evaluation_protocol.md`](../docs/evaluation_protocol.md).

| File | What it does | Touches h_test? |
|---|---|---|
| `eval_utils.py` | Metrics, odds baselines, paired bootstrap, scoreboard, h_test guard | Only with `final_run=True` |
| `rolling_validation.py` | Chronological 3-fold check of market + logit / LightGBM blends on Kaggle | No (Kaggle only) |
| `stage2_alpha_beta.py` | Stage 2 weights (beta for market, alpha for each model) with standard errors | No |
| `final_scoring.py` | Dry run on h_s2, freeze record, then the one-shot h_test scoring | Only in `--final` |
| `build_combined_predictions.py` | One predictions file with every model and baseline | Writes predictions only, scores nothing |
| `provenance.py` | Refuses predictions not made by the live registry and published code | No (reads `meta.json` files only) |

Run everything from the repo root with the project venv active.

## Provenance check (added 7 Nov)

Every fold run in `artifacts/folds/` (6 Nov) recorded `registry_version: v3.0` and commit
`fc759fe`, which is not on GitHub. The registry was already v3.1, so those runs never used
the track/distance, trajectory or interaction features, and the scores gave no sign of it.
`provenance.py` now checks each run's `meta.json`:

- `registry_version` must equal `REGISTRY_VERSION` in `model_logic/feature_registry.py`;
- `git_commit` must be on `origin/main` (run `git fetch origin` first);
- a different `registry_hash` with the same version is a **warning** (registry edited after the run).

Where it is enforced:

| Script | Behaviour on a failed check |
|---|---|
| `final_scoring.py` dry run | Runs, prints the problems and writes them into the dry-run report |
| `final_scoring.py --freeze` / `--final` | **Refuses** |
| `stage2_alpha_beta.py` | Refuses; `--allow-unverified` reproduces the table stamped UNVERIFIED |
| `provenance.load_fold_predictions()` | Refuses; use it instead of `pd.read_parquet` for fold files |
| `build_combined_predictions.py` | Writes `combined_predictions.provenance.json` (source run's meta) next to the output |

```bash
git fetch origin
python analysis/provenance.py        # checks the combined file and artifacts/folds/primary
```

Today this fails on purpose: the combined file was built from the 5 Nov LightGBM v2 output,
which has no `meta.json`, and the fold runs are v3.0 from an unpushed commit. To pass, the
model scripts must write a `meta.json` (as `run_folds.py` does) from a clean checkout of a
pushed commit on registry v3.1, and the combined file must be rebuilt from that run.
Limitation: `run_folds.py` records `git rev-parse HEAD` even if the tree had uncommitted
edits, so only a run from a clean checkout really proves the code.

## Frozen odds-free baseline ladder

`model_logic.run_baselines` fits two historical conditional-logit rungs once per fold from
`s1` only. Both are unpenalised and use the same `s1` inner train/tune split as the main
conditional-logit selector; neither reads market or odds columns. `draw_only` uses
`rc_draw_pct`, `rc_draw_x_hv`, and `rc_draw_x_sprint`. `original_cl` starts with
`jt_jockey_win365_rel`, `hf_avg3_fin_pct_rel`, `rc_draw_pct`, `rc_draw_x_hv`,
`rc_weight_rel`, and `hf_days_since`, then selects only from
`hf_last_fin_pct_rel`, `jt_trainer_win365_rel`, `hf_win5_s`, `hf_first_start`,
`rc_draw_x_sprint`, `rc_weight_x_dist`, `hf_class_change`, `w3_horse_past_win_rate`, and
`w3_horse_past_top3_rate`.

The lists are deliberately frozen rather than derived from active registry sets, which contain
later feature families. The current builder defines `rc_draw_pct` by rank among actual starters;
the historical expression was `(draw - 1) / (field_size - 1)`. Each baseline artifact records
the count and fraction of runners where those definitions differ.

Aggregation writes `ladder.csv`, `ladder.md`, and `ladder_steps.csv` per configuration. All
rungs use the same market-complete valid races within a fold; positive step gain means the second
rung has lower log loss, with a paired race-day bootstrap interval. Pooled rows contain valid
races only. h1's `s1` period mixes Kaggle and HKJC data.

## eval_utils.py

```python
import eval_utils as ev
df = ev.load_predictions("artifacts/combined/combined_predictions.parquet", split="h_s2")
table = ev.scoreboard(df, ["p_market", "p_cl", "p_lgbm", "p_joint"], baseline="p_market_refit")
for _, row in table.iterrows():
    print(row["model"], ev.verdict(row, "vs_refit"))
```

Main functions:

- `check_not_test(df, final_run=False)` raises if any row is in the test window. A row counts
  as test if `split == "h_test"` **or** `race_date >= 2026-04-01`, so files without a `split`
  column (the fold-runner predictions) are caught too. `load_predictions` and `scoreboard` call it.
- `race_losses(p, df)` gives one loss per race, `-log p(winner)`. Dead heat (two or more
  winners): `-log(mean winner p)`. This is the same rule as `docs/week34_starter.py` and Fahim's
  `model_logic/validation/evaluation.race_losses`. Before 6 Nov this module used the mean of
  the two `-log` values, which differed only in dead-heat races.
- `per_race_log_loss`, `brier`, `top1`, `pseudo_r2` (1 - model loss / 1-over-field loss),
  `calibration_table`.
- `paired_bootstrap(loss_a, loss_b, days=...)` gives mean of `a - b` per race (positive means
  `b` is better) with a 95% CI. With `days` (race_id to race day, from `race_days(df)`) it
  resamples whole race days. Races on one day share the going and track bias, so their results
  move together, and resampling single races gives intervals that are too narrow. This is the
  default in `scoreboard` and `rolling_validation` since 6 Nov, and it matches Fahim's
  `paired_bootstrap_gain` exactly (same seed and day order give the same CI, see
  `FOLD_RUNNER_COMPARISON.md`). `days=None` resamples single races.
- Odds baselines: `power_probs`, `shin_probs`, `fit_market_refit_beta` / `market_refit_prob`
  (p proportional to market^beta), `add_odds_baselines`. Fit beta only on a fitting window; on
  a held-out window pass the beta from training.
- `scoreboard(df, model_cols, baseline, raw_market, final_run=False, by_day=True)`: one row per
  model with log loss, pseudo-R2, Brier, top-1 and the paired gain against the raw market and
  the refit. It only uses races where every compared column is present and the market is
  complete. `table.attrs` records `n_races`, `n_days` and the bootstrap method.
- `verdict(row, "vs_refit")` reads a row in plain words: better / worse / no clear difference.
  "Better" needs the whole CI above 0.

## rolling_validation.py

```
python analysis/rolling_validation.py
```

Kaggle only. Three folds (validation Feb-Jul 2019, Aug 2019-Jan 2020, Feb-Jul 2020). Each fold
trains on races strictly before its window. The LightGBM tree count comes from an inner date
split inside training. The blend weights are fitted on out-of-fold Stage 1 predictions from
that inner split, then scored once on the window. The baseline is the market refit with beta
from the training window. It also re-runs everything without the `jt_trainer_*` features.

Logit features come from the registry (`get_features("conditional_logit")`, 32 in v3.1). The
logit here is unpenalised, so on each fold's training window it drops a feature that has
|Spearman| > 0.9 with an earlier kept one, or that is an exact linear combination of kept ones
(e.g. `tr_last_minus_avg3_fin_pct_rel`). The dropped list is printed in the report.

Outputs: `analysis/rolling_validation_results/{fold_metrics,pooled_summary,ablation_no_trainer}.csv`
and `report.md`. Current result (6 Nov): no blend beats the refit; the best is
`mkt_cl` at +0.0003 (95% CI -0.0013 to +0.0020). Removing trainer features does not help.

## stage2_alpha_beta.py

```
python analysis/stage2_alpha_beta.py
```

Refits `score = beta * log p_market + alpha * log p_model` as a plain conditional logit on each
Stage 2 window (HKJC `h_s2` from the combined file, and the `s2` rows of Fahim's folds k1, k2,
h1), so each weight has a standard error, and tests each blend against market only
(likelihood ratio). The ridge-penalised weights the team's scripts used are listed next to it.
All of it is in-sample. Outputs in `analysis/stage2_results/`. See its `report.md` for how to
read it.

## final_scoring.py

```
python analysis/final_scoring.py                      # 1. dry run on h_s2 (safe, any time)
python analysis/final_scoring.py --freeze --s3-uri s3://<bucket>/predictions/combined_predictions.parquet
python analysis/final_scoring.py --final --confirm "score h_test once"     # 3. once, Week 5
```

1. **Dry run** scores the h_s2 rows with the same code as the final run, into
   `analysis/final_scoring_results/dry_run_h_s2/`. h_s2 is where `p_joint` and the refit beta
   were fitted, so these numbers are in-sample. It checks the pipeline; it is not evidence.
2. **Freeze** writes `analysis/final_scoring_results/freeze_record.json`: SHA-256 and MD5 of the
   predictions file, git commit and any uncommitted files, rows and races per split, columns,
   which models will be scored, and (with `--s3-uri`) the S3 VersionId and ETag of the uploaded
   copy (uses AWS profile `cloudpc`, region `ap-southeast-1`). It reads no h_test outcome.
   Commit the record before the final run.
3. **Final** refuses unless the confirmation phrase is given, the freeze record exists, the
   predictions file still has the frozen SHA-256, and `final_scoring_results/h_test/` does not
   exist yet. It scores h_test rows only, refits nothing (the refit beta and `p_joint` carry
   their h_s2 fits), and writes `scoreboard.csv`, `calibration.csv` and `report.md`. A second
   run is refused.

Order for Week 5: Track B columns added to the combined file, rebuild, upload to S3, dry run,
freeze, commit, then the final run.

## Tests

`tests/test_eval_and_final_scoring.py` uses only synthetic data. It checks the h_test guard
(by split and by date), the dead-heat rule, the day bootstrap (against a manual resample and
against Fahim's `paired_bootstrap_gain`), and every refusal path of `final_scoring.py`
(no confirmation, no freeze record, changed file, second run).

```
python -m pytest "leakage test/test_leakage.py" tests -q -p no:cacheprovider
```
