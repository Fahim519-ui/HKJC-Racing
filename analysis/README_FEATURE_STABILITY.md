# Feature stability audit: what it does and what it found

**Script:** `analysis/feature_stability_audit.py` (about 35 seconds).
**Results:** [`analysis/feature_stability_results/`](feature_stability_results/) ([`stability_report.md`](feature_stability_results/stability_report.md), [`smd_by_feature.png`](feature_stability_results/smd_by_feature.png), two CSVs, `audit_meta.json`).
**Owner:** Ek-Anankul Wirith (Inns). Roadmap of 4 Nov, priority 5 ("Job A").
**Run on:** 6 Nov 2026 evening, repo commit `fcafb9e`, **42 features**, feature registry `v3.1`. History: 4 Nov at `c22b3f3` (25 features), 6 Nov at `5933fa2` (27, after the jockey-trainer pair features). The 42-feature run adds the 15 v3.1 track/distance, trajectory and interaction features; the original 27 kept every number exactly (maximum difference 0.0, same statuses), so the builder rewrite did not change them. The run date and commit are always in `audit_meta.json`.

---

## 1. The question in one sentence

The models **learn** from Kaggle races (2017-20) and are **used** on HKJC races (2025-26). Does each feature look and behave the same in both? If not, a pattern learned in 2018 may not hold in 2026.

## 2. Short answer

Counts for registry v3.1 (8 Nov runs). The 38 model inputs come from the status sheet
[`feature_stability_results/STABILITY_V3_1.md`](feature_stability_results/STABILITY_V3_1.md);
the default audit adds 4 raw rates that no model reads directly (42 columns,
[`stability_report.md`](feature_stability_results/stability_report.md)).

| Status | 38 model inputs | Default audit (42) | Meaning |
|---|---:|---:|---|
| **review** | 1 | 2 | Real difference between Kaggle and HKJC. Discuss as a group. |
| **watch** | 9 | 10 | Smaller warning, or a shift with a known harmless cause |
| **keep** | 28 | 30 | No warning (includes the 2 pair features) |

(Earlier, 27-feature run of 6 Nov: 2 review, 8 watch, 17 keep.)

- **The main risk is the trainer win rate** (`jt_trainer_win365_rel`, the model input, and the raw `jt_trainer_win365` it is built from). HKJC trainers are more evenly matched than in 2017-20, and the trainer rate predicts winners about **3 times less strongly** in HKJC Stage 2 than in Kaggle.
- **Most other "shifts" come from one known cause:** HKJC data only starts in September 2025, so horses have shorter histories. When we compare against a Kaggle window with equally short histories, those shifts disappear.
- **Form, draw and weight features are stable.** Their values and their effects look the same in both datasets.

**Important:** this is a warning list, not a rule for dropping features. The project rules say features are chosen on the Kaggle tuning window (`k_tune`) only. The effect checks below use HKJC Stage 2 results, so act on a flag only after confirming it on `k_tune` (for example in the rolling validation, Job B).

## 3. The windows compared

| Window | Data | Runners | Role here |
|---|---|---|---|
| `k_train` | Kaggle Nov 2017 - Jul 2019 | 15,836 | **Reference**: where the models learn |
| `k_tune` | Kaggle 2019/20 | 9,299 | Drift inside Kaggle over time |
| `k_early` | Kaggle Nov 2017 - Mar 2018 (part of `k_train`) | 4,711 | **History-matched** comparison (see below) |
| `h_s2` | HKJC Nov 2025 - Mar 2026 | 5,287 | **Main target**: HKJC Stage 2 |
| `h_test` | HKJC Apr - Oct 2026 | 4,978 | Information only. Distributions and blank rates, never results |

**Why `k_early`?** Both datasets start in early September (Kaggle 2017, HKJC 2025). `h_s2` covers months 2-6 after the start, so a horse has at most a few months of history. `k_train` runs up to two years after the start, so its horses have much longer histories. Any feature built from past starts will look different for that reason alone. `k_early` covers the same months after the start as `h_s2` (Nov-Mar). If a shift disappears against `k_early`, it is caused by **history length, not by a different era**.

## 4. The four measurements (in plain words)

| Measure | What it asks | Read it as |
|---|---|---|
| **SMD** (standardised mean difference) | Did the average move, in units of the feature's spread? | below 0.10 small, 0.10-0.25 moderate, above 0.25 large |
| **PSI** (population stability index) | Did the whole shape change? Values are put into 10 bins by Kaggle-train deciles (plus a bin for blanks), and the share in each bin is compared | below 0.10 stable, 0.10-0.25 moderate, above 0.25 large |
| **Blank rate** | Is the feature missing more often? | flag a difference of 5 percentage points or more |
| **Effect** | Does the feature predict winners as strongly? A one-feature conditional logit per window, coefficient per 1 SD of the feature (Kaggle-train scale) | flag if the HKJC coefficient differs by 2 or more combined standard errors, or changes sign |

The KS statistic is also in the CSV. With thousands of runners, its p-value flags almost any tiny difference, so only the statistic is kept and it is not used for the status.

**Note on `_rel` and `_rk` features:** `_rel` = value minus the race average, `_rk` = rank within the race. Their average is 0 (or fixed) **by construction**, so their SMD is always about 0 and tells you nothing. For them, look at PSI and the effect.

## 5. How the status is decided

1. **review** if (a) the HKJC Stage 2 shift is large **and** still large against the history-matched `k_early`, or (b) the effect changes sign.
2. **watch** if anything else is flagged: a moderate shift, a shift explained by history length, drift inside Kaggle (train to tune), a blank-rate difference, or a different effect size.
3. **keep** if nothing is flagged.

"Explained by history length" is only applied to features built from past starts (`hf_`, `jt_`, `jtp_`, `td_`, `w3_`). Field size, for example, cannot be explained that way.

## 6. Results, feature by feature

### Review (2)

| Feature | What happened |
|---|---|
| `jt_trainer_win365` | **Shape changed:** PSI 0.288 against Kaggle train, 0.292 against the history-matched window. HKJC trainer rates are bunched closer together: 10th to 90th percentile 0.055-0.115, against 0.049-0.137 in Kaggle train. **Effect much weaker:** coefficient +0.291 ± 0.025 (Kaggle train), +0.332 ± 0.043 (Kaggle early), **+0.092 ± 0.071** (HKJC Stage 2). It even drifts inside Kaggle (PSI 0.267 train to tune). |
| `jt_trainer_win365_rel` | Same story (PSI 0.264; coefficient +0.281 to +0.088). |

The rank version, `jt_trainer_win365_rk`, has a stable shape by construction, but its effect is also weaker (+0.298 to +0.084), even against the history-matched window. So the problem is not only the scale: **the trainer signal itself is weaker in HKJC 2025-26.**

Why it matters: the conditional logit learned a trainer weight of about +0.11 from Kaggle (Fahim's coefficients). In HKJC that weight may be too high. This may be one reason the model's view runs against the market in Stage 2, but the audit cannot prove that.

### Watch (8)

| Feature | What happened | Likely cause |
|---|---|---|
| `jt_jockey_win365`, `_rel` | Shape shift (PSI 0.214 / 0.196), drift inside Kaggle (PSI 0.250). Effect +0.487 to +0.378. | Against the history-matched window the effect difference is not significant (+0.448 to +0.378), so mostly time and history, not a different era |
| `jt_jockey_win365_rk` | Effect +0.551 to +0.306 | Not significant against the history-matched window (+0.401) |
| `jt_trainer_win365_rk` | Effect +0.298 to +0.084 | Real, see Review above |
| `hf_win5_s` | PSI 0.312 against Kaggle train, but 0.012 against the history-matched window | **History length.** It is a rate over the last 5 starts, and HKJC horses have fewer starts so far |
| `hf_days_since` | SMD -0.18 against Kaggle train, +0.03 against the history-matched window | **History length.** It is capped at 365 days and HKJC history is only months long |
| `hf_class_change` | Drift inside Kaggle (PSI 0.122 train to tune), stable for HKJC (PSI 0.094) | Prize levels change between seasons |
| `n_runners` | SMD +0.24: HKJC fields are bigger (the Week 2 EDA found 37.1% full fields vs 28.7%) | **A real difference**, not history. It is race-level (cancels in the logit) but LightGBM v2 uses it directly |

### Keep (17)

Draw (`rc_draw_pct`, `rc_draw_x_hv`, `rc_draw_x_sprint`), weight (`rc_weight_rel`, `rc_weight_x_dist`), finishing form (`hf_last_fin_pct`, `hf_avg3_fin_pct` and their `_rel` / `_rk` versions), `hf_first_start`, `hf_class_change_rk`, `hf_win5_s_rk`, the two `w3_` horse rates, and the two new jockey-trainer pair features. For example, the 3-race average finish has coefficient -0.696 in Kaggle and -0.642 in HKJC, very close.

**New on 4 Nov evening: jockey-trainer pair features** (Fahim, `model_logic/jt_partnership_features.py`)

| Feature | PSI h_s2 | PSI vs early | Coef k_train | Coef k_early | Coef h_s2 |
|---|---|---|---|---|---|
| `jtp_pair_effect365_rel` | 0.047 | 0.009 | +0.334 ± 0.022 | +0.415 ± 0.056 | +0.274 ± 0.060 |
| `jtp_pair_effect_last_50_rel` | 0.044 | 0.014 | +0.350 ± 0.022 | +0.393 ± 0.045 | +0.285 ± 0.056 |

Both keep their shape in HKJC, and their effect is slightly smaller but within error. Unlike the trainer rate on its own, the pair signal carries over. Both also pass the leakage tests (26 of 26), including the source-separation test that matters for the "last 50 rides" version, which has no time window.

### h_test (information only)

Two big differences between `h_test` and `h_s2`: `hf_days_since` (SMD +0.42) and `hf_win5_s` (PSI 0.71). The test window is later in the HKJC data, so horses have longer histories, and it spans the end of one season and the start of the next, so many horses come back from a long break. **This is a description, not a reason to change anything.** Nothing here used test-window results.

## 7. What to do with this (suggestions for the group)

1. **Trainer rate:** in the rolling validation (Job B), compare models with and without the trainer features, judged on `k_tune` as the rules require. Mention the weaker HKJC trainer effect in the report as a limitation of training on 2017-20.
2. **History-length shifts** (`hf_win5_s`, `hf_days_since`, part of the jockey rates): nothing to fix. They come from the HKJC data starting in September 2025. Mention them in the report.
3. **`n_runners`:** HKJC fields really are bigger. Within-race relative or rank versions are safer for LightGBM than the raw count.
4. **New features** (Nayeon's track/distance form, the jockey-trainer pairs): re-run this audit when they arrive. Pair rates are sparse, so expect history-length shifts. Check them against `k_early`.
5. **Rank (`_rk`) features fix the scale, not the meaning.** They removed the shape shift (PSI about 0.002), but not the weaker trainer effect.

## 8. How to run it again

```powershell
cd "C:\Users\Lenovo\Documents\HKU3\ELEC3544\Group project\6"
.\.venv\Scripts\Activate.ps1
cd github\HKJC-Racing
python analysis\feature_stability_audit.py
```

Expected end of output (about 1 minute):

```
status
keep      30
watch     10
review     2
registry disagreements: 0
42 features in about 50-80s -> ...\analysis\feature_stability_results
```

Until 7 Nov this showed 17 disagreements: the 2 trainer features (registry `watch`, audit `review`) and the 15 new v3.1 features, which the registry marked `unaudited`. On 7 Nov the registry statuses were set from this audit (branch `inns/provenance-cleanup`): trainer features `review`; v3.1 features 13 `keep` and 2 `watch` (`tr_days_since_bucket` for a moderate Kaggle-to-HKJC shift, `tr_fin_pct_slope3_rel` for a blank rate 5.6 points higher in HKJC). Status is a label only; no model's feature list changed.

The numbers change whenever Fahim or Nayeon add features. That is expected; check that the new features appear in the report.

Options:
- `--data <parquet>`: another input file.
- `--output-dir <folder>`: another output folder.
- `--all-registry`: also audit the intermediate helper columns in the registry (ride counts, baselines, raw pair rates, history-depth counts) that no model reads directly. Takes about 50-80 seconds. Text columns (`td_distance_band`, v3.1) are listed as skipped, not audited (before 8 Nov this option crashed on that column). Write it to its own folder so the default results stay as they are: `--output-dir analysis\feature_stability_results\all_registry`.

**Where the feature list comes from (changed 6 Nov).** The audit reads Fahim's registry (`model_logic/feature_registry.py`) and joins it with the model scripts' own lists. So a feature added to the registry is audited automatically, as long as the feature builder actually produces it. A feature the registry or a model lists but the builder does not produce is reported in the "not built" line instead of crashing. Run the leakage tests first (`leakage test/README_LEAKAGE_TESTS.md`), then this audit.

**Agreement with the registry.** Each feature in the registry carries its own status (`keep`, `watch`, `review`, `unaudited`). The report has a section listing every feature where the registry and this audit disagree. After the 7 Nov registry update there are none. The audit script never edits the registry; a label change goes through a reviewed commit (the 7 Nov one is in a PR for Fahim, who owns the registry).

**What the `--all-registry` run showed (8 Nov, registry v3.1, 55 features: 33 keep, 20 watch, 2 review).** Results in `feature_stability_results/all_registry/`. Beyond the 42 default features it audits 13 building blocks that no model reads: 10 `jtp_pair_*` helpers and the 3 history-depth counts `td_track_starts`, `td_distance_band_starts`, `td_track_distance_starts`. None is `review`; 1 is `keep` (`jtp_pair_baseline_last_50`) and 12 are `watch`, mostly because HKJC horses and pairs have shorter histories. Two (`jtp_pair_win365_rel`, `jtp_pair_win_last_50_rel`) are marked `keep` in the registry but `watch` here (PSI 0.19 and 0.15); the models read the `*_effect_*_rel` versions, which are `keep` in both. The registry labels of the building blocks are left for Fahim to decide. The default features get the same status in both runs. (6 Nov, registry v3.0: 37 features, 20 keep, 15 watch, 2 review.)

**One status sheet for v3.1 (8 Nov).** `python analysis/stability_v31_summary.py` (after both audits) writes `feature_stability_results/STABILITY_V3_1.md`: an explicit status for every one of the 38 model inputs, the building blocks, Nayeon's history-depth and missingness tables (`analysis/history_depth_results/`, h_test left out), and a proposed treatment rule per status that is not agreed yet. It exits with an error if the three sources use different data or registry hashes, or if the two audit runs disagree.

**Reading the outputs:**

| File | What |
|---|---|
| `stability_report.md` | The table above, sorted review, watch, keep, with reasons |
| `smd_by_feature.png` | Dots per feature: Kaggle tune, HKJC Stage 2, and HKJC Stage 2 against the history-matched window. Dotted lines 0.10, dashed 0.25. |
| `stability_summary.csv` | One row per feature, every number (SMD, PSI, KS, blank rates, coefficients, SEs, z-scores) |
| `window_stats.csv` | Mean, SD, percentiles and blank rate of each feature in each window |
| `audit_meta.json` | Date, data hash, repo commit, windows, thresholds, feature list |

## 9. How I checked the audit itself

- **PSI tested on made-up data:** identical data gives 0; two samples from the same distribution give 0.002; a 0.5-SD shift gives 0.23; a yes/no feature going from 10% to 20% matches the exact formula (0.0841); a jump in blanks from 10% to 30% gives 0.27.
- **A bug found and fixed:** the first version put 0 and 1 in the same bin, so yes/no features (like `hf_first_start`) always showed PSI 0. Fixed before the results above.
- **Trainer flag checked by hand:** the share of trainer rates sitting exactly at the prior is similar in all windows (0.8-1.2%), so the shift is a real narrowing of the spread, not a pile-up at the prior.
- **No test-window results used:** effects are fitted only on `k_train`, `k_early`, `k_tune` and `h_s2` (the code asserts this).
- **Registry change checked (6 Nov):** after switching the feature list to the registry, the default run was compared with the previous one. Same 27 features, maximum numeric difference 0.0, all statuses identical. The only new things are two columns in `stability_summary.csv` (`registry_status`, `status_agrees`) and the registry section in the report.
- **Import fix (4 Nov evening):** after Fahim's builder started importing `jt_partnership_features`, the audit crashed when run from the repo root. The loader now adds both the repo root and `model_logic/` to Python's import path. The same fix was made in `leakage test/test_leakage.py`.

## 10. Limits

- **One feature at a time.** The effect checks ignore correlations between features, so they can differ from the coefficients in the full model.
- **Small HKJC sample.** 423 races in `h_s2` give wide standard errors (for the trainer rate, ±0.071).
- **Thresholds are rules of thumb**, not laws.
- **`k_early` is not perfect.** It matches history length and season months, but it is also older (2017-18) and smaller (4,711 runners).
- **PSI depends on the binning** (10 Kaggle-train deciles).
- **Effects use Stage 2 results.** That is why the flags are warnings, not selection decisions (see section 2).

## 11. Sources (paraphrased)

- PSI rule of thumb (below 0.10 little change, 0.10-0.25 moderate, 0.25 or more significant): [Statistical properties of the population stability index (Western Michigan University)](https://files.wmich.edu/s3fs-public/attachments/u730/2022/PSIfinal.pdf); bands are starting points, not universal law: [Arize, PSI glossary](https://arize.com/glossary/population-stability-index-psi/).
- SMD: good balance at an absolute SMD of 0.25 or less (citing Rubin 2001 and Stuart 2010): [SAS PSMATCH documentation](https://go.documentation.sas.com/api/docsets/statug/v_023/content/statug_psmatch_details22.htm); 0.10 meaningful and 0.25 serious: [magentabook balance table](https://www.rdocumentation.org/packages/magentabook/versions/0.1.1/topics/mb_balance_table).
- KS is oversensitive with large samples: [Drift detection: KS test, PSI, and interpreting signals](https://www.statstest.com/drift-detection-ks-test-psi-interpret-signals).
- Project thresholds: `docs/WEEK3-4_PLAN.md` check 5.5 #7 (SMD above 0.25, `k_train` vs `h_s2`).
