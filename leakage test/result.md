# Results: leakage tests, stability audit, residual-edge check

**Date:** Tue 4 Nov 2026, evening. **By:** Ek-Anankul Wirith (Inns).
**Code checked:** Fahim's repo at commit `b4a355e` ("Add residual market edge evidence"), including the new jockey-trainer pair features.
**Data:** `data/clean data/runners_merged.parquet`. No HKJC test-window (`h_test`) results were used anywhere below.

**In one line:** the features are clean and mostly carry over to HKJC, but once tested fairly, the models still add nothing measurable beyond the odds.

---

## 1. Leakage tests: pass

**26 of 26 pass** (`leakage test/test_leakage.py`, about 15 seconds). Details: `leakage test/README_LEAKAGE_TESTS.md`.

> **Update 6 Nov evening (commit `fcafb9e` + my fix):** both models now build features with `model_logic/feature_builder.py` only, so the checks run against one builder instead of two (that is why the count fell from 33 to 26: 8 duplicate runs removed, 1 new check added that fails if a model stops using the tested builder). The suite now checks **55 features**, including the **18 new v3.1 track/distance, trajectory and interaction features**. **Nayeon's same-day leak is fixed** in the integrated version: the same-day and cut-off tests pass on every `td_`/`tr_`/`int_` feature. Fahim's and Nayeon's own tests in `tests/` also pass (27 of 27).

- No feature uses same-day results.
- No feature mixes Kaggle and HKJC history.
- No feature uses post-race or odds columns.
- **Cut-off check:** no feature leans on a whole-period average that includes the test races. It now passes on Nayeon's integrated `td_*` features.
- Every structural check runs against the builder the models use (`feature_builder.py`); a separate check fails if a model ever builds features another way.
- The two jockey-trainer pair features (`jtp_pair_effect365_rel`, `jtp_pair_effect_last_50_rel`) are covered and pass. The "last 50 rides" version has no time window, so the source-separation test is the one that matters for it, and it passes.
- Every check also catches a deliberately leaky feature (8 controls), so a pass means something. The cut-off control reproduces Nayeon's `td_*` leak and is caught.

## 2. Stability audit: mostly stable, one real problem

42 features checked (registry v3.1): **30 keep, 10 watch, 2 review.** Script and details: `analysis/feature_stability_audit.py`, `analysis/README_FEATURE_STABILITY.md`. Report: `analysis/feature_stability_results/stability_report.md`.

The original 27 features are unchanged by the builder rewrite (every number identical). Of the 15 new model features, 13 are `keep` and 2 are `watch` (`tr_days_since_bucket`: moderate Kaggle-to-HKJC shift; `tr_fin_pct_slope3_rel`: blank rate 5.6 points higher in HKJC). On 7 Nov the registry was updated to these statuses (and the trainer features to `review`), so the registry and the audit now agree.

| What | Result |
|---|---|
| **Trainer win rate** (review) | The one real problem. It predicts winners about 3 times less strongly in HKJC (coefficient +0.09 ± 0.07) than in Kaggle (+0.29 ± 0.03). This holds against a Kaggle window with equally short histories, so it is a real difference between the eras. The model may be leaning on it too much. |
| **New jockey-trainer pair features** (keep) | Carry over well: about +0.27 to +0.29 in HKJC against +0.33 to +0.35 in Kaggle, within error. Shape barely changes (PSI below 0.05). |
| **Form, draw, weight** (keep) | Look and behave the same in both datasets. |
| **Watch features** | Mostly shifted because HKJC horses have shorter histories (the HKJC data starts in Sep 2025). Harmless. One real difference: HKJC fields are bigger (`n_runners`). |

The audit is a warning list, not a rule for dropping features. Features are chosen on `k_tune` only (`docs/DATA_RULES.md`), so settle the trainer question there.

## 3. Fahim's "residual edge" claim: not supported

**What the claim is** (`legacy_code/artifacts/market_offset_lightgbm/residual_edge_evidence.md`): a LightGBM that starts from the market odds and learns corrections from the features improves log loss in all 4 Kaggle walk-forward periods (Aug 2018 - Jul 2020), by +0.0013 on average.

**What I checked:** I re-ran it on the same 4 periods (1,563 races, Kaggle only) and reproduced his numbers exactly. Then I changed one thing at a time:

| Version | Gain over the market (log loss per race) | 95% interval | Real? |
|---|---|---|---|
| Fahim's (the scored period also picks the number of trees) | +0.0013 | −0.0011 to +0.0036 | Optimistic: this setup almost guarantees a gain |
| Fair (tree count picked on training data only) | **+0.00005** | −0.0008 to +0.0009 | No |
| Market with the favourite-longshot correction (fitted on training data) | +0.0009 | −0.0017 to +0.0036 | No |
| Fair model vs the corrected market | **−0.0009** | −0.0036 to +0.0018 | Slightly worse, also not significant |

Intervals come from 2,000 bootstrap resamples of races (seed 17). Positive means better than the comparison.

**Why Fahim's version looks better than it is:**
- Each period stops adding trees at the point where *that same period* scores best. The score at that point is almost always at least as good as the market, so "better in all 4 periods" is nearly built in.
- One period stopped at **1 tree**, which means it found nothing. It still counts as "a gain" (+0.00002).
- Even his own number has an interval that includes zero.

**Plain meaning:** once the test is done fairly, the model adds nothing measurable beyond the odds. That matches every earlier result (the conditional logit's blend gain of +0.0003, the Stage 2 results), so it is not a new finding.

## 4. Two other files in the same batch to fix

1. **`legacy_code/artifacts/market_diagnostics/longshot_bias_evidence.md`** reports market win rates on the **HKJC test window**. These are market numbers, not model scores, but they still look at test results before Week 5. Remove the test rows, or note in the report that they were seen early.
2. **`legacy_code/artifacts/market_offset_feature_families/report.md`** picks the best of 15 feature sets using the same periods that already choose the number of trees. Its winner (track/distance interactions) is likely luck. Re-check it with the tree count picked on training data only before building on it.

## 5. What to tell the group

1. The features are clean and safe to use.
2. The trainer rate is the weak spot. Test removing it in the rolling validation, judged on `k_tune`.
3. "We beat the market" is not supported yet. Fahim should pick the number of trees inside the training data only, and compare against the corrected odds (market-only refit), not the raw odds.
4. Fix or flag the two files in section 4.

## 6. How to reproduce

From `github\HKJC-Racing`, with the project venv active:

```powershell
python -m pytest "leakage test\test_leakage.py" -v  # section 1
python analysis\feature_stability_audit.py          # section 2
```

Section 3 was a one-off check script. It is not saved in the repo. The method: same folds and settings as `legacy_code/model_logic/market_offset_lightgbm.py`, but the number of trees is chosen by early stopping on the last 25% of each training period's race days, then the model is refit on the whole training period and scored once on the validation period. The corrected market is p ∝ market^β with β fitted on each training period (β came out 1.06 to 1.08).
