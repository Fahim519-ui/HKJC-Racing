# Not evidence (quarantined 8 Nov 2026)

**What this is:** Fahim's 6 Nov fold and ablation runs (moved here from artifacts/folds/).

**Why it is not evidence:** Every meta.json records registry_version v3.0 and git commit fc759fe, which is not on GitHub. LightGBM used 24 columns (23 distinct features; n_runners twice) while registry v3.1 has 38, so the td_/tr_/int_ features were never in these runs; primary and no_track_distance are identical for that reason.

**What replaces it:** The v3.1 re-run of `python -m model_logic.run_ablation` from a clean, pushed main, written to artifacts/folds/.

These files are kept for the record only. Do not quote their numbers as results, and do not
use them for any freeze or final scoring: `analysis/provenance.py` refuses any run or
predictions file inside this folder (list: `analysis/quarantine.json`).
