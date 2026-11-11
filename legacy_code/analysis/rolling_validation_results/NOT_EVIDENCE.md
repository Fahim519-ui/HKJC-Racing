# Not evidence (quarantined 8 Nov 2026)

**What this is:** Inns's 6 Nov rolling validation (three Kaggle folds).

**Why it is not evidence:** Uses registry v3.1 features but simplified stand-in models (binary LightGBM, unpenalised logit), not the production models, and writes no meta.json, so its code and data cannot be verified by the provenance check.

**What replaces it:** The v3.1 fold-runner results in artifacts/folds/ (Tier 2 aggregation).

These files are kept for the record only. Do not quote their numbers as results, and do not
use them for any freeze or final scoring: `analysis/provenance.py` refuses any run or
predictions file inside this folder (list: `analysis/quarantine.json`).
