# Not evidence (quarantined 8 Nov 2026)

**What this is:** 4 Nov market-offset feature-family attribution.

**Why it is not evidence:** Same setup as market_offset_lightgbm, and the best of 15 feature sets was picked on the same periods that chose the tree count; no meta.json.

**What replaces it:** Feature-family ablations of the v3.1 fold runner.

These files are kept for the record only. Do not quote their numbers as results, and do not
use them for any freeze or final scoring: `analysis/provenance.py` refuses any run or
predictions file inside this folder (list: `analysis/quarantine.json`).
