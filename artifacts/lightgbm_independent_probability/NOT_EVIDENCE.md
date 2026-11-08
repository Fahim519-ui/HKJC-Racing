# Not evidence (quarantined 8 Nov 2026)

**What this is:** 3-4 Nov odds-free LightGBM v1 results.

**Why it is not evidence:** Pre-registry feature set, no meta.json, and the run scored h_test (those rows were removed on 7 Nov).

**What replaces it:** The v3.1 fold-runner results; the final workflow is Fahim's frozen-config file (Tier 3).

These files are kept for the record only. Do not quote their numbers as results, and do not
use them for any freeze or final scoring: `analysis/provenance.py` refuses any run or
predictions file inside this folder (list: `analysis/quarantine.json`).
