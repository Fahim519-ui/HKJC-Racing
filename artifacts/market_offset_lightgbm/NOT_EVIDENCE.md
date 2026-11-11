# Not evidence (quarantined 8 Nov 2026)

**What this is:** 4 Nov market-offset LightGBM (residual edge) diagnostic.

**Why it is not evidence:** The model starts from the market odds (a residual diagnostic, not an independent comparator); the scored period also chose the tree count; pre-registry code; no meta.json. A fair re-check gave about +0.00005 (leakage test/result.md).

**What replaces it:** Nothing: keep as a labelled diagnostic only.

These files are kept for the record only. Do not quote their numbers as results, and do not
use them for any freeze or final scoring: `analysis/provenance.py` refuses any run or
predictions file inside this folder (list: `analysis/quarantine.json`).
