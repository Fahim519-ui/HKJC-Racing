# HKJC Racing Execution Log

**Date:** 2026-10-12  
**Scope:** Shareable record of observable repository work, commands, generated outputs,
validation, and Git operations. Private reasoning and hidden system/tool internals are excluded.

## Repository and source inspection

```bash
git status --short
find . -maxdepth 2 -type f -not -path './.git/*' | sort | head -200
```

The repository initially had no committed files. The referenced template repository
(`j-csc/HK-Horse-Racing-Data-Scraper`) was inspected for its Python/BeautifulSoup/requests
approach and dependency choices.

Official HKJC sources inspected:

```text
https://racing.hkjc.com/en-us/local/information/resultsall?racedate=2026/10/11&Racecourse=ST
https://racing.hkjc.com/en-us/local/information/localresults?racedate=2026/10/11&Racecourse=ST&RaceNo=1
https://racing.hkjc.com/en-us/local/information/racereportfull?racedate=2026/10/11&Racecourse=ST
https://racing.hkjc.com/en-us/local/information/displaysectionaltime?racedate=11/10/2026&RaceNo=1
https://racing.hkjc.com/en-us/local/information/corunning?Date=20261011&RaceNo=1
https://racing.hkjc.com/racing/information/json/DateList/ResultsAll.aspx?lang=en-us
```

## Implemented meeting-results scraper

Files created:

```text
main.py
requirements.txt
README.md
```

Dependencies:

```text
beautifulsoup4>=4.12,<5
requests>=2.31,<3
```

The scraper gathers:

- Meeting race list
- Race metadata: class, distance, prize money, going, course, race and sectional times
- Runner results: placing, horse, jockey, trainer, weight, declared horse weight, draw,
  beaten margin, running positions, finish time, and odds
- Dividends
- Winning horse breeding
- Replay and sectional-time links
- Comments on running
- Runner sectional records
- Racing incident reports

## Environment setup and first meeting scrape

The system Python rejected a global dependency installation:

```bash
python3 -m pip install --quiet -r requirements.txt
```

Error:

```text
error: externally-managed-environment
```

A project-local environment was then created with `uv` and Python 3.13:

```bash
rm -rf .venv
uv venv .venv --python /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 --quiet
uv pip install --python .venv/bin/python -r requirements.txt --quiet
.venv/bin/python main.py --date 2026/10/11 --racecourse ST
```

Result:

```text
Saved 11 races, 140 runners, and 140 incident records to data.
```

Validation:

```bash
.venv/bin/python -m py_compile main.py
```

Validated result data:

```text
11 races
140 runner features
140 incident records
134 sectional-time records
136 comments-on-running records
```

Generated files:

```text
data/2026-10-11_st_complete.json
data/2026-10-11_st_runner_features.csv
```

## First data commit

```bash
git add data
git commit -m "Add Sha Tin meeting results data" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push -u origin main
```

Published commit:

```text
a247f67 Add Sha Tin meeting results data
```

## Historical CSV exporter

Created:

```text
historical_export.py
```

The exporter:

1. Retrieves the HKJC completed-meeting calendar.
2. Checks Sha Tin (`ST`) and Happy Valley (`HV`) for every calendar date in the requested
   date interval.
3. Retrieves all detailed `localresults` pages.
4. Extracts the requested race, runner, jockey, trainer, and identifier fields.
5. Retrieves per-horse rating history to obtain historic final rating (`FGrating`).
6. Retrieves country of origin and current age, then calculates age for the relevant season.
7. Writes one flat CSV with these 21 columns:

```text
Dato
Track
Race Number
Distance
Surface
Prize money
Starting position
Jockey
Jockey weight
Country
Horse age
TrainerName
Race time
Path
Final place
FGrating
Odds
RaceType
HorseId
JockeyId
TrainerID
```

One-meeting validation:

```bash
.venv/bin/python historical_export.py \
  --start-date 2026/10/11 \
  --end-date 2026/10/11 \
  --output /tmp/hkjc-test.csv \
  --workers 6
.venv/bin/python -m py_compile historical_export.py
```

Result:

```text
140 rows plus CSV header
All 21 requested columns present
Historic rating extraction verified
```

Full-range export:

```bash
.venv/bin/python historical_export.py \
  --start-date 2025/07/26 \
  --end-date 2026/10/11 \
  --output data/hkjc_results_2025-07-26_to_2026-10-11.csv \
  --workers 6
```

Exporter completion:

```text
Completed 336 meeting-track checks.
Completed 1,506 horse profiles.
Saved 12,202 rows to data/hkjc_results_2025-07-26_to_2026-10-11.csv.
```

Final validation:

```bash
python3 - <<'PY'
import csv
from collections import Counter
from datetime import datetime

path = "data/hkjc_results_2025-07-26_to_2026-10-11.csv"
with open(path, newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

headers = list(rows[0])
dates = [datetime.strptime(row["Dato"], "%Y/%m/%d").date() for row in rows]

assert len(rows) == 12202
assert len(headers) == 21
assert min(dates).isoformat() == "2025-09-07"
assert max(dates).isoformat() == "2026-10-11"
assert all(row["Track"] in {"Sha Tin", "Happy Valley"} for row in rows)

print(len(rows), len(headers), min(dates), max(dates), Counter(row["Track"] for row in rows))
PY
```

Validation result:

```text
12,202 rows
21 requested columns
Coverage: 2025-09-07 through 2026-10-11
98 completed meeting dates
Sha Tin: 7,956 rows
Happy Valley: 4,246 rows
```

The requested date range began on 2025-07-26. HKJC returned no completed meetings until
2025-09-07 because of the seasonal break.

## Historical CSV commit

```bash
git add data/hkjc_results_2025-07-26_to_2026-10-11.csv
git commit -m "Add historical HKJC results CSV" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push
```

Published commit:

```text
42c5c4c Add historical HKJC results CSV
```

## Transcript publishing

Created and committed:

```text
chat-transcript-2026-10-12.md
```

Initial push was rejected because the remote had received a new commit:

```text
! [rejected] main -> main (fetch first)
```

Resolved with:

```bash
git pull --rebase origin main
git push
```

Published commit:

```text
08248e0 Add project chat transcript
```

## Current published data assets

```text
data/2026-10-11_st_complete.json
data/2026-10-11_st_runner_features.csv
data/hkjc_results_2025-07-26_to_2026-10-11.csv
chat-transcript-2026-10-12.md
execution-log-2026-10-12.md
```
