# Weeks 1-2 runbook: data cleaning + AWS (Inns, Group 17)

**Project folder:** `C:\Users\Lenovo\Documents\HKU3\ELEC3544\Group project\4`. It has a space in it, so put it in quotes in the terminal.

**Due: Friday 16 Oct 2026.**

- The data is final: the two CSVs as they are, with no more scraping.
- No Data Wrangler report.
- What's been done so far, with results, is in `docs/PROGRESS_SUMMARY.md`.

---

## Where things stand (updated Sat 31 Oct)

**Weeks 1-2 are done and were handed over on Fri 16 Oct.** The work has moved on to Weeks 3-4: see `docs/WEEK3-4_RUNBOOK.md` (how) and `docs/WEEK3-4_PLAN.md` (why). This file stays as the record of Weeks 1-2 and the reference for the AWS setup, the virtual environment and clean-up. Fahim's access: `docs/FAHIM_ACCESS_RUNBOOK.md`.

**Left over from Weeks 1-2:**

1. **Check the findings cell.** On 16 Oct the findings were in cell 2 as a **code** cell. If it still is: click it, Esc then M (Markdown), Shift+Enter, Ctrl+S, then re-upload (5.4). Otherwise "Run All" stops at that cell.
2. **SageMaker budget** (4.2): created by `docs/FAHIM_ACCESS_RUNBOOK.md` Step 3 (US$20/month, covers both spaces).
3. **Clean up the accidental Unified Studio setup** (Costs / clean-up section). Ask for exact steps first.

The table and checklist below are the record as of Thu 15 Oct; the last items were completed on Fri 16 Oct.

| Day | Step | Status |
|---|---|---|
| Tue 13 | 1. Set up the folder and Python | **Done** |
| Tue 13 | 2. Run the cleaning | **Done.** All checks passed. |
| Tue 13 | 3. Check the result and share it | **Done.** Output verified; 4 files sent to the group. |
| Wed 14 - Thu 15 | 4. AWS: S3 bucket + SageMaker | **Done.** SageMaker reproduced the laptop table (`same table: True`). Space stopped. |
| Thu 15 - Fri 16 | 5. EDA notebook + findings | **Done.** All 11 cells ran, numbers checked, 5 figures + notebook in S3; findings saved and PDF made 16 Oct. |
| Fri 16 | Hand-over | **Done.** Message sent (`docs/PROGRESS_SUMMARY.md`, section 12). |
| After Fri | SageMaker budget (4.2): US$20 budget created with Fahim's access (`FAHIM_ACCESS_RUNBOOK.md` Step 3); clean up the accidental Unified Studio setup | Budget: 31 Oct. Clean-up: to do |

**AWS resources (all in Singapore, `ap-southeast-1`)**

| Resource | Name | Notes |
|---|---|---|
| S3 bucket | `inns-elec3544-dsfm-sagemaker` | private, versioning on, tag `Project = ELEC3544 DSFM Group Project` |
| SageMaker AI domain | `QuickSetupDomain-20261014T163783` (`d-ftva2wtce0s1`) | Quick setup, IAM sign-in, idle shutdown **120 min**, same Project tag |
| JupyterLab space | `inns-elec3544-dsfm-datacleaning` | private, `ml.t3.medium`, 5 GB, SageMaker Distribution 4.5.0. **Stopped.** |
| Not used | Unified Studio domain `SageMakerUnifiedStudio-6nx6o6rfru3dsx-...` (`d-j0t74widgmd8`) and project `admin-project-...` | created by mistake on 13 Oct; still to clean up (31 Oct) |

**Checklist for Friday 16 Oct (all done)**

- [x] `clean_data.py` runs on the laptop with "All checks passed"
- [x] `runners_merged.parquet` / `.csv`, `cleaning_report.md` and `data_dictionary.md` sent to the group
- [x] S3 bucket in Singapore holding `raw/`, `processed/`, `interim/`, `reports/`, `code/`
- [x] SageMaker JupyterLab space (idle shutdown on) re-runs the cleaning and prints `same table: True` (screenshot taken)
- [x] `notebooks/02_eda.ipynb` run, figures saved and uploaded to S3
- [x] Findings written in the notebook, saved, re-uploaded (16 Oct; check the cell is Markdown - see top)
- [x] EDA report PDF made (5.5)
- [x] Hand-over message sent to the group (Fri 16 Oct)

---

## Every time you open the project

1. Open Kiro, then **File > Open Folder > `Group project\4`**.
2. Open a terminal with **Terminal > New Terminal**. It's PowerShell.
3. Activate the environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   The prompt should start with `(.venv)`. If you see "running scripts is disabled", run `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.
4. For any AWS command, sign in first: `aws sso login --profile cloudpc`. Not `aws login`, which is a different command that signs in your *default* profile. The AWS CLI is a separate Windows program, so it works with or without `(.venv)`.

`.venv\Scripts\activate` (no extension) is the version for Linux/Mac terminals. Don't open or edit it, or anything else in `.venv`.

### How the virtual environment works

`.venv` is a private Python install for this project only:

```
.venv\
  pyvenv.cfg            <- points to the base Python 3.14.7 installed on this PC
  Scripts\python.exe    <- runs that Python, but with THIS folder's packages
  Scripts\pip.exe, jupyter.exe, Activate.ps1
  Lib\site-packages\   <- pandas, pyarrow, matplotlib, jupyterlab ... (~125 MB)
```

- **Activating changes only the current terminal.** `.venv\Scripts` goes first on the PATH, so `python` and `pip` mean this project's copies, and the prompt shows `(.venv)`. Close the terminal, or type `deactivate`, to undo it.
- **Without activation, `python` is your global Python**, which has no pandas, so you get `ModuleNotFoundError: No module named 'pandas'`. You can skip activating by calling `.\.venv\Scripts\python.exe clean_data.py` instead.
- **Why use it:**
  - projects can't clash over package versions;
  - your global Python stays clean;
  - `requirements-cleaning.txt` rebuilds the same environment on any PC.
- **Kiro** uses it once you pick `.venv\Scripts\python.exe` as the interpreter.
- **SageMaker doesn't use it.** It has its own Python (3.12, pandas 2.3.3), which is why 4.8 re-runs the cleaning there and compares the result.
- **It's tied to this folder's path.** If you move or rename folder `4`, delete `.venv` and recreate it (see "To redo it" below). Never upload `.venv` to GitHub.

---

## Steps 1-3: done on 13 Oct (record, and how to redo them)

### What was done

| Step | Result |
|---|---|
| Kaggle file | Renamed to `data\Horse_Racing_Results_2017-2020.csv` |
| Python | 3.14.7 in `.venv`. `Activate.ps1` signature: **Valid**, Python Software Foundation. |
| Packages | `pip install -r requirements-cleaning.txt`: pandas 3.0.6, pyarrow 26.0.0, matplotlib 3.11.2, jupyterlab 4.6.4. `pip check`: no broken requirements. |
| `python clean_data.py` | `hkjc: 12,202 rows, 964 races (963 valid)`, `kaggle: 27,008 rows, 2224 races (2216 valid)`, `All checks passed` |
| Output check | Identical, value for value, to the reference runs (pandas 3.0.6, and pandas 2.2 on Python 3.12). Two HKJC 2025-26 races matched the official results pages exactly. |
| Shared | `runners_merged.parquet`, `runners_merged.csv`, `data_dictionary.md`, `cleaning_report.md` |

### To redo it (new PC, or if something breaks)

In the project folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cleaning.txt
python clean_data.py
```

### Output files

| File | Type | Size | Use |
|---|---|---|---|
| `data\processed\runners_merged.parquet` | Parquet | 1.1 MB | **Main output**: 39,210 rows x 51 columns. For Python: `pd.read_parquet(...)` |
| `data\processed\runners_merged.csv` | CSV | 12.8 MB | Same table, for Excel. View only, don't re-save. |
| `data\interim\hkjc_2025_26_clean.*` / `kaggle_2017_2020_clean.*` | Parquet + CSV | | One source each |
| `reports\cleaning_report.md` | Markdown | small | Counts, flagged races, missing values, name links, market check |

---

## Step 4 - AWS: S3 bucket + SageMaker (done 14-15 Oct)

**Region for everything in this project: Asia Pacific (Singapore), `ap-southeast-1`.**

- Singapore has every SageMaker feature the group might want later, including Canvas, which isn't offered in Thailand or Hong Kong.
- Your CLI profile `cloudpc` may default to Thailand (`ap-southeast-7`), so every command below includes `--region ap-southeast-1`.

### 4.1 Sign in from the Kiro terminal

With `(.venv)` active and in the project folder:

```powershell
aws sso login --profile cloudpc
aws sts get-caller-identity --profile cloudpc     # shows your account ID; note it down
```

### 4.2 Budget alert (console, 5 min)

This account also runs your other projects, so give SageMaker its own budget:

1. Go to **Billing and Cost Management > Budgets > Create budget > Customize (advanced) > Cost budget**.
2. Set it to **Monthly**, US$10.
3. Under **Filters**, set **Service = Amazon SageMaker**.
4. Add alerts at 80% (actual) and 100% (forecasted) to your email.

### 4.3 Create the S3 bucket

**Done 15 Oct.**

The bucket name must be globally unique and contain the word `sagemaker` (anywhere in the name): SageMaker's managed policy only gives S3 access to buckets with "sagemaker" in the name. The name and tag follow the same style as your other projects (`inns-...`, Project tag in words).

```powershell
$B = "inns-elec3544-dsfm-sagemaker"
aws s3api create-bucket --bucket $B --region ap-southeast-1 --create-bucket-configuration LocationConstraint=ap-southeast-1 --profile cloudpc
aws s3api put-bucket-versioning --bucket $B --versioning-configuration Status=Enabled --region ap-southeast-1 --profile cloudpc
aws s3api put-bucket-tagging --bucket $B --tagging 'TagSet=[{Key=Project,Value=ELEC3544 DSFM Group Project}]' --region ap-southeast-1 --profile cloudpc
aws s3api get-bucket-tagging --bucket $B --region ap-southeast-1 --profile cloudpc
```

- New buckets are private and encrypted by default. Leave those settings alone.
- `$B` only exists in this terminal. In a new terminal, set it again: `$B = "inns-elec3544-dsfm-sagemaker"`.

**The same Project tag on the SageMaker domain and space** (done for the domain, optional for the space):

```powershell
$DOMARN = aws sagemaker describe-domain --domain-id d-ftva2wtce0s1 --query DomainArn --output text --region ap-southeast-1 --profile cloudpc
aws sagemaker add-tags --resource-arn $DOMARN --tags 'Key=Project,Value=ELEC3544 DSFM Group Project' --region ap-southeast-1 --profile cloudpc
$SPACEARN = aws sagemaker describe-space --domain-id d-ftva2wtce0s1 --space-name inns-elec3544-dsfm-datacleaning --query SpaceArn --output text --region ap-southeast-1 --profile cloudpc
aws sagemaker add-tags --resource-arn $SPACEARN --tags 'Key=Project,Value=ELEC3544 DSFM Group Project' --region ap-southeast-1 --profile cloudpc
```

### 4.4 Upload the data, outputs and code

**Done 15 Oct** (12 files). Run these from the project folder:

```powershell
$R = "--region", "ap-southeast-1", "--profile", "cloudpc"
aws s3 cp data\hkjc_results_2025-07-26_to_2026-10-11.csv "s3://$B/raw/" @R
aws s3 cp data\Horse_Racing_Results_2017-2020.csv       "s3://$B/raw/" @R
aws s3 cp data\processed "s3://$B/processed/" --recursive @R
aws s3 cp data\interim   "s3://$B/interim/"   --recursive @R
aws s3 cp reports        "s3://$B/reports/"   --recursive @R
aws s3 cp clean_data.py              "s3://$B/code/" @R
aws s3 cp requirements-cleaning.txt  "s3://$B/code/" @R
aws s3 cp notebooks "s3://$B/code/notebooks/" --recursive @R
aws s3 ls "s3://$B/" --recursive @R
```

`@R` adds the region and profile to each command. Like `$B`, it only exists in this terminal.

**Bucket layout:**

| Prefix | What |
|---|---|
| `raw/` | the two input CSVs, never changed (the source of truth) |
| `interim/` | one clean table per source |
| `processed/` | the merged table, `runners_merged.parquet` |
| `reports/` | the cleaning report and `figures/` (the 5 EDA charts) |
| `code/` | script and notebook, to copy into SageMaker |

### 4.5 Create the SageMaker AI domain (console, one-time, about 10 min)

**Done 14 Oct:** `QuickSetupDomain-20261014T163783` (`d-ftva2wtce0s1`).

1. At the top right of the console, switch the region to **Asia Pacific (Singapore)**.
2. Search for **SageMaker AI** and open the **Amazon SageMaker AI** console.
   - **Not** "Amazon SageMaker". That's the newer Unified Studio, and its address contains `/datazone/`.
   - If you see "Setup Amazon SageMaker Unified Studio", you're on the wrong page: choose **Cancel**.
3. On the SageMaker AI landing page, under **Get started**, choose **Set up SageMaker Studio**. That's the single-user Quick setup.
   - The same thing is under **Environment configuration > Domains > Create domain > Set up for single user (Quick setup)**.
   - Don't use **Advanced setup for organizations**.
4. Go to **Environment configuration > Domains** and wait a few minutes until the status is **InService**.

Quick setup creates:

- a domain (name like `QuickSetupDomain-<date>`), using IAM authentication;
- a user profile, `default-<date>`;
- an execution role with `AmazonSageMakerFullAccess`, `AmazonS3FullAccess` and the Canvas policies;
- its own S3 bucket, `sagemaker-studio-...`. Leave that bucket alone.

Why IAM sign-in and not Identity Center: a domain that uses Identity Center sign-in must be in the same region as Identity Center, and yours is in Thailand. You sign in to the console as usual and open Studio from there.

### 4.6 Turn on idle shutdown (before creating the space)

1. Open the domain, then the **App Configurations** tab > **JupyterLab** > **Edit**.
2. Tick **Enable idle shutdown** and type the minutes (range 60 to 525,600). **Set to 120 on 15 Oct.** If you leave it blank, it defaults to 10,080 minutes (7 days).
3. Choose **Submit**.

- A space counts as idle when no kernel or terminal is running.
- Idle shutdown needs the SageMaker Distribution image version 2.0 or later. The latest image is fine.
- Idle shutdown is only a safety net. You still stop the space yourself (4.9).

### 4.7 Open Studio and create a JupyterLab space

1. **Environment configuration > Domains >** your domain **> User profiles >** `default-...` **> Launch > Studio**. Or: **Applications and IDEs > SageMaker Studio** in the left menu.
2. In Studio, go to **Applications > JupyterLab > + Create JupyterLab space**.
3. Name it `inns-elec3544-dsfm-datacleaning`, leave **Share with my domain** unticked (private), and choose **Create space**.
   - Private means only your user profile can see, open and run it. If you later want to co-edit with Fahim, create a separate shared space.
4. On the space page, set:
   - **Instance:** `ml.t3.medium`, the cheapest and enough for 39k rows
   - **Image:** SageMaker Distribution, latest version
   - **Storage:** 5 GB
5. Leave **Remote access** off. The "Open in Kiro / VS Code / Cursor" buttons aren't needed.
6. Choose **Run space**, wait about 1-2 minutes, then choose **Open JupyterLab**.
7. In JupyterLab, ignore the built-in **Kiro chat panel** (close it; don't run `kiro-cli login`) and answer **No** to the Jupyter news pop-up.

### 4.8 Re-run the cleaning in SageMaker and compare

In JupyterLab, go to **File > New > Terminal**. This is a Linux (bash) terminal, and the AWS CLI is already installed in the image.

```bash
B=inns-elec3544-dsfm-sagemaker
mkdir -p ~/dsfm && cd ~/dsfm
aws s3 cp s3://$B/code/ . --recursive           # script + notebook
aws s3 cp s3://$B/raw/ data/ --recursive        # the two raw CSVs
python -c "import sys, pandas, pyarrow; print(sys.version.split()[0], pandas.__version__, pyarrow.__version__)"
python clean_data.py                            # same result lines as on the laptop

# does SageMaker give the same table as the laptop?
aws s3 cp s3://$B/processed/runners_merged.parquet /tmp/laptop.parquet
python -c "import pandas as pd; a=pd.read_parquet('/tmp/laptop.parquet'); b=pd.read_parquet('data/processed/runners_merged.parquet'); pd.testing.assert_frame_equal(a, b); print('same table: True')"
```

**Expected:**

- The version line shows `3.12.14 2.3.3 21.0.0` (SageMaker Distribution 4.5.0). Those are different versions from your laptop, which is fine.
- Then `All checks passed`.
- Then `same table: True`. **Result on 15 Oct: exactly this.**

**Notes:**

- No login is needed inside Studio. The space uses the domain's execution role.
- You can also upload files with the up-arrow button in JupyterLab's file browser. S3 is just tidier.
- **Screenshot** the terminal showing `All checks passed` and `same table: True`. That's your evidence for Friday.

### 4.9 Stop the space when you finish

Studio > **Compute > Running instances** > the `inns-elec3544-...` row > **Stop**. Or open the space page and stop it there. You should see "Successfully stopped space". **Done 15 Oct.**

- A stopped space costs nothing for compute.
- Its 5 GB disk costs well under US$1 a month and keeps your files.

---

## Step 5 - EDA notebook (done 15 Oct)

`notebooks\02_eda.ipynb` is ready. It reads `runners_merged.parquet` and compares **Kaggle 2017-20** (blue) with **HKJC 2025-26** (orange).

### 5.1 Run it on the laptop

In the Kiro terminal, with `(.venv)` active and in the project folder:

```powershell
jupyter lab
```

1. A browser tab opens. Open `notebooks/02_eda.ipynb`.
2. Choose **Kernel > Restart Kernel and Run All Cells**. It takes under a minute. Every section should show a chart and/or a table, with no red error boxes.
3. The 5 charts are saved to `reports\figures\01...05_*.png`.
4. Write the findings (5.3) in the first cell and save with `Ctrl+S`.
5. When you're done, close the browser tab and press `Ctrl+C` in the terminal to stop Jupyter.

You can also open the `.ipynb` directly in Kiro (needs the Jupyter extension; pick `.venv` as the kernel), or in SageMaker JupyterLab (`~/dsfm/notebooks`). The notebook finds the project folder by itself.

### 5.2 What each section shows

| # | Section | What to look at |
|---|---|---|
| 1 | Races per month | The Kaggle gaps (after 16 May 2018, after 21 Jun 2020), and no racing in August |
| 2 | Missing values | Only columns filled in **both** sources can be features on the merged data. The second table shows HKJC's blanks are not random. |
| 3 | Distributions | Odds, field size, weight, age, draw, distance. Different shapes mean drift between periods. |
| 4 | Market calibration + log loss | **The baseline the models must beat** |
| 5 | Draw bias | Win rate by draw against 1/field size, at HV 1200/1650 and ST 1200/1400 |
| 6 | Weight and rating vs win rate | Handicap effect |
| 7 | Winning times | Sanity check: the same distance and surface should give the same time |
| 8 | Jockeys and trainers in both periods | How much history the Week 3 features can use |

### 5.3 Findings from my run: check them against yours, then put them at the top of the notebook

Each notebook section ends with a small table that shows these numbers. Yours should match. If one doesn't, tell me.

1. **Coverage.** 3,179 valid races and 38,942 starters.
   - Kaggle: 2,216 races, 2017/18 to 2019/20. The ends of two seasons are missing: after 16 May 2018 and after 21 Jun 2020.
   - HKJC: 963 races, all of 2025/26 plus 98 races of 2026/27.
2. **The baseline to beat** (per-race log loss of the market, lower is better):

   | | Market | Uniform (1/field size) | Favourite wins |
   |---|---|---|---|
   | Kaggle | **1.987** | 2.486 | 31.3% |
   | HKJC | **2.035** | 2.516 | 29.2% |

   The market is close to well calibrated in both periods.
3. **Kaggle odds are capped at 99 until 13 Mar 2019.** That affects 8.7% of 2017/18 runners and 5.4% of 2018/19 runners, and shows as a spike at 99 in the odds histogram. Read "99" as "99 or longer".
4. **The two periods look alike.** Median odds are 15 vs 16; median field size is 12, age 5 and distance 1,400 m in both. Two differences:
   - HKJC fields are full (14 runners) more often: 37% of races vs 29%.
   - Kaggle has more 3-year-olds (11% vs 3%). But HKJC age is blank for 26% of rows, mostly retired horses, so the HKJC age data is biased.
5. **Draw bias at Happy Valley, steady across both periods.** At HV 1200 m and 1650 m:
   - draws 1-2 win about 14-15% of the time;
   - draws 10+ win about 4-5%;
   - the no-bias rate (1/field size) is about 8.5%.

   Sha Tin's bias is weaker: about 9-12% for draws 1-2 vs 6-7% for draws 10+.
6. **Weight.** The win rate rises with carried weight: Kaggle from 5.7% to 9.7%, HKJC from 6.6% to 8.7%. Better horses carry more weight, and the handicap doesn't fully even them out. A higher HKJC rating relative to the race also wins a little more, but that signal is noisy.
7. **Winning times agree** between the sources within about 0.6 s at every common distance (1200 m turf: 69.3 vs 69.7 s), so the time parsing is right. The exception is 2400 m, which has only 3 HKJC races.
8. **People carry over.** 27 of 47 HKJC jockeys and 22 of 45 trainers also appear in 2017-20. They account for 68% and 67% of 2025-26 rides, so history features will exist for most rides, but not all.
9. **HKJC blanks are not random.**
   - 28.5% of 2025/26 starters have no country, age or rating, against 2.9% in 2026/27.
   - These horses won 4.0% of the time vs 9.7%, at median odds of 22 vs 14.
   - They are mostly horses that left Hong Kong racing before the scrape, so the blank carries future information. Don't use it as a feature, and don't drop the rows.

### 5.4 Save and upload

**Figures done 15 Oct. The notebook needs re-uploading once the findings are saved.** Press **Ctrl+S** in Jupyter first. Jupyter keeps running in its own terminal, so use a second terminal (the **+** in Kiro's terminal panel):

```powershell
$B = "inns-elec3544-dsfm-sagemaker"
$R = "--region", "ap-southeast-1", "--profile", "cloudpc"
aws s3 cp reports "s3://$B/reports/" --recursive @R
aws s3 cp notebooks "s3://$B/code/notebooks/" --recursive --exclude "*.ipynb_checkpoints*" @R
aws s3 ls "s3://$B/" --recursive @R
```

The `--exclude` skips Jupyter's automatic backup folder (`.ipynb_checkpoints`). When you're finished, press **Ctrl+C** in the Jupyter terminal and answer `y`.

### 5.5 Export a readable copy (HTML or PDF)

The notebook in Chrome is the real file on your PC (`notebooks\02_eda.ipynb`), and Ctrl+S saves it. Export **after** the findings are saved.

**Recommended (no installs): HTML report without code, then print it to PDF.** In a terminal with `(.venv)` active, in the project folder:

```powershell
jupyter nbconvert --to html --no-input notebooks/02_eda.ipynb --output 02_eda_report
start notebooks\02_eda_report.html
```

1. The report opens in your browser. Press **Ctrl+P**.
2. Set **Destination** to **Save as PDF**.
3. Under **More settings**, tick **Background graphics** (keeps the table shading).
4. Click **Save** and name it `02_eda_report.pdf`.

The warning "Alternative text is missing on 5 image(s)" is only an accessibility note, so ignore it.

| Other options | Notes |
|---|---|
| HTML with code | Jupyter: **File > Save and Export Notebook As > HTML** (goes to Downloads) |
| Jupyter menu **PDF** | Fails with "Pandoc wasn't found". It needs LaTeX + Pandoc, which are large installs. |
| **WebPDF** (`nbconvert[webpdf]` + Playwright) | Not used. On this laptop, Windows' Application Control blocked `pip.exe`, and Playwright's own Chromium may be blocked the same way. |

---

## Friday - hand-over

1. Send the group the message in `docs/PROGRESS_SUMMARY.md`, section 12, with these attached:
   - the EDA report (PDF or HTML, 5.5), or the 5 PNGs from `reports\figures\`;
   - the SageMaker screenshot (`All checks passed` + `same table: True`).
2. Never share AWS keys, passwords or sign-in links. The bucket name alone is fine.
3. Tick the last checklist item at the top.

---

## Not available (the data is final)

The report and data dictionary list what's missing:

- each HKJC horse's own finish time;
- going, course, LBW and declared weight;
- HKJC profile data for horses that had left HK racing.

If the group ever re-scrapes, the script already picks up a per-horse `Finish Time` column with no code change. The other fields would need a few lines; ask me then.

---

## Giving Fahim access for modelling

**Replaced on 31 Oct 2026 by `docs/FAHIM_ACCESS_RUNBOOK.md`** (step by step, tested files in `aws/fahim/`). Decisions:

- Identity Center user `fahim.ahmed` (`fahim.ahmed@connect.hku.hk`), permission set `ELEC3544-DSFM-Fahim` (12-hour sessions, the AWS maximum); access kept for about a month, removed around Mon 30 Nov.
- His own SageMaker user profile `fahim` and **private** space, with a scoped execution role `inns-elec3544-dsfm-fahim-role`: project bucket only, his own spaces only, `ml.t3.medium` up to `ml.t3.xlarge`; no training jobs, endpoints or Canvas. (The earlier draft here used `AmazonSageMakerFullAccess`, which reaches every bucket with "sagemaker" in its name and any instance type.)
- A lifecycle script copies the project files from S3 into his space at each JupyterLab start.
- US$20/month SageMaker budget alert.

---

## Costs and cleaning up

| Item | Cost |
|---|---|
| S3, about 50 MB | well under US$0.01 a month |
| JupyterLab `ml.t3.medium` | a few US cents per running hour (the SageMaker free tier may cover it, if your account is eligible) |
| Space storage, 5 GB | under US$1 a month while the space exists |
| Canvas / Data Wrangler | not used. Don't open it: US$1.90 an hour while you're logged in. |

Keep the bucket private, and never put AWS keys in files, chats or GitHub.

**Still to do (31 Oct): the unfinished SageMaker Unified Studio setup (13 Oct).**

- A one-click Unified Studio setup was started by mistake. The console showed "Polling timed out after 60 attempts", but on 14 Oct the domain and project (`admin-project-...`) turned out to exist. Its default execution role is very broad (`s3:*` on all buckets), and it may also have created a default S3 bucket and the S3 Tables integration.
- It isn't needed: this project uses SageMaker AI.
- Check the Amazon SageMaker (Unified Studio) / DataZone console in Singapore, and remove whatever was created. Ask me for the exact steps first.

**At the end of the course:**

1. Delete the spaces.
2. Delete the user profiles.
3. Delete the domain.
4. Empty and delete both buckets (`inns-elec3544-dsfm-sagemaker` and `sagemaker-studio-...`), including old versions.
5. Fahim's access: see "Removing his access" in `docs/FAHIM_ACCESS_RUNBOOK.md` (around 30 Nov).

---

## When you upload to GitHub later

- **Upload:**
  - `clean_data.py`, `requirements-cleaning.txt`, `README_cleaning.md`;
  - the `docs\`, `notebooks\`, `reports\` and `.kiro\` folders;
  - the two raw CSVs (3.3 MB and 1.8 MB).
- **Don't upload:**
  - the `.venv` folder (hundreds of MB);
  - `data\interim` and `data\processed`, which one command rebuilds. Share the Parquet directly instead.
  - the filled-in files in `aws\fahim\` (`exec-role-policy.json`, `permission-set-policy.json`, `user-settings.json`, `budget-notifications.json`): they contain your account ID. The `.template.json` files are fine. `.gitignore` already lists them.
- GitHub's web upload accepts files up to 25 MB each.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` / "running scripts is disabled" | `Set-ExecutionPolicy -Scope Process Bypass` (this terminal only), then activate again |
| The prompt has no `(.venv)` | Run `.\.venv\Scripts\Activate.ps1`. Every new terminal needs it. |
| `No Kaggle 2017-2020 file found` | The Kaggle CSV must be in `data\` with "2017" and "2020" in its name |
| `CHECKS FAILED` | Read the listed check. Usually an input file changed. Send me the output. |
| `ExpiredToken` / `Token has expired` | `aws sso login --profile cloudpc` |
| `The config profile (cloudpc) could not be found` | Run `aws configure list-profiles` to check the profile name |
| `$B` or `@R` is empty (an S3 path like `s3:///raw/`) | New terminal: set `$B` (4.3) and `$R` (4.4) again |
| You're on "Setup Amazon SageMaker Unified Studio" (address has `/datazone/`) | Wrong product: choose **Cancel**, and don't click **Set up**. Use SageMaker AI (4.5). |
| Quick setup fails with a VPC or subnet error | VPC console (Singapore) > **Actions > Create default VPC**, then try again |
| `AccessDenied` on S3 from SageMaker | The bucket name lacks `sagemaker`, or the role lacks S3 access. Check the role in IAM. |
| `aws: command not found` in SageMaker | It's included in the image, so this shouldn't happen. If it does, use the upload button. |
| Notebook: `StopIteration` on the first cell | Jupyter can't find `clean_data.py`. Start `jupyter lab` from the project folder. |
| Excel shows odd dates or `1.00E+05` | View only. Use the Parquet for analysis. |
| Jupyter export to **PDF**: "500: nbconvert failed: Pandoc wasn't found" | Use HTML + Chrome **Ctrl+P > Save as PDF**, or the WebPDF setup in 5.5 |
| `jupyter lab` doesn't open a browser | Ctrl+click the `http://localhost:8888/lab?token=...` line in the terminal |
| `pip` says "An Application Control policy has blocked this file" | Windows security (Smart App Control / Application Control) blocked the `pip.exe` launcher. Use `python -m pip install ...` instead, which runs the same pip through the signed `python.exe`. Don't turn the Windows protection off. |
| You typed `aws login` by mistake | Harmless for this project; the commands use `--profile cloudpc`. Use `aws sso login --profile cloudpc`. |
| SageMaker JupyterLab shows a Kiro chat asking you to sign in | Close the panel; it isn't needed. Don't run `kiro-cli login` there. |
