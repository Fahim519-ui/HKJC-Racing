# Giving Fahim AWS access for Week 4 - step by step (AWS CLI)

**Written:** Sat 31 Oct 2026 (CLI version, checked against current AWS docs and API models).
**Who runs it:** Inns, on the laptop, in the Kiro terminal (PowerShell), project folder `Group project\4`.
**Time:** about 45 minutes, including the test.

## What you're building

Fahim does **Week 4 (models)** in SageMaker, in **his own private JupyterLab space**. The project files are already copied into it, and he can't reach anything else in your account. His access lasts about a month; remove it around **Mon 30 Nov** (a reminder is set in the chat).

```
Fahim's browser
   │  signs in to your AWS access portal (Identity Center, Thailand)
   ▼
Permission set "ELEC3544-DSFM-Fahim"  ── can only: open Studio as "fahim", see the project bucket
   │
   ▼
SageMaker domain d-ftva2wtce0s1 (Singapore) → user profile "fahim"
   │  his notebooks run as role "inns-elec3544-dsfm-fahim-role"
   ▼
His private JupyterLab space (ml.t3.medium … ml.t3.xlarge)
   │  at every start, script "inns-elec3544-dsfm-fahim-sync" copies files from S3 into ~/dsfm
   ▼
s3://inns-elec3544-dsfm-sagemaker   (the only bucket he can read/write)
```

**What he can and can't do**

| Can | Can't |
|---|---|
| Open Studio as `fahim` | open your profile or your space |
| Create, run, stop and delete **his own private** spaces | create shared spaces |
| Use `ml.t3.medium`, `ml.t3.large`, `ml.t3.xlarge` | any other instance type (blocked by the role; also hidden in his Studio) |
| Read, write and delete files in `inns-elec3544-dsfm-sagemaker` | any other bucket; delete old versions (so anything he deletes can be got back); change bucket settings |
| See SageMaker resources (read-only) | training jobs, endpoints, Canvas, IAM, billing |

## Can it all be done with the AWS CLI?

Almost. Only one step needs the console:

| Step | CLI? |
|---|---|
| Upload files, role, copy script, user profile, budget | Yes |
| Permission set + its policy | Yes (`aws sso-admin`) |
| Giving / removing access (assignments) | Yes (`aws sso-admin`) |
| **Creating Fahim's Identity Center user** | **Console (one screen).** A user created with `aws identitystore create-user` gets **no invitation email** and has no password. Fixing that needs a console-only setting (Settings → Authentication → "Send email OTP") or a console password reset. The console's **Add user** sends the invitation directly, so it's simpler. |
| Testing that it works | Browser (you open Studio as `fahim`) |

## Files (all in `aws\fahim\`)

| File | What it's for | Step |
|---|---|---|
| `exec-role-trust.json` | lets SageMaker use Fahim's role | 2.1 |
| `exec-role-policy.template.json` | what his notebooks can do (project bucket, his own private spaces, t3 up to xlarge) | 2.1 |
| `lcc-fahim-sync.sh` | copies the project files into his space at each JupyterLab start | 2.2 |
| `user-settings.template.json` | his user profile: role, default `ml.t3.medium`, the copy script | 2.3 |
| `user-settings-ui.json` | hides the instance types he can't use (and Canvas, RStudio, Studio Classic) in his Studio | 2.4 |
| `budget.json`, `budget-notifications.template.json` | US$20/month SageMaker budget with email alerts | 3 |
| `permission-set.json` | the permission set's name, 12-hour session, start page and tag | 4 |
| `permission-set-policy.template.json` | what he can do in the AWS console | 4 |

`.template` files contain `<ACCOUNT_ID>`, `<LCC_ARN>` or `<EMAIL>`. The commands fill them in and save the real file next to the template. The filled-in files contain your account ID: **don't put them on GitHub** (`.gitignore` already excludes them).

## Overview

| Step | What | Where | Time |
|---|---|---|---|
| 0 | Extract, sign in, set variables, pre-flight checks | Laptop | 5 min |
| 1 | Upload the files Fahim needs to S3 | Laptop | 2 min |
| 2 | SageMaker: role, copy script, user profile `fahim`, tidy his Studio | Laptop | 5 min |
| 3 | US$20 SageMaker budget | Laptop | 2 min |
| 4 | Permission set `ELEC3544-DSFM-Fahim` | Laptop | 3 min |
| 5 | **Test it as yourself**, then remove your test access | Laptop + browser | 15 min |
| 6 | Create Fahim's user (console), give him access, send the message | Browser + laptop | 5 min |
| 7 | Final check | Laptop | 2 min |
| - | Later: remove his access (~30 Nov) | Laptop + console | 10 min |

**How to read each step:** run the commands in order, then compare with **Expect**. If something doesn't match, stop and check **Troubleshooting** at the end (or send me the output). Commands that only *read* are marked "(read-only)".

---

## Step 0 - Get ready (laptop)

### 0.1 Extract the update

Extract `update_2026-10-31_fahim_cli.zip` into `C:\Users\Lenovo\Documents\HKU3\ELEC3544\Group project\4` → **Replace all**.

### 0.2 Sign in and set the variables

Open Kiro → Terminal. Paste this whole block:

```powershell
cd "C:\Users\Lenovo\Documents\HKU3\ELEC3544\Group project\4"
aws sso login --profile cloudpc
$P    = "--profile", "cloudpc"
$R    = "--region", "ap-southeast-1", "--profile", "cloudpc"   # Singapore: SageMaker, S3
$RI   = "--region", "ap-southeast-7", "--profile", "cloudpc"   # Thailand: Identity Center
$B    = "inns-elec3544-dsfm-sagemaker"
$DOM  = "d-ftva2wtce0s1"
$F    = "aws/fahim"
$TAG  = 'Key=Project,Value=ELEC3544 DSFM Group Project'
$ROLE = "inns-elec3544-dsfm-fahim-role"
$LCCN = "inns-elec3544-dsfm-fahim-sync"
$PSN  = "ELEC3544-DSFM-Fahim"
$ACCT = aws sts get-caller-identity --query Account --output text @P
$INST = aws sso-admin list-instances --query "Instances[0].InstanceArn" --output text @RI
$IDS  = aws sso-admin list-instances --query "Instances[0].IdentityStoreId" --output text @RI
"ACCT=$ACCT"; "INST=$INST"; "IDS=$IDS"
```

**Expect:**

```
ACCT=<12 digits>
INST=arn:aws:sso:::instance/ssoins-<16 characters>
IDS=d-<10 characters>
```

If any line is empty, the sign-in didn't finish: run `aws sso login --profile cloudpc` again, then the block again.

**If you close the terminal at any point,** run this block again before continuing, plus the "Reload" block at the end of this file.

### 0.3 Pre-flight checks (read-only)

```powershell
aws sagemaker describe-domain --domain-id $DOM --query "[DomainName,Status,AuthMode]" @R
aws sagemaker list-user-profiles --domain-id-equals $DOM --query "UserProfiles[].UserProfileName" @R
aws s3 ls "s3://$B/" @R
aws identitystore list-users --identity-store-id $IDS --query "Users[].[UserName,DisplayName]" --output table @RI
```

**Expect:**

- domain: `QuickSetupDomain-20261014T163783`, `InService`, `IAM`;
- profiles: only your own (no `fahim` yet);
- bucket folders: `code/`, `interim/`, `processed/`, `raw/`, `reports/`;
- a table with **your** Identity Center user. Note its **UserName** (left column).

Now set your own user ID (replace `<your-username>` with the UserName you just saw, keep the quotes):

```powershell
$MYUSER = "<your-username>"
$ME = aws identitystore list-users --identity-store-id $IDS --query "Users[?UserName=='$MYUSER'].UserId" --output text @RI
"ME=$ME"
```

**Expect:** `ME=` followed by an ID like `xxxxxxxxxx-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` or `xxxxxxxx-xxxx-...` (not empty).

---

## Step 1 - Upload the files Fahim needs (laptop)

Everything in `code/docs/` is copied into Fahim's space, so put **only these four files** there. Never the Weeks 1-2 runbook, this file or the chat export.

```powershell
aws s3 cp week34_starter.py          "s3://$B/code/" @R
aws s3 cp docs/WEEK3-4_PLAN.md        "s3://$B/code/docs/" @R
aws s3 cp docs/WEEK4_FAHIM.md         "s3://$B/code/docs/" @R
aws s3 cp docs/data_dictionary.md     "s3://$B/code/docs/" @R
aws s3 cp docs/DATA_RULES.md          "s3://$B/code/docs/" @R
aws s3 ls "s3://$B/code/" --recursive @R
aws s3 ls "s3://$B/processed/" @R
aws s3 ls "s3://$B/reports/" --recursive @R
```

**Expect:**

| Key | Note |
|---|---|
| `code/week34_starter.py` | about 13,138 bytes (the 31 Oct version) |
| `code/docs/` | exactly 4 files: `DATA_RULES.md`, `WEEK3-4_PLAN.md`, `WEEK4_FAHIM.md`, `data_dictionary.md` |
| `code/notebooks/02_eda.ipynb` | from Week 2; **no** `.ipynb_checkpoints` copy |
| `processed/runners_merged.parquet` | **1,122,677** bytes |
| `reports/cleaning_report.md` + `reports/figures/` (5 PNGs) | from Week 2 |

If `code/docs/` has anything else: `aws s3 rm "s3://$B/code/docs/<name>" @R`.

---

## Step 2 - SageMaker: Fahim's role, copy script and user profile (laptop)

### 2.1 Execution role (what his notebooks run as)

```powershell
aws iam create-role --role-name $ROLE --assume-role-policy-document "file://$F/exec-role-trust.json" --description "ELEC3544 DSFM - SageMaker execution role for Fahim (project bucket only, t3 up to xlarge)" --tags $TAG @P
(Get-Content "$F/exec-role-policy.template.json" -Raw).Replace("<ACCOUNT_ID>", $ACCT) | Set-Content "$F/exec-role-policy.json" -Encoding ascii
aws iam put-role-policy --role-name $ROLE --policy-name fahim-studio-and-project-bucket --policy-document "file://$F/exec-role-policy.json" @P
aws iam list-role-policies --role-name $ROLE @P
aws iam list-attached-role-policies --role-name $ROLE @P
```

**Expect:** `create-role` prints the role (ARN ends in `role/inns-elec3544-dsfm-fahim-role`). `list-role-policies` shows `fahim-studio-and-project-bucket`. `list-attached-role-policies` shows an **empty** list. That's intended: no `AmazonSageMakerFullAccess`, no `AmazonS3FullAccess`.

### 2.2 Copy script (lifecycle configuration)

```powershell
$sh  = (Get-Content "$F/lcc-fahim-sync.sh" -Raw).Replace("`r`n", "`n")
$LCC = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($sh))
$LCC_ARN = aws sagemaker create-studio-lifecycle-config --studio-lifecycle-config-name $LCCN --studio-lifecycle-config-content $LCC --studio-lifecycle-config-app-type JupyterLab --tags $TAG --query StudioLifecycleConfigArn --output text @R
"LCC_ARN=$LCC_ARN"
```

**Expect:** `LCC_ARN=arn:aws:sagemaker:ap-southeast-1:<account>:studio-lifecycle-config/inns-elec3544-dsfm-fahim-sync`.

What it does on each JupyterLab start (about 2 MB, a few seconds):

- **Every start:** copies `processed/`, `features/` (Nayeon's, into `data/features_week3/`) and `code/docs/` into `~/dsfm`.
- **First start only:** also copies the starter script, the cleaning report, the figures and the EDA notebook.

It never overwrites his own work and never stops JupyterLab from starting. Its log is in `~/.dsfm_sync.log` and in CloudWatch (`/aws/sagemaker/studio`). SageMaker allows lifecycle scripts 5 minutes; this one stops itself well before that.

### 2.3 User profile `fahim`

```powershell
(Get-Content "$F/user-settings.template.json" -Raw).Replace("<ACCOUNT_ID>", $ACCT).Replace("<LCC_ARN>", $LCC_ARN) | Set-Content "$F/user-settings.json" -Encoding ascii
aws sagemaker create-user-profile --domain-id $DOM --user-profile-name fahim --user-settings "file://$F/user-settings.json" --tags $TAG @R
aws sagemaker describe-user-profile --domain-id $DOM --user-profile-name fahim --query "[Status, UserSettings.ExecutionRole, UserSettings.JupyterLabAppSettings.DefaultResourceSpec.InstanceType, UserSettings.JupyterLabAppSettings.LifecycleConfigArns]" @R
```

**Expect:**

- `create-user-profile` prints `UserProfileArn` ending in `user-profile/d-ftva2wtce0s1/fahim`;
- `describe-user-profile`: `InService` (if `Pending`, repeat after 30 s), the role ARN ending in `role/inns-elec3544-dsfm-fahim-role`, `ml.t3.medium`, and the copy script ARN.

If it says the role can't be assumed or doesn't exist, wait 20 s (new roles take a moment to appear everywhere) and run the `create-user-profile` line again.

The domain's 120-minute idle shutdown applies to his space too.

### 2.4 Tidy his Studio (hide what he can't use)

```powershell
aws sagemaker update-user-profile --domain-id $DOM --user-profile-name fahim --user-settings "file://$F/user-settings-ui.json" @R
aws sagemaker describe-user-profile --domain-id $DOM --user-profile-name fahim --query "[Status, length(UserSettings.StudioWebPortalSettings.HiddenInstanceTypes), UserSettings.StudioWebPortalSettings.HiddenAppTypes, UserSettings.ExecutionRole]" @R
```

**Expect:** `InService`, `172`, `["Canvas", "RStudioServerPro", "JupyterServer"]`, and the role ARN is **still** Fahim's (the update only adds settings).

This only changes what he *sees*: he'll only be offered `ml.t3.medium`, `ml.t3.large` and `ml.t3.xlarge`. The role in 2.1 is what actually *blocks* the others. If this command fails, skip it; everything else still works.

---

## Step 3 - US$20 SageMaker budget (laptop)

```powershell
(Get-Content "$F/budget-notifications.template.json" -Raw).Replace("<EMAIL>", "innswirith@gmail.com") | Set-Content "$F/budget-notifications.json" -Encoding ascii
aws budgets create-budget --account-id $ACCT --budget "file://$F/budget.json" --notifications-with-subscribers "file://$F/budget-notifications.json" --region us-east-1 --profile cloudpc
aws budgets describe-budget --account-id $ACCT --budget-name ELEC3544-DSFM-SageMaker-Monthly --query "Budget.[BudgetName,BudgetLimit.Amount,FilterExpression]" --region us-east-1 --profile cloudpc
```

**Expect:** `create-budget` prints nothing (that means success). `describe-budget` shows the name, `20.0` and the filter `SERVICE = Amazon SageMaker`.

You'll get an email at **50%** (US$10) and **100%** of actual spend, and when the month is **forecast** to go over US$20. It covers your space and Fahim's together. Change the email in the first line if your AWS alerts go elsewhere. (Budgets is a global service, which is why this step uses `us-east-1`.)

---

## Step 4 - Permission set `ELEC3544-DSFM-Fahim` (laptop)

```powershell
(Get-Content "$F/permission-set-policy.template.json" -Raw).Replace("<ACCOUNT_ID>", $ACCT) | Set-Content "$F/permission-set-policy.json" -Encoding ascii
$PS = aws sso-admin create-permission-set --instance-arn $INST --cli-input-json "file://$F/permission-set.json" --query PermissionSet.PermissionSetArn --output text @RI
"PS=$PS"
aws sso-admin put-inline-policy-to-permission-set --instance-arn $INST --permission-set-arn $PS --inline-policy "file://$F/permission-set-policy.json" @RI
aws sso-admin describe-permission-set --instance-arn $INST --permission-set-arn $PS --query "PermissionSet.[Name,SessionDuration,RelayState]" @RI
```

**Expect:** `PS=arn:aws:sso:::permissionSet/ssoins-.../ps-...`; then `ELEC3544-DSFM-Fahim`, `PT12H`, and the SageMaker Singapore link.

- **12 hours** is the AWS maximum for one session. His *access* lasts until you remove it; after 12 hours he just signs in again.
- The relay state opens SageMaker in Singapore straight after sign-in.
- The settings come from `permission-set.json` (`--cli-input-json`), because some AWS CLI versions try to *download* any parameter value that starts with `https://`.

---

## Step 5 - Test it as yourself first

You give the permission set to **yourself**, open Studio as `fahim`, check everything, then take it away again. That way Fahim's first try works.

### 5.1 Give yourself the test access (laptop)

```powershell
$REQ = aws sso-admin create-account-assignment --instance-arn $INST --target-id $ACCT --target-type AWS_ACCOUNT --permission-set-arn $PS --principal-type USER --principal-id $ME --query AccountAssignmentCreationStatus.RequestId --output text @RI
aws sso-admin describe-account-assignment-creation-status --instance-arn $INST --account-assignment-creation-request-id $REQ --query "AccountAssignmentCreationStatus.[Status,FailureReason]" @RI
```

**Expect:** `SUCCEEDED` (if `IN_PROGRESS`, run the second line again after 10 s).

### 5.2 Open it in the browser

1. Open a **private/incognito window** (so it doesn't mix with your normal sign-in).
2. Go to your **AWS access portal**, the same sign-in page your `cloudpc` profile uses. To print its link (laptop):

   ```powershell
   Select-String -Path "$HOME\.aws\config" -Pattern "sso_start_url"
   ```

   It's the `https://...awsapps.com/start` address (by default `https://<IDS>.awsapps.com/start`). You can also see it in the IAM Identity Center console → **Dashboard → AWS access portal URL**. Keep it for Fahim's message in 6.3.
3. Sign in as yourself → your account → **ELEC3544-DSFM-Fahim** → **Management console**. It should open SageMaker AI in **Singapore**.

### 5.3 Checklist

| # | Do | Expect |
|---|---|---|
| a | **Applications and IDEs → Studio** → domain `QuickSetupDomain-...` → user profile **fahim** → **Open Studio** | Studio opens. On the left there's no Canvas, RStudio or Studio Classic |
| b | Back in the console, same page, pick **your** user profile → Open Studio | **Access denied** |
| c | In Studio: **JupyterLab → + Create JupyterLab space** → name `test`, sharing **Private** → instance list shows only `ml.t3.medium / large / xlarge` → keep `ml.t3.medium` → **Run space** → **Open JupyterLab** | starts in 2-5 min |
| d | JupyterLab: File → New → Terminal: `ls ~/dsfm ~/dsfm/docs ~/dsfm/data/processed; tail -5 ~/.dsfm_sync.log` | `week34_starter.py`, `data`, `docs` (4 files), `notebooks`, `reports`; the parquet; log ends with `sync done` |
| e | `aws s3 ls s3://inns-elec3544-dsfm-sagemaker/` | lists the folders |
| f | `aws s3 ls` | **AccessDenied** (can't list your other buckets) |
| g | `python -c "import pandas, lightgbm, statsmodels; print(pandas.__version__, lightgbm.__version__, statsmodels.__version__)"` | `2.3.3 4.6.0 0.15.0` |
| h | `cd ~/dsfm && python week34_starter.py` (about 1 min) | ends with `LightGBM per-race log loss: 2.1837 (trees: 102)`, `market on the same races: 1.9947`, `predictions: 10,265 rows` |
| i | `aws s3 cp ~/dsfm/reports/week4_models.md s3://inns-elec3544-dsfm-sagemaker/reports/test_by_inns.md && aws s3 rm s3://inns-elec3544-dsfm-sagemaker/reports/test_by_inns.md` | `upload:` then `delete:` - write access works |
| j | Studio → **Running instances** → stop `test`. Then in the space list: `test` → ⋮ → **Delete** | gone (Fahim starts clean, and its copy script runs again for him) |

(Checks c/g/h: a new space may get SageMaker Distribution 4.6 instead of your 4.5. I checked: both have the same pandas, LightGBM, statsmodels, scikit-learn, numpy and scipy, so the numbers are identical.)

### 5.4 Remove your test access (laptop)

```powershell
$DREQ = aws sso-admin delete-account-assignment --instance-arn $INST --target-id $ACCT --target-type AWS_ACCOUNT --permission-set-arn $PS --principal-type USER --principal-id $ME --query AccountAssignmentDeletionStatus.RequestId --output text @RI
aws sso-admin describe-account-assignment-deletion-status --instance-arn $INST --account-assignment-deletion-request-id $DREQ --query "AccountAssignmentDeletionStatus.[Status,FailureReason]" @RI
aws sagemaker list-spaces --domain-id-equals $DOM --query "Spaces[].[SpaceName,OwnershipSettingsSummary.OwnerUserProfileName,Status]" --output table @R
```

**Expect:** `SUCCEEDED`; the space table shows only **your** space (`inns-elec3544-dsfm-datacleaning`). No `test`.

**If any check in 5.3 failed:** don't do Step 6 yet. Send me the exact error text or a screenshot. Most fixes are one command (for example, adding a missing permission to the role and re-running the `put-role-policy` line from 2.1).

---

## Step 6 - Fahim: create his user, give him access, tell him

### 6.1 Create his Identity Center user (console - the only console step)

AWS console → region **Asia Pacific (Thailand) ap-southeast-7** → **IAM Identity Center → Users → Add user**:

| Field | Value |
|---|---|
| Username | `fahim.ahmed` |
| Password | **Send an email to this user with password setup instructions** |
| Email address (and confirm) | `fahim.ahmed@connect.hku.hk` |
| First name / Last name | `Fahim` / `Ahmed` |
| Display name | `Fahim Ahmed` |

→ **Next** → no groups → **Next** → **Add user**. AWS emails him an invitation. It's valid for **7 days**; tell him to check junk mail. If it expires: Users → fahim.ahmed → **Send email verification link** / **Reset password**.

### 6.2 Give him the access (laptop)

```powershell
$FAHIM = aws identitystore list-users --identity-store-id $IDS --query "Users[?UserName=='fahim.ahmed'].UserId" --output text @RI
"FAHIM=$FAHIM"
$REQ = aws sso-admin create-account-assignment --instance-arn $INST --target-id $ACCT --target-type AWS_ACCOUNT --permission-set-arn $PS --principal-type USER --principal-id $FAHIM --query AccountAssignmentCreationStatus.RequestId --output text @RI
aws sso-admin describe-account-assignment-creation-status --instance-arn $INST --account-assignment-creation-request-id $REQ --query "AccountAssignmentCreationStatus.[Status,FailureReason]" @RI
```

**Expect:** `FAHIM=` followed by an ID (not empty), then `SUCCEEDED`.

### 6.3 Message to Fahim (group chat or DM)

Fill in your AWS access portal link first:

> Hi Fahim, your AWS access for Week 4 is ready (until about 30 Nov).
> 1. Accept the invitation email from AWS ("Invitation to join AWS IAM Identity Center", check junk). Set your password and MFA.
> 2. Sign in at `<AWS access portal URL>` → our account → **ELEC3544-DSFM-Fahim** → Management console. It opens SageMaker in Singapore.
> 3. **Applications and IDEs → Studio →** user **fahim** → **Open Studio → JupyterLab → Create JupyterLab space** (Private, `ml.t3.medium`; you can go up to `ml.t3.xlarge`) → Run → Open JupyterLab.
> 4. Your files are already in `~/dsfm`. Read `~/dsfm/docs/WEEK4_FAHIM.md` first: it has the steps, the rules and where to save outputs. Target: models done by **Wed 11 Nov**.
> 5. **Please stop the space every time you finish** (Studio → Running instances → Stop). It auto-stops after 2 h idle, but it runs on my account.
> Each sign-in lasts 12 h, then just sign in again. Nayeon's Week 3 features will appear in `~/dsfm/data/features_week3/` the next time you start the space after they're uploaded to S3.

---

## Step 7 - Final check (laptop, read-only)

```powershell
aws sso-admin list-account-assignments --instance-arn $INST --account-id $ACCT --permission-set-arn $PS --query "AccountAssignments[].[PrincipalType,PrincipalId]" --output table @RI
"FAHIM=$FAHIM  ME=$ME"
aws sagemaker describe-user-profile --domain-id $DOM --user-profile-name fahim --query Status @R
aws sagemaker list-apps --domain-id-equals $DOM --query "Apps[?Status!='Deleted'].[SpaceName,AppType,Status,ResourceSpec.InstanceType]" --output table @R
```

**Expect:**

- the assignment table has **one** row: `USER` with Fahim's ID (not yours);
- `InService`;
- the app list is empty, or shows only apps you know about.

Done. From now on you can check at any time with the last command (or the "anything running" check in the Weeks 3-4 runbook).

---

## Reload (if you closed the terminal)

Run the Step 0.2 block, then:

```powershell
$PS = @(aws sso-admin list-permission-sets --instance-arn $INST --query "PermissionSets" --output text @RI) -split "\s+" | Where-Object { $_ -and (aws sso-admin describe-permission-set --instance-arn $INST --permission-set-arn $_ --query PermissionSet.Name --output text @RI) -eq $PSN }
$LCC_ARN = aws sagemaker describe-studio-lifecycle-config --studio-lifecycle-config-name $LCCN --query StudioLifecycleConfigArn --output text @R
$FAHIM = aws identitystore list-users --identity-store-id $IDS --query "Users[?UserName=='fahim.ahmed'].UserId" --output text @RI
"PS=$PS"; "LCC_ARN=$LCC_ARN"; "FAHIM=$FAHIM"
```

(`$ME`: repeat the two `$MYUSER` / `$ME` lines from 0.3.) Empty values just mean that step hasn't been done yet.

---

## Later - removing his access (around Mon 30 Nov)

**First** ask Fahim to copy anything worth keeping to S3. Deleting his space deletes its disk.

Run the Step 0.2 block and the Reload block, then:

```powershell
# 1. Take away his access, then delete the permission set and his user
$DREQ = aws sso-admin delete-account-assignment --instance-arn $INST --target-id $ACCT --target-type AWS_ACCOUNT --permission-set-arn $PS --principal-type USER --principal-id $FAHIM --query AccountAssignmentDeletionStatus.RequestId --output text @RI
aws sso-admin describe-account-assignment-deletion-status --instance-arn $INST --account-assignment-deletion-request-id $DREQ --query "AccountAssignmentDeletionStatus.Status" @RI
aws sso-admin delete-permission-set --instance-arn $INST --permission-set-arn $PS @RI
aws identitystore delete-user --identity-store-id $IDS --user-id $FAHIM @RI

# 2. Stop his apps and delete his spaces (names from the first command)
aws sagemaker list-spaces --domain-id-equals $DOM --query "Spaces[?OwnershipSettingsSummary.OwnerUserProfileName=='fahim'].SpaceName" --output text @R
aws sagemaker list-apps --domain-id-equals $DOM --query "Apps[?Status=='InService'].[SpaceName,AppType,AppName]" --output table @R
# for each of HIS running apps:  aws sagemaker delete-app --domain-id $DOM --space-name <space> --app-type JupyterLab --app-name default @R
# for each of HIS spaces:        aws sagemaker delete-space --domain-id $DOM --space-name <space> @R

# 3. Delete his user profile, the copy script and the role
aws sagemaker delete-user-profile --domain-id $DOM --user-profile-name fahim @R
aws sagemaker delete-studio-lifecycle-config --studio-lifecycle-config-name $LCCN @R
aws iam delete-role-policy --role-name $ROLE --policy-name fahim-studio-and-project-bucket @P
aws iam delete-role --role-name $ROLE @P
```

Wait about a minute between deleting the spaces and deleting the user profile (spaces take a moment to go). Keep the budget until the end of the course; it also watches your own space.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `$ACCT`, `$INST` or `$IDS` empty | `aws sso login --profile cloudpc`, then the 0.2 block again |
| `$ME` or `$FAHIM` empty | the username doesn't match exactly (it's case-sensitive): run the `list-users ... --output table` line from 0.3 and copy the name |
| `MalformedPolicyDocument` | a placeholder wasn't filled because a variable was empty: fix the variable, re-run the `Get-Content ... Set-Content` line, then the command |
| `EntityAlreadyExists` (role) / `ResourceInUse` (profile, copy script) | already created on an earlier try - skip that line and continue (for the copy script, get its ARN with the Reload block) |
| `ConflictException` (permission set) | already created - get `$PS` with the Reload block and continue with `put-inline-policy-to-permission-set` |
| `DuplicateRecordException` (budget) | already created - fine |
| `create-user-profile`: role can't be assumed | wait 20 s, retry |
| Assignment status `FAILED` | send me the `FailureReason` |
| Test (a): "not authorized to perform sagemaker:CreatePresignedDomainUrl" | the permission set policy has a wrong account ID: re-run the Step 4 `Get-Content` line and `put-inline-policy-to-permission-set`, then `aws sso-admin provision-permission-set --instance-arn $INST --permission-set-arn $PS --target-type ALL_PROVISIONED_ACCOUNTS @RI`, and sign in again |
| Test (c): space won't create/start, "not authorized ... CreateSpace / CreateApp" | send me the full message; I'll adjust the role policy (one `put-role-policy` re-run) |
| Test (d): `~/dsfm` empty or incomplete | `cat ~/.dsfm_sync.log` and send it to me. Fahim can always copy by hand (see `WEEK4_FAHIM.md`) |
| After changing `permission-set-policy.json` later | re-run `put-inline-policy-to-permission-set`, then `provision-permission-set` (line above) - otherwise the change doesn't reach the account |
| After changing `exec-role-policy.json` later | re-run `put-role-policy` (2.1); it applies within a minute, no re-provisioning |
| After changing `lcc-fahim-sync.sh` later | ask me - a lifecycle config can't be edited; it's replaced with a new one and the profile is updated |
