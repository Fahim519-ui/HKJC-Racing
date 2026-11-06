#!/bin/bash
# Group 17 (ELEC3544 DSFM) - runs each time Fahim's JupyterLab starts.
# Copies the project files from S3 into ~/dsfm.
#   every start : data/ and docs/ refreshed (inputs - don't edit these)
#   first start : starter script, report, figures, EDA notebook (never overwritten later)
# Worst case about 4.5 min of timeouts; normally a few seconds (about 2 MB).
# Never stops JupyterLab from starting: errors go to ~/.dsfm_sync.log, exit is always 0.
set -u
B=inns-elec3544-dsfm-sagemaker
H=${DSFM_HOME:-/home/sagemaker-user}   # override only for testing
D=$H/dsfm
LOG=$H/.dsfm_sync.log
mkdir -p "$D"
{
  echo "=== $(date -u '+%F %T') sync start"
  mkdir -p "$D/data/processed" "$D/data/features_week3" "$D/docs" "$D/reports/figures" "$D/notebooks"
  timeout 90 aws s3 cp "s3://$B/processed/runners_merged.parquet" "$D/data/processed/" --only-show-errors
  timeout 30 aws s3 sync "s3://$B/features/" "$D/data/features_week3/" --only-show-errors
  timeout 30 aws s3 sync "s3://$B/code/docs/" "$D/docs/" --only-show-errors
  if [ ! -f "$D/.seeded" ]; then
    timeout 30 aws s3 cp "s3://$B/code/week34_starter.py" "$D/" --only-show-errors \
      && timeout 30 aws s3 cp "s3://$B/reports/cleaning_report.md" "$D/reports/" --only-show-errors \
      && timeout 30 aws s3 sync "s3://$B/reports/figures/" "$D/reports/figures/" --only-show-errors \
      && timeout 30 aws s3 cp "s3://$B/code/notebooks/02_eda.ipynb" "$D/notebooks/" --only-show-errors \
      && date -u '+%F %T' > "$D/.seeded"
  fi
  if [ "$(id -u)" = "0" ]; then chown -R 1000:100 "$D"; fi
  ls -l "$D" "$D/data/processed" "$D/docs"
  echo "=== $(date -u '+%F %T') sync done"
} >> "$LOG" 2>&1
if [ "$(id -u)" = "0" ]; then chown 1000:100 "$LOG"; fi
exit 0
