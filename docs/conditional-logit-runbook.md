# Conditional-logit tuning runbook

Run this in SageMaker from the repository root:

```bash
python legacy_code/modeling/conditional_logit_tuning.py
```

The script automatically prefers the latest pushed Week-3 feature table,
`data/clean data/runners_merged_with_features.parquet`. Otherwise it reads the
tracked repository's `data/clean data/runners_merged.parquet`, or SageMaker's
`data/processed/runners_merged.parquet`. It uses `k_train` for fitting and the
fixed 2019/20 Kaggle `k_tune` window for feature selection.

It recomputes the pushed horse past-three-race rates with source-separated,
earlier-race history before it considers them. It deliberately excludes
`early_position` and `position_change`, which describe the current race after
it happens, and the pushed jockey/trainer expanding rates, which change within
the same day. The existing smoothed, earlier-day `jt_*` rates remain eligible.

To run the new tracked script in your existing SageMaker space, copy it there
first:

```bash
mkdir -p ~/dsfm/modeling
aws s3 cp s3://inns-elec3544-dsfm-sagemaker/code/fahim/conditional_logit_tuning.py \
  ~/dsfm/modeling/
cd ~/dsfm
python legacy_code/modeling/conditional_logit_tuning.py
```

Upload `legacy_code/modeling/conditional_logit_tuning.py` to `code/fahim/` before this
step; the existing lifecycle sync does not automatically copy tracked
repository files into `~/dsfm`.

It begins with six planned strong features and tries the other starter features
one at a time. A feature is retained only when it reduces `k_tune` per-race log
loss by at least 0.001. It writes the selection trace, selected coefficients,
and Spearman correlations to `legacy_code/artifacts/conditional_logit_tuning/`.

The market probability is a benchmark only, never a Stage-1 input. The job
always saves the best selected conditional logit and its Stage-2/test
predictions. `tuning_report.md` states whether it beat market log loss on
`k_tune`; that comparison is not evidence of final performance, which is
measured once on the untouched Apr-Oct 2026 HKJC test window in Week 5.

If accepted, upload the generated artefacts:

```bash
B=inns-elec3544-dsfm-sagemaker
aws s3 cp legacy_code/artifacts/conditional_logit_tuning \
  s3://$B/models/conditional_logit_tuning/ --recursive
```
