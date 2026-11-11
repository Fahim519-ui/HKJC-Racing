"""One status sheet for registry v3.1: every model input, every building block, and Nayeon's
history-depth and missingness tables side by side (status sheet of 8 Nov, Tier 2, Inns).

Reads only files that already exist; it builds no features and fits nothing:
  analysis/feature_stability_results/stability_summary.csv            (default audit: model inputs)
  analysis/feature_stability_results/all_registry/stability_summary.csv (--all-registry audit)
  analysis/history_depth_results/{history_depth_by_window,missingness_by_window}.csv (Nayeon)
and writes analysis/feature_stability_results/STABILITY_V3_1.md.

h_test is never shown: Nayeon's h_test rows are information only and are filtered out here, and
the audit's h_test columns are not read. Run after both audits:
    python analysis/feature_stability_audit.py
    python analysis/feature_stability_audit.py --all-registry --output-dir analysis/feature_stability_results/all_registry
    python analysis/stability_v31_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model_logic.feature_registry import REGISTRY_VERSION, get_features, get_spec, registry_hash  # noqa: E402

RES = REPO / "analysis/feature_stability_results"
HIST = REPO / "analysis/history_depth_results"
OUT = RES / "STABILITY_V3_1.md"
WINDOWS = ["k_train", "k_early", "k_tune", "h_s2"]          # never h_test
RAW_VERSIONS = ["jt_jockey_win365", "jt_trainer_win365", "hf_last_fin_pct", "hf_avg3_fin_pct"]

# Proposed meaning of each status. NOT agreed yet: Fahim decides with Inns before the re-run.
PROPOSED_RULE = [
    ("keep", "Used as normal.", "-"),
    ("watch", "Used. Named in the limitations; its fold-level effect is reported, not tuned on.",
     "Shift explained by shorter HKJC history (compare h_s2 with k_early) or a moderate drift."),
    ("review", "Kept in the primary model (features are chosen on k_tune only, not by this audit). "
               "Its contribution is tested by the existing `no_trainer_rate` ablation and reported.",
     "Effect differs between eras even against the history-matched window."),
]


def md_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return rows


def main() -> int:
    main_audit = pd.read_csv(RES / "stability_summary.csv")
    full_audit = pd.read_csv(RES / "all_registry/stability_summary.csv")
    main_meta = json.loads((RES / "audit_meta.json").read_text(encoding="utf-8"))
    full_meta = json.loads((RES / "all_registry/audit_meta.json").read_text(encoding="utf-8"))
    hist_meta = json.loads((HIST / "meta.json").read_text(encoding="utf-8"))
    depth = pd.read_csv(HIST / "history_depth_by_window.csv")
    miss = pd.read_csv(HIST / "missingness_by_window.csv")

    cl, lgbm = set(get_features("conditional_logit")), set(get_features("lgbm"))
    inputs = [f for f in get_features("lgbm")] + [f for f in get_features("conditional_logit") if f not in lgbm]
    audited = main_audit.set_index("feature")
    missing_inputs = [f for f in inputs if f not in audited.index]
    if missing_inputs:
        raise SystemExit(f"model inputs missing from the audit: {missing_inputs}")

    # consistency: both audits and Nayeon's tables describe the same data and registry
    hashes = {"audit data": main_meta.get("data_sha256"), "all-registry data": full_meta.get("data_sha256"),
              "history-depth data": hist_meta.get("data_sha256")}
    regs = {"live": registry_hash(), "audit": main_meta["coverage"]["registry_hash"],
            "all-registry": full_meta["coverage"]["registry_hash"], "history-depth": hist_meta["registry_hash"]}
    same_data, same_reg = len(set(hashes.values())) == 1, len(set(regs.values())) == 1
    full_status = full_audit.set_index("feature")["status"]
    agree_runs = all(full_status.get(f) == audited.loc[f, "status"] for f in audited.index)
    reg_disagree = [f for f in inputs if get_spec(f).status != audited.loc[f, "status"]]

    # 1. model inputs
    rows = []
    for f in inputs:
        r = audited.loc[f]
        rows.append({"feature": f"`{f}`", "used by": "CL + LGBM" if f in cl else "LGBM",
                     "audit": f"**{r['status']}**", "registry": get_spec(f).status,
                     "why": r["reasons"]})
    inputs_df = pd.DataFrame(rows)
    order = {"**review**": 0, "**watch**": 1, "**keep**": 2}
    inputs_df = inputs_df.sort_values(["audit", "feature"], key=lambda s: s.map(order) if s.name == "audit" else s)
    counts = main_audit.set_index("feature").loc[inputs, "status"].value_counts().to_dict()

    # 2. building blocks (not model inputs)
    helpers = [f for f in full_status.index if f not in inputs and f not in RAW_VERSIONS]
    fa = full_audit.set_index("feature")
    blocks = pd.DataFrame([{"feature": f"`{f}`", "audit": f"**{fa.loc[f, 'status']}**",
                            "registry label": get_spec(f).status, "why": fa.loc[f, "reasons"]}
                           for f in sorted(helpers)])
    text_cols = full_meta["coverage"].get("not_numeric_skipped", [])

    # 3. history depth (Nayeon), no h_test
    d = depth[depth["window"].isin(WINDOWS) & ~depth["info_only"].astype(bool)]
    piv = (d.assign(cell=lambda t: t["p50"].astype(int).astype(str) + " / "
                    + (100 * t["share_zero"]).round(1).astype(str) + "%")
             .pivot(index="level", columns="window", values="cell")[WINDOWS]
             .reindex(["career", "track", "distance_band", "track_distance"]).reset_index())
    m = miss[miss["window"].isin(WINDOWS) & ~miss["info_only"].astype(bool) & miss["feature"].isin(inputs)]
    mp = m.pivot(index="feature", columns="window", values="missing_pct")[WINDOWS]
    mp = mp[(mp > 0).any(axis=1)].sort_values("h_s2", ascending=False)
    mp_df = mp.round(2).reset_index().assign(feature=lambda t: "`" + t["feature"] + "`",
                                             audit=lambda t: [audited.loc[f.strip("`"), "status"] for f in t["feature"]])
    never = sorted(set(inputs) - set(mp.index))

    lines = [
        "# Stability status sheet, registry v3.1 (generated)", "",
        f"Generated by `analysis/stability_v31_summary.py` from the stability audit (default and "
        f"`--all-registry`) and Nayeon's `analysis/history_depth_results/`. Registry `{REGISTRY_VERSION}`, "
        f"hash `{registry_hash()[:12]}...`; data SHA-256 `{str(hashes['audit data'])[:12]}...`.", "",
        f"Consistency: all three sources use the same data file: **{same_data}**; the same registry "
        f"(and it is the live one): **{same_reg}**; the model inputs get the same status in the "
        f"default and the all-registry audit: **{agree_runs}**.", "",
        "**No h_test value appears here.** Nayeon's h_test rows and the audit's h_test columns are "
        "information only and are left out. This sheet is a warning list, not a feature-selection "
        "rule: features are chosen on `k_tune` only (`docs/DATA_RULES.md`).", "",
        f"## 1. Every model input has an explicit status ({len(inputs)} inputs)", "",
        f"All {len(inputs)} inputs (LightGBM uses all of them; the conditional logit {len(cl)}) are "
        f"audited. Audit: {counts.get('keep', 0)} keep, {counts.get('watch', 0)} watch, "
        f"{counts.get('review', 0)} review. "
        + ("Registry and audit agree on every input." if not reg_disagree else
           f"Registry and audit disagree on: {', '.join('`' + f + '`' for f in reg_disagree)}."), "",
        *md_table(inputs_df), "",
        "## 2. Building blocks (not model inputs)", "",
        f"The {len(helpers)} intermediate columns the builder makes but no model reads (pair ride "
        "counts and baselines, history-depth counts). Audited with `--all-registry` "
        "(`all_registry/stability_report.md`). None is `review`; most `watch` flags come from "
        "shorter HKJC histories, which section 3 confirms.", "",
        *md_table(blocks), "",
    ]
    if text_cols:
        lines += [f"Not audited: {', '.join('`' + c + '`' for c in text_cols)} is a text label "
                  "(sprint / mile / middle / staying); SMD and PSI need numbers. It only selects "
                  "which history the `td_distance_band_*` features use, and those are audited.", ""]
    lines += [
        "The registry labels of these building blocks are **not changed** in this branch: they are "
        "not model inputs, and a label change alters the registry hash that Fahim's re-run "
        "records. Proposal: update them in the same commit as any other registry change, before "
        "the re-run, if Fahim agrees.", "",
        "## 3. History depth and missingness (Nayeon), no h_test", "",
        "Earlier-date starts at race time, median / share of runners with zero starts:", "",
        *md_table(piv), "",
        "h_s2 (HKJC Stage 2) has about as much history as k_early, the history-matched Kaggle "
        "window, and much less than k_train. So a Kaggle-to-HKJC shift in a history feature is "
        "first a history-length effect; the audit already tests every flag against k_early.", "",
        "Missing values of model inputs (% of rows). Per Nayeon's report, every blank comes from a "
        "short history (a first start has no previous finish; the slope needs two earlier "
        "finishes), which the equal rates across these features fit:", "",
        *md_table(mp_df), "",
        f"Never missing ({len(never)}): {', '.join('`' + f + '`' for f in never)}.", "",
        "## 4. Treatment rule per status: PROPOSED, not agreed", "",
        "**To decide with Fahim before the v3.1 re-run.** Until then nothing is enforced; the "
        "registry status is a label only and no feature list depends on it.", "",
        *md_table(pd.DataFrame(PROPOSED_RULE, columns=["status", "proposed treatment", "typical reason"])), "",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}: {len(inputs)} inputs {counts}, {len(helpers)} building blocks, "
          f"same data {same_data}, same registry {same_reg}, runs agree {agree_runs}, "
          f"registry disagreements {len(reg_disagree)}")
    return 0 if same_data and same_reg and agree_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
