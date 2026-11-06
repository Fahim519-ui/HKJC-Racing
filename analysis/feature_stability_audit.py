"""Feature stability audit (Week 4 roadmap of 4 Nov, priority 5).

Question: does each model feature look and behave the same in the Kaggle data the models
learn from (2017-20) and in the HKJC data they are used on (2025-26)?

For every feature the three model scripts can use, it reports:
  1. Distribution shift against Kaggle train: standardised mean difference (SMD),
     population stability index (PSI) and the Kolmogorov-Smirnov statistic (KS).
  2. Missing (blank) rate in every window.
  3. Effect stability: the coefficient of a one-feature conditional logit in k_train,
     k_tune and h_s2. Results from h_test are never used.
  4. A history-matched comparison: "k_early" (Kaggle Nov 2017 - Mar 2018) sits at the same
     point after the data starts as h_s2 (HKJC Nov 2025 - Mar 2026). If a shift shrinks
     against k_early, it comes from shorter horse histories, not from a different era.

h_test is included for distributions and blank rates only ("info only"). Feature choices
must be made on k_train / k_tune / h_s2, never on h_test.

Run from the repo root, venv active:
    python analysis/feature_stability_audit.py
Writes (default): analysis/feature_stability_results/ inside the repo
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from statsmodels.discrete.conditional_models import ConditionalLogit

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parents[1]

# Thresholds (see README_FEATURE_STABILITY.md for sources)
SMD_SMALL, SMD_LARGE = 0.10, 0.25
PSI_SMALL, PSI_LARGE = 0.10, 0.25
MISSING_PP = 5.0           # blank-rate difference, percentage points
EFFECT_Z = 2.0             # coefficient difference, in combined standard errors
REFERENCE = "k_train"
EFFECT_WINDOWS = ["k_train", "k_early", "k_tune", "h_s2"]   # never h_test
# features built from a horse's, person's or pair's past starts (shorter HKJC history shifts them)
HISTORY_GROUPS = ("hf", "jt", "jtp", "td", "w3")
RAW_VERSIONS = ["jt_jockey_win365", "jt_trainer_win365", "hf_last_fin_pct", "hf_avg3_fin_pct"]


# --------------------------------------------------------------------------- loading
def load(name: str):
    # Fahim's modules import each other (e.g. jt_partnership_features), so put both the
    # repo root and model_logic/ on the import path, whatever folder we are run from.
    for path in (str(REPO), str(REPO / "model_logic")):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(name, REPO / "model_logic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(data_path: Path, all_registry: bool = False) -> tuple[pd.DataFrame, list[str], dict]:
    """Build the features the models use and decide which ones to audit.

    The feature list comes from Fahim's registry (model_logic/feature_registry.py), joined with
    the model scripts' own lists, so a feature added to the registry is audited automatically.
    Default = every feature a model can use (registry "lgbm" set, which includes the
    conditional-logit set, plus the scripts' lists and the raw versions). With all_registry=True
    every IMPLEMENTED registry feature is audited, including intermediate helper columns
    (ride counts, baselines) that no model reads directly.

    Returns the built data, the audited feature names, and a coverage dict.
    """
    cl = load("conditional_logit_tuning_v2")
    lgb2 = load("lightgbm_independent_probability_v2")
    v1 = load("lightgbm_independent_probability")
    fr = load("feature_registry")
    data = cl.build_features(pd.read_parquet(data_path))
    v2 = lgb2.add_context_features(data, list(cl.ALL_FEATURES))
    scripts = set(cl.ALL_FEATURES) | set(v2) | set(v1.FEATURES) | set(RAW_VERSIONS)
    registry_models = set(fr.get_features("lgbm")) | set(fr.get_features("conditional_logit"))
    registry_all = set(fr.get_features("all"))
    wanted = scripts | registry_models | (registry_all if all_registry else set())
    # a feature the builder does not produce cannot be audited; report it instead of crashing
    feats = {f for f in wanted if f in data.columns}
    order = {"rc": 0, "hf": 1, "jt": 2, "jtp": 3, "td": 4, "w3": 5}
    feats = sorted(feats, key=lambda f: (order.get(f.split("_")[0], 9), f))
    coverage = {
        "registry_version": fr.REGISTRY_VERSION,
        "registry_hash": fr.registry_hash(),
        "registry_implemented": len(registry_all),
        "registry_implemented_not_audited": sorted(registry_all - set(feats)),
        "wanted_but_not_built": sorted(wanted - set(feats)),
        "all_registry": all_registry,
    }
    return data, feats, coverage


def add_registry_status(summary: pd.DataFrame) -> pd.DataFrame:
    """Attach the status Fahim's registry gives each feature, and flag disagreements."""
    fr = load("feature_registry")
    registry = {spec.name: spec.status for spec in fr.REGISTRY}
    out = summary.copy()
    out["registry_status"] = out["feature"].map(registry)
    out["status_agrees"] = out["status"].eq(out["registry_status"])
    return out


def windows(d: pd.DataFrame) -> dict[str, pd.Series]:
    t = d["race_date"]
    return {
        "k_train": d["split"].eq("k_train"),
        "k_tune": d["split"].eq("k_tune"),
        "k_early": d["source"].eq("kaggle") & (t >= "2017-11-01") & (t < "2018-04-01"),
        "h_s2": d["split"].eq("h_s2"),
        "h_test": d["split"].eq("h_test"),
    }


# ----------------------------------------------------------------------- statistics
def describe(x: pd.Series) -> dict:
    v = x.astype("float64")
    return {
        "n": int(len(v)), "missing_pct": float(100 * v.isna().mean()),
        "mean": float(v.mean()), "sd": float(v.std()),
        "p10": float(v.quantile(0.10)), "median": float(v.median()), "p90": float(v.quantile(0.90)),
    }


def smd(ref: pd.Series, cur: pd.Series) -> float:
    a, b = ref.astype("float64").dropna(), cur.astype("float64").dropna()
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else float("nan")


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI with decile bins taken from the reference, plus one bin for blanks."""
    r, c = ref.to_numpy("float64"), cur.to_numpy("float64")
    rv = r[~np.isnan(r)]
    edges = np.unique(np.quantile(rv, np.linspace(0, 1, bins + 1)[1:-1]))

    def shares(x):
        nan = np.isnan(x)
        idx = np.searchsorted(edges, x[~nan], side="left")   # bins (-inf,e1], (e1,e2], ...
        counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
        counts = np.append(counts, nan.sum())
        return np.clip(counts / counts.sum(), eps, None)

    p, q = shares(r), shares(c)
    return float(np.sum((q - p) * np.log(q / p)))


def ks(ref: pd.Series, cur: pd.Series) -> float:
    a, b = ref.astype("float64").dropna(), cur.astype("float64").dropna()
    return float(ks_2samp(a, b).statistic) if len(a) and len(b) else float("nan")


def is_race_level(d: pd.DataFrame, feat: str) -> bool:
    """True when the feature is the same for every runner in a race (it cancels in a logit)."""
    spread = d.groupby("race_id")[feat].std().fillna(0.0)
    return bool((spread < 1e-12).all())


def effect(d: pd.DataFrame, feat: str, mu: float, sd: float) -> tuple[float, float]:
    """One-feature conditional logit. Feature standardised with k_train mean/sd, blanks = 0
    (the training mean), as in the model scripts. Returns (coefficient, standard error)."""
    x = ((d[feat].astype("float64") - mu) / sd).fillna(0.0).to_numpy()
    res = ConditionalLogit(d["is_winner"].astype("float64").to_numpy(), x[:, None],
                           groups=d["race_id"].to_numpy()).fit(disp=0)
    return float(res.params[0]), float(res.bse[0])


def level(smd_value: float, psi_value: float) -> str:
    s = abs(smd_value) if np.isfinite(smd_value) else 0.0
    if s >= SMD_LARGE or psi_value >= PSI_LARGE:
        return "large"
    if s >= SMD_SMALL or psi_value >= PSI_SMALL:
        return "moderate"
    return "stable"


RANK = {"stable": 0, "moderate": 1, "large": 2}


def verdict(row: pd.Series) -> tuple[str, str]:
    """Status rules (also in README_FEATURE_STABILITY.md):
    review = large Kaggle->HKJC shift that is still there in the history-matched window,
             or an effect that changes sign;
    watch  = any smaller warning; keep = no warning."""
    reasons = []
    main, matched = row["shift_h_s2"], row["shift_h_s2_vs_k_early"]
    history = row["group"] in HISTORY_GROUPS
    depth = history and RANK[main] > RANK[matched]          # shift shrinks when histories match
    real_large = main == "large" and RANK[matched] == 2
    if main != "stable":
        if depth:
            reasons.append(f"{main} shift, mostly from shorter HKJC histories")
        elif real_large:
            reasons.append("large Kaggle->HKJC shift, also against the history-matched window")
        else:
            reasons.append(f"{main} Kaggle->HKJC shift")
    if row["shift_k_tune"] != "stable":
        reasons.append(f"{row['shift_k_tune']} drift inside Kaggle (train->tune)")
    if abs(row["missing_pp_h_s2"]) >= MISSING_PP:
        reasons.append(f"blank rate differs by {row['missing_pp_h_s2']:+.1f} pp")

    flip = bool(row.get("effect_sign_flip", False))
    z_train = row.get("effect_z_h_s2", np.nan)
    z_early = row.get("effect_z_h_s2_vs_k_early", np.nan)
    if flip:
        reasons.append("effect changes sign in HKJC")
    elif np.isfinite(z_train) and abs(z_train) >= EFFECT_Z:
        direction = "weaker" if row["coef_h_s2"] ** 2 < row["coef_k_train"] ** 2 else "stronger"
        if np.isfinite(z_early) and abs(z_early) < EFFECT_Z:
            reasons.append(f"effect {direction} in HKJC, but not against the history-matched window")
        else:
            reasons.append(f"effect {direction} in HKJC, also against the history-matched window")

    status = "review" if (real_large or flip) else ("watch" if reasons else "keep")
    return status, "; ".join(reasons) if reasons else "-"


# ----------------------------------------------------------------------------- main
def audit(data: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    win = windows(data)
    ref = data.loc[win[REFERENCE]]
    stats_rows, summary_rows = [], []
    for feat in feats:
        for name, mask in win.items():
            stats_rows.append({"feature": feat, "window": name, **describe(data.loc[mask, feat])})
        row = {"feature": feat, "group": feat.split("_")[0]}
        for name, base, label in [("k_tune", REFERENCE, "k_tune"), ("h_s2", REFERENCE, "h_s2"),
                                  ("h_s2", "k_early", "h_s2_vs_k_early"), ("h_test", "h_s2", "h_test_info")]:
            a, b = data.loc[win[base], feat], data.loc[win[name], feat]
            row[f"smd_{label}"] = smd(a, b)
            row[f"psi_{label}"] = psi(a, b)
            row[f"ks_{label}"] = ks(a, b)
            row[f"shift_{label}"] = level(row[f"smd_{label}"], row[f"psi_{label}"])
        miss = {n: 100 * data.loc[m, feat].isna().mean() for n, m in win.items()}
        row["missing_pp_h_s2"] = float(miss["h_s2"] - miss[REFERENCE])
        row["missing_pp_h_test_info"] = float(miss["h_test"] - miss["h_s2"])

        row["race_level"] = is_race_level(data, feat)
        if not row["race_level"]:
            mu, sd = ref[feat].astype("float64").mean(), ref[feat].astype("float64").std()
            for name in EFFECT_WINDOWS:
                assert name != "h_test"
                b, se = effect(data.loc[win[name]], feat, mu, sd if sd > 0 else 1.0)
                row[f"coef_{name}"], row[f"se_{name}"] = b, se
            bk, sk, bh, sh = row["coef_k_train"], row["se_k_train"], row["coef_h_s2"], row["se_h_s2"]
            be, se_e = row["coef_k_early"], row["se_k_early"]
            row["effect_z_h_s2"] = (bh - bk) / np.sqrt(sk ** 2 + sh ** 2)
            row["effect_z_h_s2_vs_k_early"] = (bh - be) / np.sqrt(se_e ** 2 + sh ** 2)
            row["effect_sign_flip"] = bool(np.sign(bk) != np.sign(bh)
                                           and abs(bk / sk) >= EFFECT_Z and abs(bh / sh) >= EFFECT_Z)
        row["status"], row["reasons"] = verdict(pd.Series(row))
        summary_rows.append(row)
    return pd.DataFrame(stats_rows), pd.DataFrame(summary_rows)


def figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary.iloc[::-1]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(s) + 1.6))
    for col, mark, label in [("smd_k_tune", "o", "Kaggle tune vs Kaggle train"),
                             ("smd_h_s2", "s", "HKJC Stage 2 vs Kaggle train"),
                             ("smd_h_s2_vs_k_early", "^", "HKJC Stage 2 vs Kaggle early (history-matched)")]:
        ax.scatter(s[col], y, marker=mark, label=label, s=28)
    for x in (-SMD_LARGE, -SMD_SMALL, SMD_SMALL, SMD_LARGE):
        ax.axvline(x, color="grey", lw=0.8, ls="--" if abs(x) == SMD_LARGE else ":")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y, s["feature"])
    ax.set_xlabel("Standardised mean difference (dotted 0.10, dashed 0.25)")
    ax.set_title("Feature stability: shift against the Kaggle training data")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), fontsize=8, ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(summary: pd.DataFrame, meta: dict, path: Path) -> None:
    def fmt(x, nd=2):
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"

    order = {"review": 0, "watch": 1, "keep": 2}
    s = summary.assign(o=summary["status"].map(order)).sort_values(["o", "feature"])
    counts = summary["status"].value_counts().reindex(["review", "watch", "keep"], fill_value=0)
    lines = [
        "# Feature stability audit (generated)", "",
        f"Generated {meta['generated']} from `{meta['data']}` (SHA-256 `{meta['data_sha256'][:12]}...`), "
        f"repo commit `{meta['repo_commit']}`. {meta['n_features']} features.", "",
        f"**Result:** {counts['review']} review, {counts['watch']} watch, {counts['keep']} keep.", "",
        "Thresholds: SMD 0.10 / 0.25, PSI 0.10 / 0.25, blank rate 5 pp, effect difference 2 SE. "
        "Effects use k_train, k_early, k_tune and h_s2 only, never h_test. The h_test columns are "
        "information only.", "",
        "**This is a warning list, not a feature-selection rule.** Features are chosen on k_tune only "
        "(DATA_RULES.md). Confirm any change on k_tune before acting on a flag here.", "",
        "## Summary", "",
        "Columns: shift of HKJC Stage 2 (h_s2) against Kaggle train, and against Kaggle early "
        "(history-matched). Coefficients are per 1 SD of the feature in Kaggle train.", "",
        "| Feature | Status | SMD h_s2 | SMD vs early | PSI h_s2 | PSI vs early | PSI tune | Blank pp | "
        "Coef k_train | Coef k_early | Coef h_s2 | Reasons |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def coef(r, w):
        return "race-level" if r["race_level"] else f"{r[f'coef_{w}']:+.3f} ± {r[f'se_{w}']:.3f}"

    for _, r in s.iterrows():
        lines.append(
            f"| `{r['feature']}` | **{r['status']}** | {fmt(r['smd_h_s2'])} | "
            f"{fmt(r['smd_h_s2_vs_k_early'])} | {r['psi_h_s2']:.3f} | {r['psi_h_s2_vs_k_early']:.3f} | "
            f"{r['psi_k_tune']:.3f} | {fmt(r['missing_pp_h_s2'], 1)} | {coef(r, 'k_train')} | "
            f"{coef(r, 'k_early')} | {coef(r, 'h_s2')} | {r['reasons']} |")
    # where this audit and Fahim's feature registry give a feature different statuses
    cov = meta.get("coverage", {})
    lines += ["", "## Agreement with the feature registry", "",
              f"Registry `{cov.get('registry_version', '?')}` (hash `{str(cov.get('registry_hash', ''))[:12]}...`). "
              f"{cov.get('registry_implemented', '?')} implemented features; "
              f"{len(cov.get('registry_implemented_not_audited', []))} not audited in this run "
              f"({'all registry features' if cov.get('all_registry') else 'default: model inputs only; use --all-registry for helper columns'}).", ""]
    if "status_agrees" in summary.columns:
        bad = summary.loc[~summary["status_agrees"]]
        if bad.empty:
            lines.append("Every audited feature has the same status in the registry and in this audit.")
        else:
            lines += ["These features have a different status in the registry than in this audit. "
                      "Ask the registry owner to update the label, or explain why not.", "",
                      "| Feature | This audit | Registry | Why this audit says so |", "|---|---|---|---|"]
            for _, r in bad.iterrows():
                lines.append(f"| `{r['feature']}` | **{r['status']}** | {r['registry_status']} | {r['reasons']} |")
    not_built = cov.get("wanted_but_not_built", [])
    if not_built:
        lines += ["", f"Listed by a model or the registry but not built by the feature builder, so not audited: "
                      f"{', '.join(f'`{n}`' for n in not_built)}."]

    lines += ["", "## h_test (information only, do not use for feature choice)", "",
              "| Feature | SMD h_test vs h_s2 | PSI h_test vs h_s2 | Blank pp |", "|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| `{r['feature']}` | {fmt(r['smd_h_test_info'])} | {r['psi_h_test_info']:.3f} | "
                     f"{fmt(r['missing_pp_h_test_info'], 1)} |")
    lines += ["", "Files: `stability_summary.csv`, `window_stats.csv`, `smd_by_feature.png`, `audit_meta.json`."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    default_out = REPO / "analysis" / "feature_stability_results"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=REPO / "data/clean data/runners_merged.parquet")
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--all-registry", action="store_true",
                        help="also audit intermediate registry helper columns (ride counts, baselines)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data, feats, coverage = build(args.data, all_registry=args.all_registry)
    stats, summary = audit(data, feats)
    summary = add_registry_status(summary)
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "data": args.data.name,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "repo_commit": commit, "n_features": len(feats), "features": feats,
        "windows": {n: int(m.sum()) for n, m in windows(data).items()},
        "thresholds": {"smd": [SMD_SMALL, SMD_LARGE], "psi": [PSI_SMALL, PSI_LARGE],
                       "missing_pp": MISSING_PP, "effect_z": EFFECT_Z},
        "effect_windows": EFFECT_WINDOWS,
        "coverage": coverage,
    }
    stats.to_csv(args.output_dir / "window_stats.csv", index=False)
    summary.to_csv(args.output_dir / "stability_summary.csv", index=False)
    (args.output_dir / "audit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    figure(summary, args.output_dir / "smd_by_feature.png")
    report(summary, meta, args.output_dir / "stability_report.md")
    print(summary["status"].value_counts().to_string())
    disagree = summary.loc[~summary["status_agrees"], ["feature", "status", "registry_status"]]
    print(f"registry disagreements: {len(disagree)}")
    if len(disagree):
        print(disagree.to_string(index=False))
    print(f"{len(feats)} features in {time.time() - t0:.1f}s -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
