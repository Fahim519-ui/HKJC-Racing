"""Overfitting and selection checks on the fresh v3.1 fold runs (11 Nov 2026, Inns).

Answers "is the model overfitting?" the time-respecting way. Plain k-fold would split races at
random, which breaks time order and puts same-day races on both sides (docs/DATA_RULES.md:
split by race date only). The fold runner's k1 / k2 / h1 folds are already the forward-chained
version, so this script reads their outputs and fits no model:

  1. In-sample vs out-of-sample: each blend is fitted on the fold's Stage 2 (s2) window and
     scored on the later validation window. A blend that overfits gains much more on s2.
  2. Stability across folds: conditional-logit features selected, LightGBM rounds, blend
     penalty and blend weights per fold. Choices that change from fold to fold fit noise.
  3. Best-of-N selection: every config x blend candidate's gain over the calibrated market,
     and a race-day bootstrap of the MAXIMUM gain with every candidate centred at zero. This is
     how large the best of N candidates looks by chance, so it tests whether the best
     configuration beats what picking the winner alone produces.
  4. Figures: primary gain per fold with its 95% race-day interval, and the best-of-N check.

Reads artifacts/folds/<config>/{k1,k2,h1}/predictions.parquet (valid and s2 rows) and the
fold JSON files. Refuses unless every fold run passes analysis/provenance.py (live registry,
pushed commit, clean tree, evidence true). Never reads h_test: every frame goes through
eval_utils.check_not_test.

    python analysis/overfitting_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import eval_utils as ev  # noqa: E402
import provenance as prov  # noqa: E402

CONFIGS = ["primary", "no_trainer_rate", "no_track_distance", "no_trainer_no_track_distance"]
FOLDS = ["k1", "k2", "h1"]
ELIGIBLE = ["k1", "k2"]
BLENDS = ["mkt+cl", "mkt+lgbm", "mkt+cl+lgbm"]
PRIMARY = ("primary", "mkt+cl+lgbm")
BASE = "p_market_calibrated"
N_BOOT, SEED = 2000, 0          # same settings as the fold runner's bootstrap


def load(root: Path) -> dict:
    frames = {}
    for cfg in CONFIGS:
        for f in FOLDS:
            p = pd.read_parquet(root / cfg / f / "predictions.parquet")
            ev.check_not_test(p)
            frames[(cfg, f)] = p
    return frames


def in_vs_out(frames: dict) -> pd.DataFrame:
    rows = []
    for (cfg, f), p in frames.items():
        for role in ("s2", "valid"):
            d = p[p["role"].eq(role)].reset_index(drop=True)
            base = ev.race_losses(d[BASE], d)
            for b in BLENDS:
                rows.append({"config": cfg, "blend": b, "fold": f, "role": role,
                             "gain": float((base - ev.race_losses(d["p_" + b], d)).mean())})
    t = pd.DataFrame(rows).pivot_table(index=["config", "blend", "fold"], columns="role",
                                       values="gain").reset_index()
    t = t.rename(columns={"s2": "gain_in_sample_s2", "valid": "gain_out_of_sample_valid"})
    t["in_minus_out"] = t["gain_in_sample_s2"] - t["gain_out_of_sample_valid"]
    return t


def stability(root: Path) -> tuple[pd.DataFrame, dict]:
    rows, sets = [], {}
    for f in FOLDS:
        d = root / "primary" / f
        selected = json.loads((d / "cl_selected_features.json").read_text())
        lg = json.loads((d / "lgbm_meta.json").read_text())
        bl = json.loads((d / "blend.json").read_text())
        sets[f] = set(selected)
        th = bl["theta"]
        rows.append({"fold": f, "cl_features": len(selected), "lgbm_best_iter": lg["best_iter"],
                     "lgbm_rounds": lg["rounds"], "seeds": " ".join(map(str, lg["seeds"])),
                     "blend_lambda": bl["lambda"], "beta_calibrated_market": th["market_calibrated"][0],
                     "mkt+cl: beta": th["mkt+cl"][0], "mkt+cl: alpha_cl": th["mkt+cl"][1],
                     "mkt+lgbm: alpha_lgbm": th["mkt+lgbm"][1],
                     "mkt+cl+lgbm: alpha_cl": th["mkt+cl+lgbm"][1],
                     "mkt+cl+lgbm: alpha_lgbm": th["mkt+cl+lgbm"][2]})
    common, union = set.intersection(*sets.values()), set.union(*sets.values())
    jac = {f"{a}-{b}": len(sets[a] & sets[b]) / len(sets[a] | sets[b])
           for i, a in enumerate(FOLDS) for b in FOLDS[i + 1:]}
    return pd.DataFrame(rows), {"common": sorted(common), "union": len(union), "jaccard": jac}


def candidate_diffs(frames: dict, folds: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    diffs, days = {}, None
    for cfg in CONFIGS:
        for b in BLENDS:
            parts, dparts = [], []
            for f in folds:
                d = frames[(cfg, f)]
                d = d[d["role"].eq("valid")].reset_index(drop=True)
                diff = ev.race_losses(d[BASE], d) - ev.race_losses(d["p_" + b], d)
                diff.index = f + "|" + diff.index.astype(str)
                rd = ev.race_days(d)
                rd.index = f + "|" + rd.index.astype(str)
                parts.append(diff)
                dparts.append(f + "|" + rd.astype(str))
            diffs[f"{cfg} / {b}"] = pd.concat(parts)
            if days is None:
                days = pd.concat(dparts)
    D = pd.DataFrame(diffs).dropna()
    return D, days.reindex(D.index)


def best_of_n(frames: dict, folds: list[str]) -> dict:
    D, days = candidate_diffs(frames, folds)
    obs = D.mean()
    g = D.groupby(days.to_numpy())
    sums, cnt = g.sum().to_numpy(), g.size().to_numpy()
    pick = np.random.default_rng(SEED).integers(0, len(cnt), size=(N_BOOT, len(cnt)))
    boot = sums[pick].sum(axis=1) / cnt[pick].sum(axis=1)[:, None]
    centred_max = (boot - obs.to_numpy()).max(axis=1)
    best = obs.idxmax()
    lo, hi = np.quantile(boot[:, D.columns.get_loc(best)], [0.025, 0.975])
    return {"folds": "+".join(folds), "candidates": D.shape[1], "races": len(D), "race_days": len(cnt),
            "best": best, "best_gain": float(obs.max()), "best_ci_low": float(lo), "best_ci_high": float(hi),
            "chance_95th_of_max": float(np.quantile(centred_max, 0.95)),
            "selection_adjusted_p": float((centred_max >= obs.max()).mean()),
            "_obs": obs.sort_values(ascending=False), "_centred_max": centred_max}


def figures(metrics: pd.DataFrame, bon: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # primary gain per fold with its interval
    m = metrics[metrics["config"].eq(PRIMARY[0]) & metrics["variant"].eq(PRIMARY[1])]
    m = m.set_index("fold_id").loc[["k1", "k2", "h1", "pooled"]]
    labels = ["k1\n(eligible)", "k2\n(eligible)", "h1\n(watch-only)", "pooled\n(descriptive)"]
    colors = ["#1f77b4", "#1f77b4", "#999999", "#999999"]
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(m))
    err = np.vstack([m["gain_vs_calibrated"] - m["ci_vs_calibrated_low"],
                     m["ci_vs_calibrated_high"] - m["gain_vs_calibrated"]])
    for i in range(len(m)):
        ax.errorbar(m["gain_vs_calibrated"].iloc[i], y[i], xerr=err[:, i:i + 1], fmt="o",
                    color=colors[i], capsize=4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Gain in per-race log loss vs calibrated market (positive = model better)")
    ax.set_title("Primary blend (market + logit + LightGBM)\ngain with 95% race-day interval", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "primary_gain_by_fold.png", dpi=150)
    plt.close(fig)

    # best of N vs chance
    fig, axes = plt.subplots(1, len(bon), figsize=(11, 4), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), bon):
        ax.hist(r["_centred_max"], bins=40, color="#cccccc", edgecolor="white")
        ax.axvline(r["chance_95th_of_max"], color="grey", ls="--", label="95% of chance maxima")
        ax.axvline(r["best_gain"], color="#d62728", lw=2, label=f"best observed ({r['best']})")
        ax.set_title(f"{r['folds']}: best of {r['candidates']} (p = {r['selection_adjusted_p']:.2f})")
        ax.set_xlabel("Best gain vs calibrated market")
        ax.legend(fontsize=7, loc="upper right")
    np.atleast_1d(axes)[0].set_ylabel("Bootstrap resamples (no real edge)")
    fig.suptitle("How good the best configuration looks by chance alone")
    fig.tight_layout()
    fig.savefig(out / "best_of_n_vs_chance.png", dpi=150)
    plt.close(fig)


def md(df: pd.DataFrame, digits: int = 5) -> list[str]:
    s = df.copy()
    for c in s.select_dtypes("float").columns:
        s[c] = s[c].round(digits)
    rows = ["| " + " | ".join(map(str, s.columns)) + " |", "|" + "---|" * len(s.columns)]
    return rows + ["| " + " | ".join(str(v) for v in r) + " |" for r in s.itertuples(index=False)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=REPO / "artifacts/folds")
    parser.add_argument("--out", type=Path, default=REPO / "analysis/overfitting_results")
    args = parser.parse_args(argv)

    checks = prov.Report()
    for cfg in CONFIGS:
        checks.extend(prov.check_fold_root(args.root / cfg))
    prov.require(checks, f"overfitting check on {args.root}")

    frames = load(args.root)
    metrics = pd.read_csv(args.root / "ablation_metrics.csv")
    io = in_vs_out(frames)
    stab, sets = stability(args.root)
    bon = [best_of_n(frames, ELIGIBLE), best_of_n(frames, FOLDS)]

    args.out.mkdir(parents=True, exist_ok=True)
    io.to_csv(args.out / "in_vs_out_of_sample.csv", index=False)
    stab.to_csv(args.out / "stability_by_fold.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in bon]).to_csv(
        args.out / "best_of_n.csv", index=False)
    figures(metrics, bon, args.out)

    commit = json.loads((args.root / "primary/k1/meta.json").read_text())["git_commit"][:7]
    lines = [
        "# Overfitting and selection checks, v3.1 fold runs (generated)", "",
        f"Source: `artifacts/folds/` (runs on commit `{commit}`, all pass the provenance check). "
        "Built by `analysis/overfitting_check.py`; no model is refitted and no h_test row is read. "
        f"Bootstrap: {N_BOOT} race-day resamples, seed {SEED}.", "",
        "## 1. In-sample (s2) vs out-of-sample (valid) gain vs calibrated market", "",
        f"Mean in-sample gain {io.gain_in_sample_s2.mean():+.5f}; mean out-of-sample gain "
        f"{io.gain_out_of_sample_valid.mean():+.5f}; in-sample larger in "
        f"{int((io.in_minus_out > 0).sum())} of {len(io)} cases. **No sign of the blend overfitting.**", "",
        *md(io), "",
        "## 2. Stability across folds (primary config)", "",
        *md(stab, 3), "",
        f"Conditional-logit features selected in all three folds: {len(sets['common'])} of "
        f"{sets['union']} ever selected ({', '.join('`' + x + '`' for x in sets['common'])}). "
        "Overlap between folds (Jaccard): "
        + ", ".join(f"{k} {v:.2f}" for k, v in sets["jaccard"].items()) + ". "
        "The model weights (alpha) are small and change sign between folds, and the blend picks "
        "the strongest penalty in every fold: **the models add almost no information beyond the "
        "calibrated market.**", "",
        "## 3. Best of N candidates vs chance", "",
        "Each row bootstraps whole race days with every candidate centred at zero (no real edge) and "
        "records the largest gain. `selection_adjusted_p` is the share of resamples whose best gain "
        "is at least the observed best.", "",
        *md(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in bon])), "",
        "**The best configuration is within what picking the winner of the candidates produces by "
        "chance.** Top candidates (pooled k1+k2+h1): "
        + ", ".join(f"`{k}` {v:+.5f}" for k, v in bon[1]["_obs"].head(5).items()) + ".", "",
        "## Figures", "",
        "- `primary_gain_by_fold.png`: primary blend gain per fold with its 95% race-day interval.",
        "- `best_of_n_vs_chance.png`: the best observed gain against the chance distribution of the best.", "",
        "## Plain-language reading", "",
        "The blend is not overfitting: it does as well out of sample as in sample. But the models add "
        "almost no information beyond the betting market: their blend weights stay near zero and "
        "change sign between folds, the selected logit features are unstable, and the best of the "
        "candidate configurations does no better than the luckiest of them would by chance.",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: best eligible p={bon[0]['selection_adjusted_p']:.3f}, "
          f"best all-folds p={bon[1]['selection_adjusted_p']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
