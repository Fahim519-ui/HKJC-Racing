"""Create a compact visual summary of accepted fold and retrospective results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "results_comparison.png"


def main() -> None:
    ablations = pd.read_csv(ROOT / "artifacts/folds/ablation_metrics.csv")
    forward = ablations.loc[
        (ablations["fold_id"] == "pooled") & (ablations["variant"] == "mkt+cl"),
        ["config", "gain_vs_calibrated", "ci_vs_calibrated_low", "ci_vs_calibrated_high"],
    ].copy()
    labels = {
        "primary": "All features",
        "no_track_distance": "No track/distance",
        "no_trainer_rate": "No trainer rate",
        "no_trainer_no_track_distance": "Neither family",
    }
    forward["label"] = forward["config"].map(labels)
    forward = forward.sort_values("gain_vs_calibrated")

    holdout = pd.read_csv(ROOT / "analysis/final_scoring_results/h_test/scoreboard.csv")
    holdout = holdout.loc[
        holdout["model"].isin(["p_market_refit", "p_cl", "p_lgbm", "p_joint"]),
        ["model", "log_loss"],
    ].copy()
    holdout["label"] = holdout["model"].map(
        {
            "p_market_refit": "Calibrated market",
            "p_cl": "Conditional logit",
            "p_lgbm": "LightGBM",
            "p_joint": "Market + models",
        }
    )

    figure, (ax_forward, ax_holdout) = plt.subplots(1, 2, figsize=(11, 4.5))

    gains = forward["gain_vs_calibrated"]
    lower = gains - forward["ci_vs_calibrated_low"]
    upper = forward["ci_vs_calibrated_high"] - gains
    ax_forward.errorbar(
        gains,
        forward["label"],
        xerr=[lower, upper],
        fmt="o",
        color="#1f77b4",
        ecolor="#4c4c4c",
        capsize=3,
    )
    ax_forward.axvline(0, color="#4c4c4c", linewidth=1)
    ax_forward.set_title("Pooled forward folds (927 races)")
    ax_forward.set_xlabel("Gain vs calibrated market\n(per-race log loss; higher is better)")

    colors = ["#7f7f7f", "#d62728", "#ff7f0e", "#1f77b4"]
    ax_holdout.barh(holdout["label"], holdout["log_loss"], color=colors)
    ax_holdout.invert_yaxis()
    ax_holdout.set_title("Retrospective h_test (400 races)")
    ax_holdout.set_xlabel("Per-race log loss (lower is better)")

    figure.suptitle("Accepted experiment results")
    figure.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
