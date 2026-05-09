"""Plot exp 11 refusal-probe results: per-pooling fold AUCs + Arditi projection AUC."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results.json"


def main():
    r = json.loads(RESULTS.read_text())
    poolings = list(r["per_pooling"].keys())

    # bar chart: mean AUC ± std per pooling, with Arditi projection AUC overlay
    means = [r["per_pooling"][p]["auc_mean"] for p in poolings]
    stds  = [r["per_pooling"][p]["auc_std"]  for p in poolings]
    arditi = r["arditi_projection_auc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(poolings))
    ax.bar(x, means, yerr=stds, capsize=6, color=["tab:blue", "tab:orange"],
           alpha=0.85, edgecolor="black")
    ax.axhline(arditi, ls="--", color="tab:red", label=f"Arditi-direction projection ({arditi:.3f})")
    ax.axhline(0.5, ls=":", color="grey", label="chance")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(x[i], m + s + 0.005, f"{m:.4f} ± {s:.4f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(poolings)
    ax.set_ylabel("ROC-AUC (5-fold CV, ±1σ)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"Gemma 4-31B-it refusal probe — layer {r['layer']}\n"
                 f"({r['n_samples']} samples, {r['n_pos']}/{r['n_neg']} pos/neg)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(HERE / "auc_by_pooling.png", dpi=150)
    plt.close(fig)
    print(f"wrote {HERE/'auc_by_pooling.png'}")

    # per-fold AUC strip
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, p in enumerate(poolings):
        fm = r["per_pooling"][p]["fold_metrics"]
        aucs = [m["auc"] for m in fm]
        ax.scatter([i] * len(aucs), aucs, s=60, alpha=0.85, label=p)
        ax.scatter([i], [r["per_pooling"][p]["auc_mean"]], marker="_", s=600, color="black")
    ax.axhline(arditi, ls="--", color="tab:red", alpha=0.6, label=f"Arditi proj ({arditi:.3f})")
    ax.set_xticks(range(len(poolings))); ax.set_xticklabels(poolings)
    ax.set_ylabel("Fold AUC")
    ax.set_ylim(0.85, 1.0)
    ax.set_title("Per-fold refusal-probe AUC (5 folds, dot = fold; bar = mean)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fold_auc.png", dpi=150)
    plt.close(fig)
    print(f"wrote {HERE/'fold_auc.png'}")


if __name__ == "__main__":
    main()
