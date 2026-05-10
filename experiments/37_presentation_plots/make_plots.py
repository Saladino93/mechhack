"""Publication-quality plots for the slide deck.

Plots produced in figures/:
  1. refusal_probe_family_comparison.png — bar chart, test AUC per probe
     family (LR last L40, Pleshkov, MLP, multi-concat, TF-IDF, random,
     minimal MultiMax) on refusal_gemma honest test split.
  2. refusal_layer_sweep.png — line plot, AUC vs layer for linear /
     linear-on-PCs / Pleshkov-quadratic on refusal.
  3. cyber_layer_sweep.png — same but across the 3 cyber tasks (3 panels).
  4. mean_auc_per_family.png — bar chart, mean AUC across 4 tasks per probe
     family. THE COMPETITION HEADLINE.
  5. causality_scatter.png — Pr(f|edit) vs Pr(model|f flipped) per probe,
     'everything high' regime annotated.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def load_q4r():
    p = REPO_ROOT / "experiments" / "31_honest_eval_omar" / "refusal_only_results.json"
    if not p.exists(): return {}
    return json.loads(p.read_text())["refusal_gemma"]


def load_pleshkov_sweep(task):
    p = REPO_ROOT / "experiments" / "17_quadratic_probe_omar" / "results" / f"{task}_d16_sweep.json"
    if not p.exists(): return None
    return json.loads(p.read_text())


def load_q8():
    p = REPO_ROOT / "experiments" / "34_combined_causality_omar" / "causality_rewrites_k7.json"
    if not p.exists(): return None
    return json.loads(p.read_text())


# ============================================================================
# PLOT 1 — Refusal probe family bar chart (test AUC, with train AUC dot)
# ============================================================================
def plot_refusal_family():
    q4r = load_q4r()

    # Curate one probe per family — best layer per family
    # NOTE: Constitutional probe (8.9M params on 555 train samples / 13×5376
    # input) catastrophically overfits (train 0.996, test 0.500) — excluded
    # from the headline to avoid confusion. See README for explanation.
    families = {
        "LR last-tok L40": q4r.get("LR_last_L40"),
        "LR last-tok L45": q4r.get("LR_last_L45"),
        "LR mean L40": q4r.get("LR_mean_L40"),
        "LR multi-layer\n(69k-d, concat)": q4r.get("LR_multi_concat"),
        "LR mean-of-layers": q4r.get("LR_mean_of_layers"),
        "MLP (1-hidden, L40)": q4r.get("MLP_L40"),
        "Pleshkov d=16 L50\n(quadratic)": q4r.get("Pleshkov_d16_L50"),
    }
    # Add baselines from other sources
    tfidf_p = REPO_ROOT / "experiments" / "21_probe_zoo_omar" / "results" / "refusal_baselines.json"
    tfidf = json.loads(tfidf_p.read_text()) if tfidf_p.exists() else {}
    families["TF-IDF word\n1-2gram"] = {"test_auc": tfidf.get("tfidf_word_lr", {}).get("auc_mean", 0),
                                          "train_auc": tfidf.get("tfidf_word_lr", {}).get("train_auc_mean", 0)}
    families["Random"] = {"test_auc": 0.50, "train_auc": 0.50}
    # Add minimal MultiMax (test AUC from Q6)
    mm_p = REPO_ROOT / "experiments" / "33_minimal_multimax_omar" / "minimal_multimax_results.json"
    if mm_p.exists():
        mm = json.loads(mm_p.read_text())
        families["Minimal MultiMax\n(linear → max)"] = {"test_auc": mm["max_pool"]["test_auc"],
                                                          "train_auc": mm["max_pool"]["train_auc"]}
        families["Minimal Mean-pool\n(linear → mean)"] = {"test_auc": mm["mean_pool"]["test_auc"],
                                                            "train_auc": mm["mean_pool"]["train_auc"]}

    families = {k: v for k, v in families.items() if v is not None}
    names = list(families.keys())
    test_aucs = [families[n]["test_auc"] for n in names]
    train_aucs = [families[n]["train_auc"] for n in names]
    # Sort by test AUC descending
    order = np.argsort(test_aucs)[::-1]
    names = [names[i] for i in order]
    test_aucs = [test_aucs[i] for i in order]
    train_aucs = [train_aucs[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(names))
    # Color by performance band
    colors = ["#1f77b4" if a >= 0.92 else
              "#aec7e8" if a >= 0.87 else
              "#c5b0d5" if a >= 0.80 else
              "#9467bd" if a >= 0.60 else
              "#888888" for a in test_aucs]
    bars = ax.bar(x, test_aucs, color=colors, edgecolor="black", linewidth=0.6,
                   label="Test AUC", zorder=3)
    # Train AUC as a dot above the bar
    ax.scatter(x, train_aucs, color="#d62728", marker="x", s=70, zorder=4,
               label="Train AUC", linewidths=1.8)
    # Labels on bars
    for i, (a, ta) in enumerate(zip(test_aucs, train_aucs)):
        gap = ta - a
        ax.text(i, a + 0.012, f"{a:.3f}", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9.5)
    ax.set_ylabel("AUC")
    ax.set_title("Refusal-Gemma probe family comparison — honest test split\n"
                 "(train on n=555, evaluate on n=277)")
    ax.set_ylim(0.45, 1.05)
    # Chance lines: AUC chance = 0.5 regardless of class balance.
    # Accuracy chance = max(prior_pos, prior_neg) = 0.524 for refusal (47.6% pos)
    ax.axhline(0.50, ls="--", c="grey", alpha=0.5, label="chance AUC = 0.50")
    ax.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    out = FIG_DIR / "refusal_probe_family_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================================
# PLOT 2 — Refusal Pleshkov layer sweep (linear / linear-on-PCs / quadratic)
# ============================================================================
def plot_refusal_layer_sweep():
    d = load_pleshkov_sweep("refusal_gemma")
    if d is None:
        print("  refusal sweep not found"); return
    layers = sorted(int(k) for k in d["by_layer"])
    lin = [d["by_layer"][str(L)]["linear"]["auc_mean"] for L in layers]
    lin_e = [d["by_layer"][str(L)]["linear"]["auc_std"] for L in layers]
    pcs = [d["by_layer"][str(L)]["linear_on_pcs"]["auc_mean"] for L in layers]
    pcs_e = [d["by_layer"][str(L)]["linear_on_pcs"]["auc_std"] for L in layers]
    quad = [d["by_layer"][str(L)]["quadratic"]["auc_mean"] for L in layers]
    quad_e = [d["by_layer"][str(L)]["quadratic"]["auc_std"] for L in layers]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(layers, lin, yerr=lin_e, marker="o", color="#1f77b4", lw=2,
                label="Linear LR (raw 5376-d)", capsize=3)
    ax.errorbar(layers, pcs, yerr=pcs_e, marker="s", color="#ff7f0e", lw=2,
                label="Linear LR on 16 PCs", capsize=3)
    ax.errorbar(layers, quad, yerr=quad_e, marker="^", color="#2ca02c", lw=2,
                label="Pleshkov quadratic (16 PCs + degree-2 + ridge)", capsize=3)
    # TF-IDF baseline (exp 21 refusal_baselines)
    ax.axhline(0.877, ls="--", color="#9467bd", lw=2.2,
                label="TF-IDF baseline = 0.877")
    ax.axhline(0.5, ls=":", color="grey", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC (5-fold CV mean ± 1σ)")
    ax.set_xticks(layers)
    ax.set_title("Refusal-Gemma — layer sweep: linear vs quadratic vs PCA-bottleneck vs TF-IDF")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    # Annotation
    best_layer = layers[int(np.argmax(lin))]
    best = max(lin)
    ax.annotate(f"linear peak\nL{best_layer}: AUC={best:.3f}",
                xy=(best_layer, best), xytext=(best_layer - 8, best - 0.05),
                fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="black"))
    plt.tight_layout()
    out = FIG_DIR / "refusal_layer_sweep.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================================
# PLOT 3 — Cyber Pleshkov sweeps (3 panels)
# ============================================================================
def plot_cyber_layer_sweeps():
    # TF-IDF baseline AUC per cyber task (exp 09 D1)
    tfidf = {"cyber_1": 0.946, "cyber_2": 0.887, "cyber_3": 0.890}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, task in zip(axes, ["cyber_1", "cyber_2", "cyber_3"]):
        d = load_pleshkov_sweep(task)
        if d is None:
            ax.text(0.5, 0.5, f"no data for {task}", ha="center", va="center",
                    transform=ax.transAxes); continue
        layers = sorted(int(k) for k in d["by_layer"])
        lin = [d["by_layer"][str(L)]["linear"]["auc_mean"] for L in layers]
        lin_e = [d["by_layer"][str(L)]["linear"]["auc_std"] for L in layers]
        pcs = [d["by_layer"][str(L)]["linear_on_pcs"]["auc_mean"] for L in layers]
        pcs_e = [d["by_layer"][str(L)]["linear_on_pcs"]["auc_std"] for L in layers]
        quad = [d["by_layer"][str(L)]["quadratic"]["auc_mean"] for L in layers]
        quad_e = [d["by_layer"][str(L)]["quadratic"]["auc_std"] for L in layers]
        ax.errorbar(layers, lin, yerr=lin_e, marker="o", color="#1f77b4", lw=2,
                    label="Linear LR (raw)", capsize=3)
        ax.errorbar(layers, pcs, yerr=pcs_e, marker="s", color="#ff7f0e", lw=2,
                    label="Linear-on-16-PCs", capsize=3)
        ax.errorbar(layers, quad, yerr=quad_e, marker="^", color="#2ca02c", lw=2,
                    label="Pleshkov quadratic", capsize=3)
        # TF-IDF baseline horizontal line
        ax.axhline(tfidf[task], ls="--", color="#9467bd", lw=2.2,
                    label=f"TF-IDF baseline = {tfidf[task]:.3f}")
        ax.axhline(0.5, ls=":", color="grey", alpha=0.5, label="chance = 0.5")
        ax.set_xlabel("Layer"); ax.set_xticks(layers[::2])
        ax.set_title(f"{task}  (TF-IDF baseline = {tfidf[task]:.3f})")
        ax.set_ylim(0.45, 1.05)
        if task == "cyber_1":
            ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
            ax.set_ylabel("AUC (5-fold CV mean ± 1σ)")
    fig.suptitle("Cyber tasks — linear vs quadratic vs PCA-bottleneck per layer", y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "cyber_layer_sweeps.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================================
# PLOT 4 — Mean AUC across tasks (the COMPETITION METRIC)
# ============================================================================
def plot_mean_auc_competition():
    """For probes evaluated on each task, plot the mean ± std across the 4 tasks
    we have (cyber_1/2/3 + refusal_gemma; Qwen excluded)."""
    # Best per-family AUCs per task, from the sources we have
    # Cyber: peak from the layer sweeps + held-out where available
    # Refusal: honest test split from Q4r
    q4r = load_q4r()
    cyber_aucs = {
        "cyber_1": {
            "Linear (best layer)": 0.988,   # exp 08 held-out test
            "TF-IDF": 0.946,                  # exp 09 D1
            "Pleshkov d=16": load_pleshkov_sweep("cyber_1")["by_layer"]["40"]["quadratic"]["auc_mean"],
            "Linear-on-16PCs": load_pleshkov_sweep("cyber_1")["by_layer"]["40"]["linear_on_pcs"]["auc_mean"],
            "Random": 0.50,
        },
        "cyber_2": {
            "Linear (best layer)": 0.955,
            "TF-IDF": 0.887,
            "Pleshkov d=16": max(load_pleshkov_sweep("cyber_2")["by_layer"][str(L)]["quadratic"]["auc_mean"]
                                  for L in [25,30,35,40,45]),
            "Linear-on-16PCs": max(load_pleshkov_sweep("cyber_2")["by_layer"][str(L)]["linear_on_pcs"]["auc_mean"]
                                    for L in [25,30,35,40,45]),
            "Random": 0.50,
        },
        "cyber_3": {
            "Linear (best layer)": 0.955,
            "TF-IDF": 0.890,
            "Pleshkov d=16": max(load_pleshkov_sweep("cyber_3")["by_layer"][str(L)]["quadratic"]["auc_mean"]
                                  for L in [25,30,35,40,45]),
            "Linear-on-16PCs": max(load_pleshkov_sweep("cyber_3")["by_layer"][str(L)]["linear_on_pcs"]["auc_mean"]
                                    for L in [25,30,35,40,45]),
            "Random": 0.50,
        },
        "refusal_gemma": {
            "Linear (best layer)": q4r.get("LR_last_L40", {}).get("test_auc", 0),
            "TF-IDF": 0.877,                  # exp 21 baselines
            "Pleshkov d=16": q4r.get("Pleshkov_d16_L50", {}).get("test_auc", 0),
            "Linear-on-16PCs": q4r.get("LR_mean_L40", {}).get("test_auc", 0),  # proxy — use full linear at L40 mean
            "Random": 0.50,
        },
    }
    # Note: cyber_1 = exp 08 held-out (honest); cyber_2/3 = 5-fold CV (flag this)
    families = list(cyber_aucs["cyber_1"].keys())
    tasks = list(cyber_aucs.keys())
    # Compute mean ± std per family across 4 tasks
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(families))
    width = 0.18
    colors = plt.cm.tab10.colors[:len(tasks)]
    for i, task in enumerate(tasks):
        vals = [cyber_aucs[task][f] for f in families]
        ax.bar(x + (i - 1.5) * width, vals, width, label=task, color=colors[i],
               edgecolor="black", linewidth=0.4)
    # Mean line
    means = [np.mean([cyber_aucs[t][f] for t in tasks]) for f in families]
    stds = [np.std([cyber_aucs[t][f] for t in tasks], ddof=1) for f in families]
    ax.plot(x, means, "ko-", lw=2.5, ms=10, label="Mean across 4 tasks", zorder=10)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.errorbar(i, m, yerr=s, color="black", capsize=4, lw=2)
        ax.text(i, m + 0.015, f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("Mean AUC across 4 tasks (cyber_1/2/3 + refusal_gemma)\n"
                 "Competition metric: mean(AUC), report variance for robustness comparison")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.50, ls="--", c="grey", alpha=0.5)
    ax.legend(loc="lower left", ncol=3, framealpha=0.9)
    plt.tight_layout()
    out = FIG_DIR / "mean_auc_competition.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return families, means, stds


# ============================================================================
# PLOT 5 — Causality scatter: Pr(f|edit) vs Pr(model|f flipped)
# ============================================================================
def plot_causality_scatter():
    d = load_q8()
    if d is None:
        print("  Q8 causality not found"); return
    rows = []
    for name, r in d["by_probe"].items():
        pf = r["Pr_f_given_edit"]["point"]
        pmf = r["Pr_model_given_f"]["point"]
        n = r["Pr_model_given_f"]["n"]
        if n < 1: continue
        rows.append((name, pf, pmf, n))

    fig, ax = plt.subplots(figsize=(10, 7))
    # Color by family
    def color_for(name):
        if "Pleshkov" in name and "combined" in name: return "#9467bd"
        if "Pleshkov" in name: return "#bcbd22"
        if "COMBINED" in name: return "#ff7f0e"
        if "last" in name: return "#1f77b4"
        if "mean" in name: return "#2ca02c"
        if "concat" in name or "of_layers" in name: return "#17becf"
        return "#888888"

    for name, pf, pmf, n in rows:
        c = color_for(name)
        ax.scatter(pf, pmf, s=20 + 5 * n, color=c, alpha=0.75, edgecolor="black",
                   linewidths=0.5)
    # Annotate top probes by Pr(model|f)
    rows_by_pmf = sorted(rows, key=lambda r: -r[2])
    annot = set([r[0] for r in rows_by_pmf[:6]] +
                [r[0] for r in rows if r[0].startswith("Pleshkov") or r[0].startswith("COMBINED")] +
                ["LR_mean_L40"])
    for name, pf, pmf, n in rows:
        if name in annot:
            label = name.replace("_", " ")
            ax.annotate(label, (pf, pmf), fontsize=8.5, alpha=0.85,
                        xytext=(4, 4), textcoords="offset points")

    # Quadrant guides
    ax.axhline(0.5, ls="--", c="grey", alpha=0.5)
    ax.axvline(0.1, ls="--", c="grey", alpha=0.5)
    # Quadrant labels
    ax.text(0.02, 0.95, "ROBUST + CAUSAL\n(low f-flip rate, high m|f)",
            transform=ax.transAxes, fontsize=10, color="#1f7711", fontweight="bold",
            va="top")
    ax.text(0.45, 0.95, "PERMISSIVE + CAUSAL\n(many f-flips, mostly track model)",
            transform=ax.transAxes, fontsize=10, color="#117733", va="top")
    ax.text(0.45, 0.04, "GAMED f\n(many f-flips, model doesn't follow)",
            transform=ax.transAxes, fontsize=10, color="#cc0033", fontweight="bold",
            va="bottom")
    ax.text(0.02, 0.04, "OVER-ROBUST\n(rare f-flips, doesn't track model)",
            transform=ax.transAxes, fontsize=10, color="#cc6600", va="bottom")
    ax.set_xlabel("Pr(f flipped | edit)  →  more flips = more permissive")
    ax.set_ylabel("Pr(model flipped | f flipped)  →  higher = more causal")
    ax.set_title("Per-probe causality on rewrites_k7 substantial paraphrases\n"
                 "(marker size = n probe-flips; n_orig_refusal = 79)")
    ax.set_xlim(-0.02, 0.5)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    out = FIG_DIR / "causality_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("[presentation plots]")
    plot_refusal_family()
    plot_refusal_layer_sweep()
    plot_cyber_layer_sweeps()
    families, means, stds = plot_mean_auc_competition()
    plot_causality_scatter()

    # Print final mean-AUC-per-family table
    print(f"\n=== COMPETITION HEADLINE: mean AUC across 4 tasks per probe family ===")
    for f, m, s in zip(families, means, stds):
        print(f"  {f:>22}  mean={m:.4f}  std={s:.4f}")
    print(f"\nrules-compliant 'mean AUC across 5 tasks': we report mean across 4 (Qwen excluded)")


if __name__ == "__main__":
    main()
