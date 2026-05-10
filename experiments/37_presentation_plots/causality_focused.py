"""Focused causality plot — 5 probes × 2 edit aggressiveness levels.

Each point = (probe, edit-aggressiveness). Color = probe family;
shape = edit aggressiveness. Size = n probe-flips.

Quadrants annotated:
  top-left  = robust + causal
  top-right = permissive + causal
  bot-right = gamed
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 150,
})

PROBES = [
    ("LR mean L40",        "lr_mean_L40",        "LR_mean_L40",        "#1f77b4"),
    ("LR last-tok L40",    "lr_last_L40",        "LR_last_L40",        "#ff7f0e"),
    ("LR last-tok L45",    "lr_last_L45",        "LR_last_L45",        "#2ca02c"),
    ("LR multi-concat",    "lr_multi_concat",    "LR_multi_concat",    "#17becf"),
    ("Pleshkov d=16 L40 (refusal-only)",  None, "Pleshkov_d16_L40_refusal", "#9467bd"),
    ("Pleshkov d=16 L40 (cyber+refusal)",  None, "Pleshkov_d16_L40_combined", "#bcbd22"),
    ("COMBINED L40 last (cyber+refusal)",  None, "COMBINED_L40_last",  "#d62728"),
    ("LR mean L15 (early layer)", None, "LR_mean_L15", "#8c564b"),
    ("LR last L5 (very early)",   None, "LR_last_L5",  "#e377c2"),
    ("LR last L20 (mid-early)",   None, "LR_last_L20", "#7f7f7f"),
]


def main():
    minimal = json.loads((REPO_ROOT / "experiments/24_robustness_omar/robustness_summary.json").read_text())["by_probe"]
    substantial = json.loads((REPO_ROOT / "experiments/34_combined_causality_omar/causality_rewrites_k7.json").read_text())["by_probe"]

    fig, ax = plt.subplots(figsize=(11, 7.5))

    # Quadrant shading — boundaries at Pr(f|edit)=0.10 (vline) and Pr(m|f)=0.5
    ax.axhspan(0.5, 1.05, xmin=0,    xmax=0.222, alpha=0.07, color="green")  # top-left robust+causal
    ax.axhspan(0.5, 1.05, xmin=0.222, xmax=1.0,  alpha=0.10, color="green")  # top-right TARGET
    ax.axhspan(-0.05, 0.5, xmin=0.222, xmax=1.0, alpha=0.10, color="red")    # bot-right gamed
    ax.axhspan(-0.05, 0.5, xmin=0,    xmax=0.222, alpha=0.07, color="orange")  # bot-left weak

    # Points
    for label, k_min, k_sub, color in PROBES:
        # Minimal-edit point (○)
        if k_min and k_min in minimal:
            r = minimal[k_min]
            pf = r["Pr_f_given_edit"]["point"]
            pmf = r["Pr_model_given_f"]["point"]
            n = r["Pr_model_given_f"]["n"]
            if not (pmf != pmf):  # not nan
                ax.scatter(pf, pmf, s=120 + 25*n, marker="o", facecolor=color,
                           edgecolor="black", linewidth=1.5, alpha=0.85, zorder=5)
                ax.annotate(f"{label}\nminimal n={n}", (pf, pmf),
                             xytext=(5, 6), textcoords="offset points",
                             fontsize=8.5, color=color, alpha=0.85)
        # Substantial-edit point (▲)
        if k_sub and k_sub in substantial:
            r = substantial[k_sub]
            pf = r["Pr_f_given_edit"]["point"]
            pmf = r["Pr_model_given_f"]["point"]
            n = r["Pr_model_given_f"]["n"]
            if not (pmf != pmf):
                ax.scatter(pf, pmf, s=120 + 25*n, marker="^", facecolor=color,
                           edgecolor="black", linewidth=1.5, alpha=0.85, zorder=5)
                ax.annotate(f"{label}\nsubst. n={n}", (pf, pmf),
                             xytext=(5, -20), textcoords="offset points",
                             fontsize=8.5, color=color, alpha=0.85, fontweight="bold")

    # Quadrant labels — boundaries at Pr(f|edit)=0.10 and Pr(m|f)=0.50
    ax.text(0.02, 0.97, "ROBUST + CAUSAL\n(low f-flip, high m|f)",
            transform=ax.transAxes, fontsize=11, color="#117733", fontweight="bold", va="top")
    ax.text(0.32, 0.97, "PERMISSIVE + CAUSAL\n(high f-flip, high m|f)\n← TARGET REGIME ('high everything')",
            transform=ax.transAxes, fontsize=11, color="#117733", fontweight="bold", va="top")
    ax.text(0.32, 0.04, "GAMED f\n(high f-flip, low m|f)\nflip.md: 'gaming f, not causal features'",
            transform=ax.transAxes, fontsize=11, color="#cc0033", fontweight="bold", va="bottom")
    ax.text(0.02, 0.04, "WEAK ATTRIBUTION\n(low f-flip, low m|f)\nflip.md: edit agent is weak",
            transform=ax.transAxes, fontsize=11, color="#cc6600", va="bottom")

    ax.axhline(0.5, ls="--", c="grey", alpha=0.6, lw=1.2)
    ax.axvline(0.10, ls="--", c="grey", alpha=0.4, lw=1)

    # Legend — manual: shape = edit aggressiveness; color = probe family
    edit_handles = [
        Line2D([0], [0], marker="o", color="w", label="Minimal edit (1-3 word delete or DeepSeek 1-round)",
                markerfacecolor="lightgray", markeredgecolor="black", markersize=11),
        Line2D([0], [0], marker="^", color="w", label="Substantial paraphrase (rewrites_k7)",
                markerfacecolor="lightgray", markeredgecolor="black", markersize=11),
    ]
    leg1 = ax.legend(handles=edit_handles, loc="lower right", title="Edit aggressiveness",
                      framealpha=0.95, fontsize=10, title_fontsize=10)
    ax.add_artist(leg1)

    color_handles = [Line2D([0], [0], marker="s", color="w", label=label,
                             markerfacecolor=color, markeredgecolor="black", markersize=11)
                     for label, _, _, color in PROBES]
    ax.legend(handles=color_handles, loc="upper right", title="Probe (color)",
               framealpha=0.95, fontsize=10, title_fontsize=10)

    ax.set_xlabel("Pr(f flipped | edit)   →   probe is more permissive")
    ax.set_ylabel("Pr(model flipped | f flipped)   →   probe is more causal")
    ax.set_title("Causality regimes for 6 probes × 2 edit aggressiveness levels\n"
                  "marker size ∝ √n (number of probe-flips); annotation shows n")
    ax.set_xlim(-0.01, 0.40)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    out = FIG_DIR / "causality_focused.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
