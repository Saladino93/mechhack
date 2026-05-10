"""Kramár-style FNR @ fixed FPR plot — the threshold-shift on long contexts.

Reproduces the headline figure from Kramár 2026 §3 Fig 2: hold the FPR
fixed (calibrated on short bucket = 1% or 5%), then measure FNR on each
length bucket. AUC stays flat across length, but FNR @ low FPR explodes.

Data: experiments/15_refusal_length_strat_omar/kramar_results.json
       (LR mean L32, n=832 refusal prompts)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 150,
})

K = json.loads((REPO_ROOT / "experiments" / "15_refusal_length_strat_omar" / "kramar_results.json").read_text())

buckets = ["short", "medium", "long"]
auc_per_bucket = [K["auc_per_bucket"][b] for b in buckets]
fnr_1pct = [K["fnr_at_fixed_fpr"]["fpr_target_0.01"]["buckets"][b]["fnr"] for b in buckets]
fnr_5pct = [K["fnr_at_fixed_fpr"]["fpr_target_0.05"]["buckets"][b]["fnr"] for b in buckets]


fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel 1 — AUC per bucket (looks fine; misleading)
ax = axes[0]
x = np.arange(3)
ax.bar(x, auc_per_bucket, color=["#2ca02c", "#ff7f0e", "#d62728"],
       edgecolor="black", linewidth=0.8, label="AUC")
for i, v in enumerate(auc_per_bucket):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
ax.axhline(0.5, ls="--", c="grey", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels([f"{b}\nn={K['bucket_counts'][b]['total']}" for b in buckets])
ax.set_ylabel("AUC")
ax.set_ylim(0.5, 1.0)
ax.set_title("AUC per length tertile\n(looks stable — but is misleading)")

# Panel 2 — FNR @ 1% / 5% FPR (the real story)
ax = axes[1]
width = 0.35
ax.bar(x - width/2, fnr_1pct, width, label="FNR @ 1% FPR (strict)",
       color="#cc0033", edgecolor="black", linewidth=0.8)
ax.bar(x + width/2, fnr_5pct, width, label="FNR @ 5% FPR (lenient)",
       color="#ff9966", edgecolor="black", linewidth=0.8)
for i, (v1, v5) in enumerate(zip(fnr_1pct, fnr_5pct)):
    ax.text(i - width/2, v1 + 0.02, f"{v1:.0%}", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
    ax.text(i + width/2, v5 + 0.02, f"{v5:.0%}", ha="center", va="bottom",
            fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n_pos={K['bucket_counts'][b]['n_pos']})" for b in buckets])
ax.set_ylabel("False Negative Rate")
ax.set_ylim(0, 1.0)
ax.set_title("FNR @ fixed FPR (threshold calibrated on SHORT)\nKramár 2026 threshold-shift: long-context collapse")
ax.legend(loc="upper left", framealpha=0.95)

# Annotate the headline jump
ax.annotate("", xy=(2 - width/2, 0.795), xytext=(0 - width/2, 0.494),
            arrowprops=dict(arrowstyle="->", color="#cc0033", lw=2.5))
ax.text(1, 0.65, "+30 pp FNR jump\non long context",
        ha="center", fontsize=11, color="#cc0033", fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#cc0033", alpha=0.9))

fig.suptitle("Refusal LR mean L32 — same probe, three length buckets:\n"
             "AUC seems robust, but production threshold (1% FPR) collapses on long contexts",
             y=1.04, fontsize=13, fontweight="bold")
plt.tight_layout()
out = FIG_DIR / "fnr_kramar_threshold_shift.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
print(f"\nFNR @ 1% FPR:  short={fnr_1pct[0]:.0%}  medium={fnr_1pct[1]:.0%}  long={fnr_1pct[2]:.0%}")
print(f"FNR @ 5% FPR:  short={fnr_5pct[0]:.0%}  medium={fnr_5pct[1]:.0%}  long={fnr_5pct[2]:.0%}")
