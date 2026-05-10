"""Pr(model flipped | edit) bar plot — flip.md's headline 'honest measure'.

Per flip.md: 'Pr(model flipped|edit) — re-rolling the model on the final
edited prompt yields compliance. The honest measure; this is the headline.'

This metric is a property of the (edit_method × model) pair, NOT of the
probe. Same Pr(model|edit) regardless of which probe was used to score
the edits.
"""
from __future__ import annotations
import json
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 150,
})


def wilson(k, n, z=1.96):
    if n == 0: return 0, 0, 0
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den; s = z/den * sqrt(p*(1-p)/n + z*z/(4*n*n))
    return p, max(0, c-s), min(1, c+s)


# Load Phase 3c per-edit-method numbers
phase3 = json.loads((REPO_ROOT / "experiments" / "13_pre_rewrites_omar" / "phase3_summary.json").read_text())
# Q9 swap result
q9 = json.loads((REPO_ROOT / "experiments" / "35_prompt_start_swap_omar" / "swap_results.json").read_text())
# Original PRE rewrites_k7 result
level2 = json.loads((REPO_ROOT / "experiments" / "13_pre_rewrites_omar" / "level2_metrics.json").read_text())

methods = []
for k, v in phase3.items():
    methods.append((k.replace("_", " "), v["Pr_model_given_edit"]["k"], v["Pr_model_given_edit"]["n"]))
methods.insert(0, ("first-word swap\n(during → I'm working on)", q9["n_heuristic_flipped"], q9["n_target_prompts"]))
# rewrites_k7 from level2_metrics — Pr_model_given_edit
pme = level2["Pr_model_flipped_given_edit"]
methods.append(("substantial paraphrases\n(rewrites_k7, k=7)", pme["n_yes"], pme["n_total"]))

names = [m[0] for m in methods]
pts, los, his = zip(*[wilson(m[1], m[2]) for m in methods])
ks = [m[1] for m in methods]; ns = [m[2] for m in methods]

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(methods))
colors = ["#cc0033"] + ["#ff7f0e"]*4 + ["#2ca02c"]
err_lo = [p - lo for p, lo in zip(pts, los)]
err_hi = [hi - p for p, hi in zip(pts, his)]
ax.bar(x, pts, color=colors, edgecolor="black", linewidth=0.8)
ax.errorbar(x, pts, yerr=[err_lo, err_hi], fmt="none", color="black", capsize=5, lw=1.5)
for i, (p, k, n) in enumerate(zip(pts, ks, ns)):
    ax.text(i, p + max(0.02, err_hi[i] + 0.01), f"{k}/{n}\n{p:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
ax.set_ylabel("Pr(model flipped | edit)")
ax.set_title("Pr(model flipped | edit) — flip.md's honest headline metric\n"
             "Bars = point estimate; whiskers = Wilson 95% CI")
ax.set_ylim(0, 0.55)
# Group separators / annotations
ax.axvspan(-0.5, 0.5, alpha=0.05, color="red", zorder=-10)
ax.axvspan(0.5, 4.5, alpha=0.05, color="orange", zorder=-10)
ax.axvspan(4.5, 5.5, alpha=0.05, color="green", zorder=-10)
ax.text(0, 0.50, "single-word\n(none flip)", ha="center", fontsize=10, color="#cc0033")
ax.text(2.5, 0.50, "minimal edits\n(1.3% avg)", ha="center", fontsize=10, color="#cc6600")
ax.text(5, 0.50, "substantial\nparaphrases\n(31%)", ha="center", fontsize=10, color="#117733",
        fontweight="bold")
ax.axhline(0.5, ls="--", c="grey", alpha=0.3)

plt.tight_layout()
out = FIG_DIR / "pr_model_edit.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
for n, p, lo, hi, k, t in zip(names, pts, los, his, ks, ns):
    print(f"  {n[:50]:>50}  {k}/{t} = {p:.3f}  CI=[{lo:.3f},{hi:.3f}]")
