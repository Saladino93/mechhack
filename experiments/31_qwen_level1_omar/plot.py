"""Plot AUC-vs-layer for the four trainable heads, mark the cc_concat baseline."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent
res = json.loads((OUT_DIR / "results.json").read_text())

LAYERS = res["layers"]
fig, ax = plt.subplots(figsize=(13, 7.5))

# Slight x-jitter per arch so error bars don't stack on each other
ARCH_ORDER = [
    "arditi",
    "linear_mean", "linear_last",
    "linear_sklearn_mean", "linear_sklearn_last",
    "mlp_mean",
    "multimax_simple", "multimax_kramar",
    "attention_deepmind",
    "multimax_simple_tuned", "multimax_kramar_tuned",
    "attention_deepmind_tuned",
]
JITTER_STEP = 0.5  # in layer-index units (layers are 5 apart)
JITTER = {a: (i - len(ARCH_ORDER)/2) * (JITTER_STEP / max(len(ARCH_ORDER)/2, 1))
          for i, a in enumerate(ARCH_ORDER)}
COLORS = {
    "arditi":              "#22577a",
    "linear_mean":         "#2b6cb0",
    "linear_last":         "#319795",
    "linear_sklearn_mean": "#1e40af",
    "linear_sklearn_last": "#0d9488",
    "mlp_mean":            "#d97706",
    "attention":           "#9f1239",
    "multimax":            "#7c3aed",
    "multimax_simple":          "#a78bfa",
    "multimax_kramar":          "#7c3aed",
    "attention_deepmind":       "#be185d",
    "multimax_simple_tuned":    "#a78bfa",
    "multimax_kramar_tuned":    "#5b21b6",
    "attention_deepmind_tuned": "#831843",
}
LINESTYLES = {  # tuned variants get dashed
    "multimax_simple_tuned":    "--",
    "multimax_kramar_tuned":    "--",
    "attention_deepmind_tuned": "--",
}
for arch, info in res["by_arch"].items():
    if "per_layer" not in info: continue
    pts = []
    for L in LAYERS:
        s = info["per_layer"].get(str(L))
        if not s: continue
        # different schemas: trainable=auc_seedavg, arditi=auc, sklearn=test_auc
        auc = s.get("auc_seedavg", s.get("auc", s.get("test_auc")))
        if auc is None: continue
        pts.append((L, auc, s["ci95"][0], s["ci95"][1]))
    if not pts: continue
    Ls, aucs, los, his = zip(*pts)
    Ls = np.array(Ls, dtype=float) + JITTER.get(arch, 0.0)
    ls = LINESTYLES.get(arch, "-")
    ax.errorbar(Ls, aucs,
                yerr=[np.array(aucs)-np.array(los), np.array(his)-np.array(aucs)],
                fmt=f"{ls}o", capsize=2, markersize=4, lw=1.2,
                label=arch, color=COLORS.get(arch, None), alpha=0.9)

# CC++ concat as horizontal band
cc = res["by_arch"].get("cc_concat")
if cc:
    ax.axhspan(cc["ci95"][0], cc["ci95"][1], color="grey", alpha=0.15,
               label=f"cc_concat (13×) AUC={cc['test_auc']:.3f}")
    ax.axhline(cc["test_auc"], color="grey", lw=1, ls="--")

# TF-IDF baselines as horizontal lines
tfidf = res.get("baseline_tfidf", {})
for name, color, ls in [("word_1_2gram", "#16a34a", "-."),
                          ("char_3_5gram", "#84cc16", ":")]:
    info = tfidf.get(name)
    if not info: continue
    ax.axhline(info["test_auc"], color=color, lw=1.5, ls=ls,
               label=f"TF-IDF {name} AUC={info['test_auc']:.3f}")

ax.axhline(0.5, color="black", lw=0.5, alpha=0.4)
ax.set_xlabel("Layer index")
ax.set_ylabel("Test AUC")
ax.set_title(f"Qwen 3.6-27B Refusal — probe AUC vs layer "
             f"(test n={res['n_test']}, pos={res['pos_rate_test']:.2f})")
ax.set_xticks(LAYERS)
ax.set_ylim(0.4, 1.0)
ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.92)
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = OUT_DIR / "auc_by_layer.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("Saved", out)
