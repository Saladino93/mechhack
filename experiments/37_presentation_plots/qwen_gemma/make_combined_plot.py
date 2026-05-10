"""Side-by-side Gemma vs Qwen refusal-probe AUC by layer.

Reads:
  experiments/31_qwen_level1_omar/results.json  -> Qwen
  experiments/38_cross_model_qwen_omar/gemma_refusal_results.json  -> Gemma

Produces:
  qwen_gemma_refusal_auc_by_layer.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent.parent
QWEN = json.loads((REPO / "experiments/31_qwen_level1_omar/results.json").read_text())
GEMMA = json.loads((REPO / "experiments/38_cross_model_qwen_omar/gemma_refusal_results.json").read_text())
TFIDF_QWEN = QWEN.get("baseline_tfidf", {})

LAYERS = QWEN["layers"]
COLORS = {
    "linear_mean": "#2b6cb0", "linear_last": "#0d9488",
    "mlp_mean": "#d97706", "arditi": "#22577a",
}
ARCHS_PLOT = ["arditi", "linear_mean", "linear_last", "mlp_mean"]


def fetch_curve(res, arch):
    """Returns (Ls, aucs, los, his) for a per-layer arch."""
    info = res["by_arch"].get(arch)
    if info is None or "per_layer" not in info: return None
    pts = []
    for L in LAYERS:
        s = info["per_layer"].get(str(L))
        if not s: continue
        auc = s.get("auc_seedavg", s.get("auc"))
        if auc is None: continue
        pts.append((L, auc, s["ci95"][0], s["ci95"][1]))
    if not pts: return None
    return tuple(np.array(x) for x in zip(*pts))


fig, (ax_g, ax_q) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, res, ttl in [(ax_g, GEMMA, "Gemma 4-31B-it"),
                      (ax_q, QWEN, "Qwen 3.6-27B")]:
    for arch in ARCHS_PLOT:
        c = fetch_curve(res, arch)
        if c is None: continue
        Ls, aucs, los, his = c
        ax.errorbar(Ls, aucs,
                    yerr=[aucs - los, his - aucs],
                    fmt="-o", capsize=2, lw=1.4, markersize=4,
                    label=arch, color=COLORS[arch], alpha=0.92)
    # cc_concat horizontal band
    cc = res["by_arch"].get("cc_concat")
    if cc:
        cc_auc = cc.get("test_auc")
        cc_lo, cc_hi = cc["ci95"]
        ax.axhspan(cc_lo, cc_hi, color="grey", alpha=0.15,
                    label=f"cc_concat (13×) AUC={cc_auc:.3f}")
        ax.axhline(cc_auc, color="grey", lw=1, ls="--")

    # Qwen-only baselines
    if res is QWEN:
        for name, color, ls in [("word_1_2gram", "#16a34a", "-."),
                                  ("char_3_5gram", "#84cc16", ":")]:
            info = TFIDF_QWEN.get(name)
            if not info: continue
            ax.axhline(info["test_auc"], color=color, lw=1.5, ls=ls,
                        label=f"TF-IDF {name} AUC={info['test_auc']:.3f}")

    n_test = res["n_test"]
    pos_rate = res.get("pos_rate_test", res.get("pos_rate"))
    ttl_full = f"{ttl}\n(test n={n_test}"
    if pos_rate is not None: ttl_full += f", pos≈{pos_rate:.2f}"
    ttl_full += ")"
    ax.set_title(ttl_full)
    ax.set_xlabel("Layer index")
    ax.set_xticks(LAYERS)
    ax.axhline(0.5, color="black", lw=0.5, alpha=0.4)
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

ax_g.set_ylabel("Test AUC")
fig.suptitle("Refusal probes by layer — Gemma 4-31B-it vs Qwen 3.6-27B "
             "(same 878 prompts, different models)",
             fontsize=12, y=1.01)
fig.tight_layout()
out = Path(__file__).parent / "qwen_gemma_refusal_auc_by_layer.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved {out}")
