"""AUC + Accuracy vs layer plots for refusal_gemma honest test split.

Re-fits LR per (layer, pooling) on train split, computes AUC + Accuracy
on test split + bootstrap 95% CI. Plots both metrics with shaded CI.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

REFUSAL_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b")
REFUSAL_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def bootstrap_ci(scores, labels, n_boot=500, alpha=0.05):
    """Bootstrap CIs for AUC + accuracy on (scores, labels)."""
    rng = np.random.default_rng(0)
    n = len(scores)
    aucs, accs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            aucs.append(roc_auc_score(labels[idx], scores[idx]))
            accs.append(accuracy_score(labels[idx], (scores[idx] > 0.5).astype(int)))
        except Exception:
            continue
    aucs = np.asarray(aucs); accs = np.asarray(accs)
    a_lo = np.percentile(aucs, 100 * alpha / 2)
    a_hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    c_lo = np.percentile(accs, 100 * alpha / 2)
    c_hi = np.percentile(accs, 100 * (1 - alpha / 2))
    return float(a_lo), float(a_hi), float(c_lo), float(c_hi)


def load_split(split):
    rows = []
    with REFUSAL_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != split: continue
            if r.get("is_refusal") is None: continue
            rows.append((r["sample_id"], int(bool(r["is_refusal"]))))
    means_per_layer, lasts_per_layer, ys = [], [], []
    for sid, lbl in rows:
        p = REFUSAL_EXTRACTS / f"{sid}.pt"
        if not p.exists(): continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
        residuals = ex["residuals"].float()
        mask = ex["attention_mask"].bool().squeeze()
        n = int(mask.sum().item())
        if n < 2: continue
        last_idx = int(mask.nonzero().max().item())
        m = mask.float().unsqueeze(0).unsqueeze(-1)
        feat_mean = ((residuals * m).sum(dim=1) / n).numpy().astype(np.float32)
        feat_last = residuals[:, last_idx, :].numpy().astype(np.float32)
        if not np.isfinite(feat_mean).all() or not np.isfinite(feat_last).all(): continue
        means_per_layer.append(feat_mean); lasts_per_layer.append(feat_last); ys.append(lbl)
    return np.stack(means_per_layer), np.stack(lasts_per_layer), np.asarray(ys, np.int64)


def main():
    print("[AUC+ACC vs layer for refusal_gemma]", flush=True)
    print("loading train+test...", flush=True)
    t0 = time.time()
    Xm_tr, Xl_tr, y_tr = load_split("train")
    Xm_te, Xl_te, y_te = load_split("test")
    print(f"  train {Xm_tr.shape} test {Xm_te.shape} ({time.time()-t0:.0f}s)", flush=True)

    auc_mean, acc_mean = [], []
    auc_mean_lo, auc_mean_hi = [], []
    acc_mean_lo, acc_mean_hi = [], []
    auc_last, acc_last = [], []
    auc_last_lo, auc_last_hi = [], []
    acc_last_lo, acc_last_hi = [], []

    print("\nfitting LR per layer (mean + last)...", flush=True)
    for li, L in enumerate(LAYERS):
        # mean pool
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xm_tr[:, li, :], y_tr)
        s = clf.predict_proba(Xm_te[:, li, :])[:, 1]
        a = roc_auc_score(y_te, s); ac = accuracy_score(y_te, (s > 0.5).astype(int))
        a_lo, a_hi, c_lo, c_hi = bootstrap_ci(s, y_te)
        auc_mean.append(a); acc_mean.append(ac)
        auc_mean_lo.append(a_lo); auc_mean_hi.append(a_hi)
        acc_mean_lo.append(c_lo); acc_mean_hi.append(c_hi)
        # last token (skip L0)
        if L == 0:
            auc_last.append(np.nan); acc_last.append(np.nan)
            auc_last_lo.append(np.nan); auc_last_hi.append(np.nan)
            acc_last_lo.append(np.nan); acc_last_hi.append(np.nan)
        else:
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xl_tr[:, li, :], y_tr)
            s = clf.predict_proba(Xl_te[:, li, :])[:, 1]
            a = roc_auc_score(y_te, s); ac = accuracy_score(y_te, (s > 0.5).astype(int))
            a_lo, a_hi, c_lo, c_hi = bootstrap_ci(s, y_te)
            auc_last.append(a); acc_last.append(ac)
            auc_last_lo.append(a_lo); auc_last_hi.append(a_hi)
            acc_last_lo.append(c_lo); acc_last_hi.append(c_hi)
        print(f"  L{L:>3}  mean: AUC={auc_mean[-1]:.4f} CI=[{auc_mean_lo[-1]:.3f},{auc_mean_hi[-1]:.3f}] "
              f"ACC={acc_mean[-1]:.4f}  |  last: AUC={auc_last[-1]:.4f}", flush=True)

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.grid": True,
                          "grid.alpha": 0.25, "axes.spines.top": False,
                          "axes.spines.right": False, "savefig.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    # AUC panel
    ax = axes[0]
    ax.fill_between(LAYERS, auc_mean_lo, auc_mean_hi, color="#1f77b4", alpha=0.18)
    ax.plot(LAYERS, auc_mean, "o-", color="#1f77b4", lw=2.5, ms=7, label="mean-pool")
    ax.fill_between(LAYERS, auc_last_lo, auc_last_hi, color="#d62728", alpha=0.18)
    ax.plot(LAYERS, auc_last, "s-", color="#d62728", lw=2.5, ms=7, label="last-token")
    ax.axhline(0.50, ls="--", c="grey", alpha=0.5, label="chance")
    ax.set_xlabel("Layer"); ax.set_ylabel("Test AUC")
    ax.set_xticks(LAYERS)
    ax.set_title("Refusal-Gemma — Test AUC vs layer (LR probe)\n"
                 "shaded = 95% bootstrap CI (n_boot=500)")
    ax.set_ylim(0.45, 1.0)
    best_li = int(np.nanargmax(auc_last))
    ax.annotate(f"L{LAYERS[best_li]} best last-tok\n{auc_last[best_li]:.3f}",
                xy=(LAYERS[best_li], auc_last[best_li]),
                xytext=(LAYERS[best_li] - 12, 0.85),
                arrowprops=dict(arrowstyle="->", color="#d62728"),
                fontsize=10, ha="center", color="#d62728", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)

    # Accuracy panel
    ax = axes[1]
    ax.fill_between(LAYERS, acc_mean_lo, acc_mean_hi, color="#1f77b4", alpha=0.18)
    ax.plot(LAYERS, acc_mean, "o-", color="#1f77b4", lw=2.5, ms=7, label="mean-pool")
    ax.fill_between(LAYERS, acc_last_lo, acc_last_hi, color="#d62728", alpha=0.18)
    ax.plot(LAYERS, acc_last, "s-", color="#d62728", lw=2.5, ms=7, label="last-token")
    ax.axhline(0.50, ls="--", c="grey", alpha=0.5, label="chance")
    ax.set_xlabel("Layer"); ax.set_ylabel("Test Accuracy (τ=0.5)")
    ax.set_xticks(LAYERS)
    ax.set_title("Refusal-Gemma — Test Accuracy vs layer (LR probe)\n"
                 "shaded = 95% bootstrap CI (n_boot=500)")
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.suptitle("Refusal-Gemma: per-layer test-split metrics — train n=555, test n=277",
                  y=1.04, fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "refusal_auc_acc_vs_layer.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}", flush=True)

    # Save raw data
    data = {"layers": LAYERS, "auc_mean": auc_mean, "auc_mean_ci": list(zip(auc_mean_lo, auc_mean_hi)),
            "auc_last": auc_last, "auc_last_ci": list(zip(auc_last_lo, auc_last_hi)),
            "acc_mean": acc_mean, "acc_mean_ci": list(zip(acc_mean_lo, acc_mean_hi)),
            "acc_last": acc_last, "acc_last_ci": list(zip(acc_last_lo, acc_last_hi))}
    (FIG_DIR / "refusal_auc_acc_vs_layer.json").write_text(json.dumps(data, indent=2))
    print(f"wrote raw data to {FIG_DIR/'refusal_auc_acc_vs_layer.json'}", flush=True)


if __name__ == "__main__":
    main()
