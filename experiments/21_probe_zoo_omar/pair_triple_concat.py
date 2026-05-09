"""Refusal layer-pair / layer-triple concat probes (LR with C-sweep).

Loads cached (N, 13, d) refusal-mean features (from
experiments/17_quadratic_probe_omar/cache/refusal_13layer_mean.npz). For each
*consecutive* pair (L25+L30, L30+L35, …) and *consecutive* triple (L30+L35+L40,
L35+L40+L45, …), concat into d/2d/3d features and run LR with C-sweep over
{0.01, 0.1, 1, 10}.

Compares against:
  - single-layer best (exp 18 says L40 mean 0.9445; L45 last-tok 0.9528)
  - all-13-layer concat (exp 18 says 0.9285 — overfits)

Output: pair_triple_concat.json + auc plot.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
CACHE = REPO_ROOT / "experiments" / "17_quadratic_probe_omar" / "cache" / "refusal_13layer_mean.npz"
OUT = HERE / "results" / "refusal_pair_triple_concat.json"
PNG = HERE / "results" / "refusal_pair_triple_concat.png"

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
N_FOLDS = 5
SEED = 0
C_GRID = [0.01, 0.1, 1.0, 10.0]


def cv_lr(X, y, c_grid=C_GRID):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs, fold_train_aucs, fold_cs, fold_l1s = [], [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        # Inner pick C on a 4:1 split
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + fold)
        itr, iva = next(iter(inner.split(X[tr], y[tr])))
        best_c, best_auc = c_grid[0], -1.0
        for c in c_grid:
            clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs").fit(X[tr][itr], y[tr][itr])
            p = clf.predict_proba(X[tr][iva])[:, 1]
            a = roc_auc_score(y[tr][iva], p)
            if a > best_auc: best_auc, best_c = a, c
        clf = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs").fit(X[tr], y[tr])
        p_te = clf.predict_proba(X[te])[:, 1]
        p_tr = clf.predict_proba(X[tr])[:, 1]
        fold_aucs.append(roc_auc_score(y[te], p_te))
        fold_train_aucs.append(roc_auc_score(y[tr], p_tr))
        fold_cs.append(best_c)
        fold_l1s.append(float(np.abs(clf.coef_).sum()))
    return {
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs, ddof=1)),
        "fold_aucs": fold_aucs,
        "train_auc_mean": float(np.mean(fold_train_aucs)),
        "fold_cs": fold_cs,
        "median_l1_norm": float(np.median(fold_l1s)),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    assert CACHE.exists(), f"Missing cache {CACHE} — run sweep_layers.py first to populate it"
    z = np.load(CACHE, allow_pickle=True)
    X_all = z["X"].astype(np.float32)  # (N, 13, d)
    y = z["y"].astype(np.int64)
    layer_idxs = list(z["layer_idxs"])
    n, n_layers, d = X_all.shape
    assert layer_idxs == LAYERS

    out = {"task": "refusal_gemma", "n_samples": int(n), "d_model": int(d),
           "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
           "C_grid": C_GRID, "n_folds": N_FOLDS, "seed": SEED,
           "single_layer": {}, "pair_concat": {}, "triple_concat": {}}

    print("[refusal pair/triple concat]\n", flush=True)

    print("=== single layer (mean) ===", flush=True)
    for li, lyr in enumerate(LAYERS):
        X = X_all[:, li, :]
        r = cv_lr(X, y)
        out["single_layer"][str(lyr)] = r
        print(f"  L{lyr:>3}: AUC={r['auc_mean']:.4f} ± {r['auc_std']:.4f}  "
              f"train={r['train_auc_mean']:.3f}  med|w|_1={r['median_l1_norm']:.0f}",
              flush=True)
        OUT.write_text(json.dumps(out, indent=2))  # incremental save

    print("\n=== consecutive pair concat ===", flush=True)
    for i in range(n_layers - 1):
        l1, l2 = LAYERS[i], LAYERS[i + 1]
        X = np.concatenate([X_all[:, i, :], X_all[:, i + 1, :]], axis=1)
        r = cv_lr(X, y)
        out["pair_concat"][f"L{l1}+L{l2}"] = r
        print(f"  L{l1}+L{l2}: AUC={r['auc_mean']:.4f} ± {r['auc_std']:.4f}  "
              f"train={r['train_auc_mean']:.3f}", flush=True)
        OUT.write_text(json.dumps(out, indent=2))

    print("\n=== consecutive triple concat ===", flush=True)
    for i in range(n_layers - 2):
        l1, l2, l3 = LAYERS[i], LAYERS[i + 1], LAYERS[i + 2]
        X = np.concatenate([X_all[:, i, :], X_all[:, i + 1, :], X_all[:, i + 2, :]], axis=1)
        r = cv_lr(X, y)
        out["triple_concat"][f"L{l1}+L{l2}+L{l3}"] = r
        print(f"  L{l1}+L{l2}+L{l3}: AUC={r['auc_mean']:.4f} ± {r['auc_std']:.4f}  "
              f"train={r['train_auc_mean']:.3f}", flush=True)
        OUT.write_text(json.dumps(out, indent=2))

    OUT.write_text(json.dumps(out, indent=2))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sx = LAYERS
    sy = [out["single_layer"][str(l)]["auc_mean"] for l in sx]
    se = [out["single_layer"][str(l)]["auc_std"] for l in sx]
    ax.errorbar(sx, sy, yerr=se, marker="o", label="single (5376d)", capsize=3)
    px = [(LAYERS[i] + LAYERS[i+1]) / 2 for i in range(len(LAYERS) - 1)]
    py = [out["pair_concat"][f"L{LAYERS[i]}+L{LAYERS[i+1]}"]["auc_mean"] for i in range(len(LAYERS) - 1)]
    pe = [out["pair_concat"][f"L{LAYERS[i]}+L{LAYERS[i+1]}"]["auc_std"] for i in range(len(LAYERS) - 1)]
    ax.errorbar(px, py, yerr=pe, marker="s", label="pair (10752d)", capsize=3)
    tx = [(LAYERS[i] + LAYERS[i+2]) / 2 for i in range(len(LAYERS) - 2)]
    ty = [out["triple_concat"][f"L{LAYERS[i]}+L{LAYERS[i+1]}+L{LAYERS[i+2]}"]["auc_mean"] for i in range(len(LAYERS) - 2)]
    te = [out["triple_concat"][f"L{LAYERS[i]}+L{LAYERS[i+1]}+L{LAYERS[i+2]}"]["auc_std"] for i in range(len(LAYERS) - 2)]
    ax.errorbar(tx, ty, yerr=te, marker="^", label="triple (16128d)", capsize=3)
    ax.axhline(0.9528, ls="--", c="grey", label="exp18 best last-tok L45 = 0.9528")
    ax.axhline(0.9285, ls=":", c="grey", label="exp18 13-layer concat = 0.9285")
    ax.set_xlabel("layer (centre for pair/triple)")
    ax.set_ylabel("AUC (5-fold CV mean ± std)")
    ax.set_title(f"Refusal: layer-concat probe family (n={n})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.80, 0.97)
    plt.tight_layout()
    fig.savefig(PNG, dpi=120)
    print(f"\nwrote {OUT} and {PNG}", flush=True)


if __name__ == "__main__":
    main()
