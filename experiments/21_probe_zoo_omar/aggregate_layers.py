"""Refusal mean/max/last-of-layers — single-vector probes (5376d each).

Three pooling-across-layers operators on the (N, 13, d) cached features:
  - mean of 13 layers   → single 5376-d vector
  - max  of 13 layers   → single 5376-d vector (per-coordinate max across layer axis)
  - last layer (L60)    → reference baseline single 5376-d vector
  - softmax-weighted mean (learned per-layer logits, simple linear head over per-layer projections)

For each, run 5-fold LR with C-sweep. Compare to single-best-layer L40 mean (0.9445).

Output: refusal_aggregate_layers.json + plot.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
CACHE = REPO_ROOT / "experiments" / "17_quadratic_probe_omar" / "cache" / "refusal_13layer_mean.npz"
OUT = HERE / "results" / "refusal_aggregate_layers.json"

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
N_FOLDS = 5
SEED = 0
C_GRID = [0.01, 0.1, 1.0, 10.0]


def cv_lr(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs, fold_train_aucs, fold_cs = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + fold)
        itr, iva = next(iter(inner.split(X[tr], y[tr])))
        best_c, best_auc = C_GRID[0], -1.0
        for c in C_GRID:
            clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs").fit(X[tr][itr], y[tr][itr])
            a = roc_auc_score(y[tr][iva], clf.predict_proba(X[tr][iva])[:, 1])
            if a > best_auc: best_auc, best_c = a, c
        clf = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs").fit(X[tr], y[tr])
        fold_aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        fold_train_aucs.append(roc_auc_score(y[tr], clf.predict_proba(X[tr])[:, 1]))
        fold_cs.append(best_c)
    return {"auc_mean": float(np.mean(fold_aucs)),
            "auc_std": float(np.std(fold_aucs, ddof=1)),
            "fold_aucs": fold_aucs,
            "train_auc_mean": float(np.mean(fold_train_aucs)),
            "fold_cs": fold_cs}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    assert CACHE.exists(), f"Missing {CACHE}"
    z = np.load(CACHE, allow_pickle=True)
    X_all = z["X"].astype(np.float32)  # (N, 13, d)
    y = z["y"].astype(np.int64)
    n, n_layers, d = X_all.shape

    out = {"task": "refusal_gemma", "n_samples": int(n), "d_model": int(d),
           "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
           "C_grid": C_GRID, "n_folds": N_FOLDS, "seed": SEED, "by_pool": {}}

    print("[refusal aggregate-layers]")

    pools = {
        "mean_of_layers":  X_all.mean(axis=1),
        "max_of_layers":   X_all.max(axis=1),
        "min_of_layers":   X_all.min(axis=1),
        "last_layer_L60":  X_all[:, -1, :],
        "first_layer_L0":  X_all[:, 0,  :],
        # mean-of-mid: L25..L45 — empirically the best single-layer band
        "mean_mid_L25_L45": X_all[:, 5:10, :].mean(axis=1),
        # max-of-mid: same band, max
        "max_mid_L25_L45":  X_all[:, 5:10, :].max(axis=1),
    }

    for name, X in pools.items():
        t0 = time.time()
        r = cv_lr(X, y)
        out["by_pool"][name] = {**r, "elapsed_s": round(time.time() - t0, 1)}
        print(f"  {name:>22}: AUC={r['auc_mean']:.4f} ± {r['auc_std']:.4f}  "
              f"train={r['train_auc_mean']:.3f}  ({out['by_pool'][name]['elapsed_s']:.0f}s)",
              flush=True)
        OUT.write_text(json.dumps(out, indent=2))

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
