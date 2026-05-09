"""Refusal LR probe — per-layer sweep + multi-layer concat.

Two analyses on the new 13-layer Gemma refusal extracts at
`/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b/`:

  1. PER-LAYER: train LR mean-pool + last-token probes at each of 13 layers
     (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60). 5-fold CV. Find
     true best layer (was constrained to 32 in exp 11).

  2. MULTI-LAYER CONCAT: concat all 13 mean-pool features → 13 × 5,376 =
     69,888-dim feature → LR with C-sweep over {0.01, 0.1, 1, 10}. Mirrors
     exp 05 Head A but for refusal.

Outputs:
  - per_layer.json   : AUC mean ± std per (layer, pooling)
  - concat.json      : best C, AUC, train AUC, |coef| stats
  - auc_vs_layer.png : refusal LR per-layer with the multi-layer concat as
                        a horizontal reference line

CPU only.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b")
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
N_FOLDS = 5
SEED = 0
C_GRID = [0.01, 0.1, 1.0, 10.0]


def load_pooled():
    """Load + mean-pool / last-token pool every refusal extract.

    Returns:
        X_mean: (N, n_layers, d_model)
        X_last: (N, n_layers, d_model)
        y:      (N,)
        layer_idxs: list[int]
    """
    attrs = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            attrs[r["sample_id"]] = r

    means, lasts, labels, ids = [], [], [], []
    layer_idxs = None
    skipped = 0
    pt_paths = sorted(EXTRACTS.glob("*.pt"))
    print(f"loading {len(pt_paths)} extracts...", flush=True)
    t0 = time.time()
    for i, p in enumerate(pt_paths):
        sid = p.stem
        if sid not in attrs:
            skipped += 1; continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
        if layer_idxs is None:
            layer_idxs = list(ex["layer_idxs"])
        residuals = ex["residuals"]  # (n_layers, n_tok, d) fp16
        if residuals.dim() != 3:
            skipped += 1; continue
        residuals = residuals.float().clone()
        mask = ex["attention_mask"].bool().squeeze().clone()
        n = int(mask.sum().item())
        if n < 2:
            skipped += 1; continue
        m = mask.float().unsqueeze(0).unsqueeze(-1)
        feat_mean = (residuals * m).sum(dim=1) / n  # (n_layers, d)
        last_idx = int(mask.nonzero().max().item())
        feat_last = residuals[:, last_idx, :]
        feat_mean_np = np.asarray(feat_mean.tolist(), dtype=np.float32)
        feat_last_np = np.asarray(feat_last.tolist(), dtype=np.float32)
        if not np.isfinite(feat_mean_np).all() or not np.isfinite(feat_last_np).all():
            skipped += 1; continue
        means.append(feat_mean_np)
        lasts.append(feat_last_np)
        labels.append(int(ex["label"]))
        ids.append(sid)
        if (i + 1) % 200 == 0:
            print(f"  loaded {i+1}/{len(pt_paths)} in {time.time()-t0:.1f}s", flush=True)
    X_mean = np.stack(means)  # (N, 13, d)
    X_last = np.stack(lasts)
    y = np.asarray(labels, dtype=np.int64)
    print(f"loaded {len(y)} ok, {skipped} skipped. layer_idxs={layer_idxs}", flush=True)
    return X_mean, X_last, y, layer_idxs


def per_layer_cv(X, y, layer_idxs, label):
    """5-fold CV at each layer; X is (N, n_layers, d)."""
    out = {}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for li, layer in enumerate(layer_idxs):
        Xl = X[:, li, :]
        aucs = []
        for fold, (tr, te) in enumerate(skf.split(Xl, y)):
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            clf.fit(Xl[tr], y[tr])
            p = clf.predict_proba(Xl[te])[:, 1]
            aucs.append(float(roc_auc_score(y[te], p)))
        out[layer] = {
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "fold_aucs": aucs,
        }
        print(f"  {label} L{layer:>2}: AUC={out[layer]['auc_mean']:.4f} ± {out[layer]['auc_std']:.4f}", flush=True)
    return out


def concat_probe(X_mean, y):
    """Concat all 13 layers' mean-pool features → C-sweep with inner val."""
    N, L, d = X_mean.shape
    X = X_mean.reshape(N, L * d)  # (N, 13*5376)
    print(f"concat features: {X.shape}", flush=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    chosen_Cs = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        # Inner val for C selection: 80/20 of the train fold
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + fold)
        inner_aucs = {C: [] for C in C_GRID}
        for itr, ival in inner.split(X[tr], y[tr]):
            for C in C_GRID:
                clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
                clf.fit(X[tr][itr], y[tr][itr])
                p = clf.predict_proba(X[tr][ival])[:, 1]
                inner_aucs[C].append(float(roc_auc_score(y[tr][ival], p)))
            break  # 1 inner split is enough for a quick C pick
        best_C = max(C_GRID, key=lambda c: np.mean(inner_aucs[c]))
        chosen_Cs.append(best_C)
        # Refit on full training fold
        clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        p_te = clf.predict_proba(X[te])[:, 1]
        auc = float(roc_auc_score(y[te], p_te))
        coef_l1 = float(np.abs(clf.coef_).sum())
        fold_results.append({
            "fold": fold, "best_C": best_C,
            "auc": auc, "coef_l1": coef_l1,
            "n_train": int(len(tr)), "n_test": int(len(te)),
        })
        print(f"  fold {fold}: best_C={best_C}, AUC={auc:.4f}, |w|_1={coef_l1:.1f}", flush=True)
    aucs = [r["auc"] for r in fold_results]
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "best_C_mode": max(set(chosen_Cs), key=chosen_Cs.count),
        "best_C_per_fold": chosen_Cs,
        "fold_results": fold_results,
        "n_features": int(X.shape[1]),
    }


def main():
    print("[refusal layer sweep + concat]\n", flush=True)
    X_mean, X_last, y, layer_idxs = load_pooled()
    print()

    print("=== per-layer LR (mean-pool) ===", flush=True)
    mean_per_layer = per_layer_cv(X_mean, y, layer_idxs, "mean")
    print()
    print("=== per-layer LR (last-token) ===", flush=True)
    last_per_layer = per_layer_cv(X_last, y, layer_idxs, "last")
    print()
    print("=== multi-layer concat (mean-pool, all 13 layers) ===", flush=True)
    concat = concat_probe(X_mean, y)
    print()

    # Headlines
    best_mean = max(mean_per_layer.items(), key=lambda x: x[1]["auc_mean"])
    best_last = max(last_per_layer.items(), key=lambda x: x[1]["auc_mean"])
    print("=== HEADLINES ===")
    print(f"  Best per-layer mean-pool: L{best_mean[0]:>2} AUC={best_mean[1]['auc_mean']:.4f}")
    print(f"  Best per-layer last-tok:  L{best_last[0]:>2} AUC={best_last[1]['auc_mean']:.4f}")
    print(f"  Multi-layer concat:           AUC={concat['auc_mean']:.4f} (C={concat['best_C_mode']})")
    print(f"  Exp 11 baseline (mean L32):   AUC=0.9265")

    # Save
    out = {
        "n_samples": int(len(y)),
        "n_pos": int((y==1).sum()),
        "n_neg": int((y==0).sum()),
        "layer_idxs": layer_idxs,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "C_grid": C_GRID,
        "per_layer_mean": {str(k): v for k, v in mean_per_layer.items()},
        "per_layer_last": {str(k): v for k, v in last_per_layer.items()},
        "multi_layer_concat": concat,
        "headline": {
            "best_layer_mean": int(best_mean[0]),
            "best_auc_mean": best_mean[1]["auc_mean"],
            "best_layer_last": int(best_last[0]),
            "best_auc_last": best_last[1]["auc_mean"],
            "concat_auc": concat["auc_mean"],
            "exp11_baseline_mean_L32": 0.9265,
        },
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT/'results.json'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        xs = layer_idxs
        for label, src, color, marker in [
            ("mean-pool", mean_per_layer, "tab:blue", "o"),
            ("last-token", last_per_layer, "tab:orange", "s"),
        ]:
            ys = [src[L]["auc_mean"] for L in xs]
            es = [src[L]["auc_std"] for L in xs]
            ax.errorbar(xs, ys, yerr=es, marker=marker, color=color, capsize=3, label=label)
        ax.axhline(concat["auc_mean"], ls="--", color="tab:purple",
                    label=f"multi-layer concat (C={concat['best_C_mode']}): {concat['auc_mean']:.4f}")
        ax.axhline(0.9265, ls=":", color="grey",
                    label="exp 11 baseline (mean L32): 0.9265")
        ax.set_xlabel("Residual-stream layer index"); ax.set_ylabel("ROC-AUC (5-fold CV)")
        ax.set_title("Refusal probe — per-layer LR + multi-layer concat\n"
                     "(Gemma 4-31B-it, 832 refusal samples)")
        ax.set_xticks(xs); ax.set_ylim(0.5, 1.0)
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "auc_vs_layer.png", dpi=150)
        plt.close(fig)
        print(f"wrote {OUT/'auc_vs_layer.png'}")
    except Exception as e:
        print(f"  [warn] plot failed: {e}")


if __name__ == "__main__":
    main()
