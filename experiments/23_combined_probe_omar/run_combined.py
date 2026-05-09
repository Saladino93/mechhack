"""Combined-data probe: train one LR on (cyber_3 ∪ refusal) labeled
"harmful intent vs not", test held-out per task.

Hypothesis: a single hyperplane separates 'harmful prompts' from 'safe prompts'
across the cyber and refusal domains. If it works, we get a more general
probe; if it underperforms task-specific probes, the harm directions are
not perfectly aligned (consistent with cross-task transfer findings in
exp 14 / 22).

Both at L35 (best for cyber_3, strong for refusal in exp 18 too).

Procedure:
  1. Load cyber_3 features + labels at L35.
  2. Load refusal features + labels at L35.
  3. Concat → label "harmful intent" = (cyber_3==1) OR (refusal==1).
  4. 5-fold CV on combined data, plus test on each task held out:
       - train on combined - cyber_3, evaluate on cyber_3
       - train on combined - refusal,  evaluate on refusal
  5. Compare to single-task LR baselines.

Output: results_combined.json
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

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "22_cross_task_4x4_omar"))
from run_4x4 import load_cyber_at_layer, load_refusal_at_layer  # noqa: E402

LAYER = 35
SEED = 0
N_FOLDS = 5
C_GRID = [0.01, 0.1, 1.0, 10.0]


def cv_lr(X, y, c_grid=C_GRID):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs, fold_train_aucs, fold_cs = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + fold)
        itr, iva = next(iter(inner.split(X[tr], y[tr])))
        best_c, best_auc = c_grid[0], -1.0
        for c in c_grid:
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
    t0 = time.time()
    print(f"[combined-data probe] L{LAYER}", flush=True)

    X_c, y_c = load_cyber_at_layer("cyber_3", LAYER)
    X_r, y_r = load_refusal_at_layer(LAYER)
    print(f"  cyber_3:  n={len(y_c)} pos={int((y_c==1).sum())} neg={int((y_c==0).sum())}", flush=True)
    print(f"  refusal:  n={len(y_r)} pos={int((y_r==1).sum())} neg={int((y_r==0).sum())}", flush=True)

    X_combo = np.concatenate([X_c, X_r], axis=0).astype(np.float32)
    y_combo = np.concatenate([y_c, y_r], axis=0).astype(np.int64)
    src_combo = np.concatenate([np.zeros(len(y_c), dtype=int), np.ones(len(y_r), dtype=int)], axis=0)
    print(f"  combined: n={len(y_combo)} pos={int((y_combo==1).sum())} neg={int((y_combo==0).sum())}",
          flush=True)

    out = {"layer": LAYER, "n_combined": int(len(y_combo)),
           "n_cyber3": int(len(y_c)), "n_refusal": int(len(y_r)),
           "C_grid": C_GRID, "n_folds": N_FOLDS, "seed": SEED}

    print("\n  -- Single-task baselines (5-fold CV with C-sweep) --", flush=True)
    out["single_task"] = {
        "cyber_3": cv_lr(X_c, y_c),
        "refusal": cv_lr(X_r, y_r),
    }
    print(f"    cyber_3 only: AUC={out['single_task']['cyber_3']['auc_mean']:.4f} "
          f"± {out['single_task']['cyber_3']['auc_std']:.4f}", flush=True)
    print(f"    refusal only: AUC={out['single_task']['refusal']['auc_mean']:.4f} "
          f"± {out['single_task']['refusal']['auc_std']:.4f}", flush=True)

    print("\n  -- Combined-data probe (5-fold CV with C-sweep) --", flush=True)
    out["combined"] = cv_lr(X_combo, y_combo)
    print(f"    combined CV: AUC={out['combined']['auc_mean']:.4f} ± {out['combined']['auc_std']:.4f}",
          flush=True)

    # Held-out per-task evaluation: train on combined excluding the test task's data,
    # evaluate on the test task's full data with its own labels.
    print("\n  -- Held-out per-task evaluation --", flush=True)
    out["held_out"] = {}
    # Train on cyber_3 + refusal; evaluate on cyber_3 with cyber_3 labels only
    # (should be redundant with combined CV but easier to interpret).
    # Actually: train on REFUSAL only, predict on CYBER_3 — measures direct cross-domain transfer.
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(X_r, y_r)
    auc = float(roc_auc_score(y_c, clf.predict_proba(X_c)[:, 1]))
    out["held_out"]["refusal_only -> cyber_3"] = {"auc": auc, "n": int(len(y_c))}
    print(f"    refusal_only → cyber_3: AUC={auc:.4f}  (cross-domain)", flush=True)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(X_c, y_c)
    auc = float(roc_auc_score(y_r, clf.predict_proba(X_r)[:, 1]))
    out["held_out"]["cyber_3_only -> refusal"] = {"auc": auc, "n": int(len(y_r))}
    print(f"    cyber_3_only → refusal: AUC={auc:.4f}  (cross-domain)", flush=True)

    # Combined train, but per-task CV testing: 5-fold split per task, train fold = combined - test fold
    print("\n  -- Combined-train, per-task held-out CV --", flush=True)
    for src_label, name, X_t, y_t in [(0, "cyber_3", X_c, y_c), (1, "refusal", X_r, y_r)]:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_aucs = []
        for fold, (tr_t, te_t) in enumerate(skf.split(X_t, y_t)):
            # Train on the OTHER task fully + this task's tr_t fold
            other = [i for i in range(len(y_combo)) if src_combo[i] != src_label]
            this_in = [i for i in range(len(y_combo)) if src_combo[i] == src_label and (
                       (i - (0 if src_label == 0 else len(y_c))) in set(tr_t.tolist()))]
            tr = np.array(other + this_in, dtype=int)
            X_tr = X_combo[tr]; y_tr = y_combo[tr]
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(X_tr, y_tr)
            p_te = clf.predict_proba(X_t[te_t])[:, 1]
            fold_aucs.append(roc_auc_score(y_t[te_t], p_te))
        out["held_out"][f"combined → {name}"] = {
            "auc_mean": float(np.mean(fold_aucs)),
            "auc_std": float(np.std(fold_aucs, ddof=1)),
            "fold_aucs": fold_aucs,
        }
        print(f"    combined → {name}: AUC={out['held_out'][f'combined → {name}']['auc_mean']:.4f} "
              f"± {out['held_out'][f'combined → {name}']['auc_std']:.4f}", flush=True)

    out["wall_seconds"] = round(time.time() - t0, 1)
    out_path = HERE / "results_combined.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
