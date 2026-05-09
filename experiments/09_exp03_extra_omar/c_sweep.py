"""D2 — Regularisation sweep at the two best layers from exp 03.

For (mean-pool, layer 30) and (last-token, layer 35), sweep C in
{1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0}, run 5-fold CV, report:
  - test AUC (mean ± σ)
  - test acc (mean ± σ)
  - train AUC (mean) — does the probe still memorise at this regularisation?
  - L1 norm of weights (mean over folds) — sparsity proxy

Headline question: at what (smaller) C do we still match the best AUC within
0.005? If C=1e-3 already matches 0.98 with much smaller L1, the probe doesn't
need 5376 dims; if even C=1e-2 collapses, the signal is genuinely distributed.

Reuses the pre-computed extracts from exp 03; no GPU needed.
"""
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
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import get_label_for_task, load_dataset  # noqa: E402

EXP03_SEL = REPO_ROOT / "experiments" / "03_layer_sweep_omar" / "selection.json"
EXTRACTS_DIR = Path("/home/ubuntu/extracts/03_layer_sweep_omar")
OUT_DIR = Path(__file__).parent
RESULTS_PATH = OUT_DIR / "results.json"
METRICS_LOG = OUT_DIR / "metrics.jsonl"

TASK = "cyber_1"
SEED = 0
N_FOLDS = 5
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

# (pooling, layer) configs to sweep — exp 03's two best layers per pooling.
CONFIGS = [
    {"pooling": "mean", "layer": 30},
    {"pooling": "last_token", "layer": 35},
]


def append_jsonl(path: Path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def pool_one_layer(pt_path: Path, layer_pos: int, pooling: str):
    """Return (d,) feature vector for the chosen pooling at the chosen layer index
    (position in the stored layer_idxs array, not the absolute layer number).
    """
    ex = torch.load(str(pt_path), weights_only=False)
    res = ex["residuals"][layer_pos].float()  # (N_tok, d)
    mask = ex["attention_mask"].bool()
    if mask.sum().item() == 0:
        return None
    if pooling == "mean":
        m_f = mask.float().unsqueeze(-1)
        feat = (res * m_f).sum(dim=0) / mask.sum().item()
    elif pooling == "last_token":
        last_idx = int(mask.nonzero().max().item())
        feat = res[last_idx]
    else:
        raise ValueError(pooling)
    return feat.numpy().astype(np.float32)


def load_features_for_layer(sample_ids, layer_idx, pooling):
    """Load + pool features for the given layer (absolute index 0..60).

    Reads the .pt's `layer_idxs` to find the right position. Drops missing /
    empty-mask samples (returns labels aligned with kept samples).
    """
    layer_pos = None
    feats, kept = [], []
    for sid in sample_ids:
        p = EXTRACTS_DIR / f"{sid}.pt"
        if not p.exists():
            continue
        if layer_pos is None:
            ex = torch.load(str(p), weights_only=False)
            layer_idxs = list(ex["layer_idxs"])
            layer_pos = layer_idxs.index(layer_idx)
        f = pool_one_layer(p, layer_pos, pooling)
        if f is None:
            continue
        feats.append(f)
        kept.append(sid)
    return np.stack(feats, axis=0), kept, layer_pos


def main():
    print("[D2] Regularisation sweep at best exp 03 layers", flush=True)

    sel = json.loads(EXP03_SEL.read_text())
    sel_ids = [row["sample_id"] for row in sel["samples"]]
    rows = {s["sample_id"]: s for s in load_dataset("cyber", split="train")}

    # Pre-compute the y vector once, by sample_id.
    label_by_id = {}
    for sid in sel_ids:
        s = rows.get(sid)
        if s is None:
            continue
        lbl = get_label_for_task(s, TASK)
        if lbl is not None:
            label_by_id[sid] = lbl

    sweep_results = {}

    for cfg in CONFIGS:
        pooling, layer = cfg["pooling"], cfg["layer"]
        cfg_key = f"{pooling}_L{layer}"
        print(f"\n  === {cfg_key} ===", flush=True)
        t0 = time.time()
        X, kept, layer_pos = load_features_for_layer(sel_ids, layer, pooling)
        y = np.asarray([label_by_id[sid] for sid in kept], dtype=np.int64)
        print(f"  loaded {X.shape[0]} feats, d={X.shape[1]} (layer_pos={layer_pos}) "
              f"in {time.time()-t0:.1f}s", flush=True)

        per_C = []
        for C in C_GRID:
            tC = time.time()
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            fold_aucs, fold_accs, fold_train_aucs, fold_l1, fold_active = [], [], [], [], []
            for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
                clf.fit(X[tr_idx], y[tr_idx])
                p_te = clf.predict_proba(X[te_idx])[:, 1]
                p_tr = clf.predict_proba(X[tr_idx])[:, 1]
                auc = float(roc_auc_score(y[te_idx], p_te))
                acc = float(((p_te > 0.5).astype(int) == y[te_idx]).mean())
                auc_tr = float(roc_auc_score(y[tr_idx], p_tr))
                w = clf.coef_[0]
                l1 = float(np.abs(w).sum())
                active = int((np.abs(w) > 1e-3).sum())
                fold_aucs.append(auc); fold_accs.append(acc)
                fold_train_aucs.append(auc_tr); fold_l1.append(l1); fold_active.append(active)
            row = {
                "pooling": pooling,
                "layer": int(layer),
                "C": C,
                "auc_mean": float(np.mean(fold_aucs)),
                "auc_std": float(np.std(fold_aucs, ddof=1)),
                "acc_mean": float(np.mean(fold_accs)),
                "acc_std": float(np.std(fold_accs, ddof=1)),
                "train_auc_mean": float(np.mean(fold_train_aucs)),
                "weight_l1_mean": float(np.mean(fold_l1)),
                "weight_l1_std": float(np.std(fold_l1, ddof=1)),
                "n_active_mean": float(np.mean(fold_active)),
                "elapsed_s": round(time.time() - tC, 2),
            }
            per_C.append(row)
            append_jsonl(METRICS_LOG, {"diagnostic": "D2_c_sweep", **row})
            print(f"    C={C:>7g}: test AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f} "
                  f"acc={row['acc_mean']:.3f} | train AUC={row['train_auc_mean']:.4f} | "
                  f"L1={row['weight_l1_mean']:.1f} | "
                  f"|w|>1e-3: {row['n_active_mean']:.0f}/{X.shape[1]}", flush=True)

        # Sparsest C within 0.005 of best.
        best_auc = max(r["auc_mean"] for r in per_C)
        within = [r for r in per_C if r["auc_mean"] >= best_auc - 0.005]
        sparsest = min(within, key=lambda r: r["C"])
        sweep_results[cfg_key] = {
            "pooling": pooling, "layer": layer,
            "n_samples": int(X.shape[0]), "d_model": int(X.shape[1]),
            "C_grid": C_GRID,
            "per_C": per_C,
            "best_auc": best_auc,
            "best_C": next(r["C"] for r in per_C if r["auc_mean"] == best_auc),
            "sparsest_within_0.005": {
                "C": sparsest["C"],
                "auc_mean": sparsest["auc_mean"],
                "weight_l1_mean": sparsest["weight_l1_mean"],
                "n_active_mean": sparsest["n_active_mean"],
            },
        }
        print(f"  best AUC = {best_auc:.4f} at C={sweep_results[cfg_key]['best_C']:g}", flush=True)
        print(f"  sparsest within 0.005: C={sparsest['C']:g} "
              f"AUC={sparsest['auc_mean']:.4f} L1={sparsest['weight_l1_mean']:.1f}", flush=True)

    results = load_results()
    results["D2_c_sweep"] = sweep_results
    atomic_write_json(RESULTS_PATH, results)
    print(f"\n  written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
