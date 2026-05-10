"""Optimized linear probe via sklearn LogisticRegression + C-sweep.

Replaces the AdamW-based linear probe with a properly-tuned LBFGS L2 fit.
This is the right baseline for "is the linear feature any good?" — any AUC
gap between this and the AdamW probe is a *training* artifact, not a
feature-quality problem.

Reads the cached `pool_cache.npz` produced by `train_probes.py`. Per layer,
fits separately for mean-pool and last-token features. Saves the
seed-averaged-equivalent (sklearn is deterministic) test predictions for
length-stratified analysis.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

OUT_DIR = Path(__file__).parent
LAYER_IDXS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
SAMPLES_FILE = (OUT_DIR.parent.parent /
                "datasets/refusal_probes/qwen36/attacks_full.jsonl")


def auc_with_ci(y_true, y_score, n_boot=1000, seed=0):
    auc = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true); aucs = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, size=n)
        if len(set(y_true[ix].tolist())) < 2: continue
        aucs.append(roc_auc_score(y_true[ix], y_score[ix]))
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# Load cache
z = np.load(OUT_DIR / "pool_cache.npz")
mean_arr = z["mean"]
last_arr = z["last"]
ids = list(z["sample_ids"])
print(f"Loaded cache: {mean_arr.shape}")

# Splits + labels
samples = {json.loads(l)["sample_id"]: json.loads(l)
           for l in open(SAMPLES_FILE)}
y = np.array([int(samples[sid]["is_refusal"]) for sid in ids], dtype=np.int64)
splits = np.array([samples[sid]["split"] for sid in ids])
train_idx = np.where(splits == "train")[0]
test_idx  = np.where(splits == "test")[0]

# Inner val for C-sweep
inner_train, inner_val = train_test_split(
    train_idx, test_size=0.2,
    stratify=y[train_idx], random_state=0)


def sklearn_one_layer(X, y_train_full, y_test, train_idx, test_idx, inner_train,
                      inner_val, Cs=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)):
    inner_aucs = {}
    for C in Cs:
        lr = LogisticRegression(solver="lbfgs", max_iter=4000, C=C,
                                 class_weight="balanced")
        lr.fit(X[inner_train], y[inner_train])
        p = lr.predict_proba(X[inner_val])[:, 1]
        inner_aucs[C] = float(roc_auc_score(y[inner_val], p))
    bestC = max(Cs, key=lambda c: inner_aucs[c])
    lr = LogisticRegression(solver="lbfgs", max_iter=4000, C=bestC,
                             class_weight="balanced")
    lr.fit(X[train_idx], y[train_idx])
    proba = lr.predict_proba(X[test_idx])[:, 1]
    auc, lo, hi = auc_with_ci(y_test, proba, seed=42)
    return {"best_C": bestC, "test_auc": auc, "ci95": [lo, hi],
            "inner_val_aucs": inner_aucs,
            "test_proba": proba.tolist()}, proba


results = {}
test_preds = {"layer_idxs": LAYER_IDXS, "test_idx": test_idx.tolist(),
              "test_sample_ids": [ids[i] for i in test_idx],
              "y_test": y[test_idx].tolist(), "by_arch": {}}

for arch_name, X_arr in [("linear_sklearn_mean", mean_arr),
                            ("linear_sklearn_last", last_arr)]:
    results[arch_name] = {"per_layer": {}}
    test_preds["by_arch"][arch_name] = {}
    for L in LAYER_IDXS:
        li = LAYER_IDXS.index(L)
        X = X_arr[:, li, :]
        rec, proba = sklearn_one_layer(X, y, y[test_idx], train_idx, test_idx,
                                         inner_train, inner_val)
        results[arch_name]["per_layer"][str(L)] = {
            "test_auc": rec["test_auc"], "ci95": rec["ci95"],
            "best_C": rec["best_C"],
        }
        test_preds["by_arch"][arch_name][str(L)] = proba.tolist()
        print(f"  {arch_name:22s} L{L:02d}: AUC {rec['test_auc']:.4f} "
              f"[{rec['ci95'][0]:.4f},{rec['ci95'][1]:.4f}] C={rec['best_C']}")

# Merge into results.json
res_path = OUT_DIR / "results.json"
all_res = json.loads(res_path.read_text()) if res_path.exists() else {}
all_res.setdefault("by_arch", {}).update(results)
res_path.write_text(json.dumps(all_res, indent=2))
print(f"Saved per-arch AUC to {res_path}")

# Save predictions separately (heavy)
np.savez_compressed(OUT_DIR / "test_predictions.npz",
                     **{f"{arch}_L{L}": np.array(test_preds["by_arch"][arch][str(L)],
                                                  dtype=np.float32)
                        for arch in test_preds["by_arch"]
                        for L in LAYER_IDXS})
(OUT_DIR / "test_predictions_meta.json").write_text(json.dumps({
    "layer_idxs": LAYER_IDXS, "test_sample_ids": test_preds["test_sample_ids"],
    "y_test": test_preds["y_test"]}, indent=2))
print(f"Saved test predictions for length-stratification.")
