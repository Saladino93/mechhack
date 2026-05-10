"""Gemma cross-task probe matrix:
   train on {cyber_1, cyber_2, cyber_3, refusal_gemma} × test on the same 4.

Reproduces exp 22 on this pod's fresh Gemma extracts. Best-layer LR per
task, with diagonal entries via stratified inner-val C-sweep, off-diagonal
entries via direct projection (no refit).

Dependencies:
   - /home/ubuntu/extracts/gemma_cyber/    (cyber_train_1500 + cyber_test)
   - /home/ubuntu/extracts/gemma_refusal/  (878 refusal)

Outputs:
   gemma_cross_task_results.json
   gemma_cross_task_matrix.png
"""
import json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent
LAYER_IDXS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
TARGET_LAYER = 35   # exp 22 used L35 mean-pool; we'll keep that for direct comparison

REPO = OUT_DIR.parent.parent
CYBER_EXTRACTS = Path("/home/ubuntu/extracts/gemma_cyber")
REFUSAL_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal")
CYBER_TRAIN = REPO / "datasets/cyber_probes/train_1500_balanced.jsonl"
CYBER_TEST  = REPO / "datasets/cyber_probes/test.jsonl"
REFUSAL     = REPO / "datasets/refusal_probes/gemma4_31b/attacks_full.jsonl"


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def cache_pool(extracts_dir, samples, key_label, layer_idx_in_extract,
                cache_path, prompt_id_field="sample_id"):
    """Build mean-pool feature for one selected layer. Returns (X, y, ids)."""
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return z["X"], z["y"], list(z["ids"])
    print(f"  caching {len(samples)} samples → {cache_path.name} (L{LAYER_IDXS[layer_idx_in_extract]})")
    X, y, ids = [], [], []
    skipped = 0
    t0 = time.time()
    for s in samples:
        sid = s[prompt_id_field]
        p = extracts_dir / f"{sid}.pt"
        if not p.exists():
            skipped += 1; continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu")
        r = ex["residuals"][layer_idx_in_extract].float()
        m = ex["attention_mask"].bool()
        if not m.any(): skipped += 1; continue
        m_f = m.float().unsqueeze(-1)
        mean = (r * m_f).sum(0) / m_f.sum(0).clamp(min=1)
        X.append(mean.numpy().astype(np.float32))
        # label_fn evaluates the row -> int 0/1 label
        ids.append(sid)
        y.append(s["_label"])
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    np.savez_compressed(cache_path, X=X, y=y, ids=np.array(ids))
    print(f"    {X.shape}  pos_rate={y.mean():.3f}  skipped={skipped}  in {time.time()-t0:.1f}s")
    return X, y, ids


def label_for_task(sample, task):
    """Return 0/1 label for the task, or None if sample isn't applicable."""
    if task == "refusal":
        v = sample.get("is_refusal")
        return None if v is None else int(bool(v))
    pos = {"cyber_1": "dual_use", "cyber_2": "high_risk_dual_use",
           "cyber_3": "prohibited"}[task]
    neg = {"cyber_1": {"benign"},
           "cyber_2": {"dual_use", "benign"},
           "cyber_3": {"high_risk_dual_use", "dual_use", "benign"}}[task]
    cat = sample.get("category")
    if cat == pos: return 1
    if cat in neg: return 0
    return None


# ---- Build per-task train/test feature blocks at TARGET_LAYER ----
li = LAYER_IDXS.index(TARGET_LAYER)

# Cyber: union of train_1500 + test = full cyber pool, then split per-task
cyber_train_samples = load_jsonl(CYBER_TRAIN)
cyber_test_samples  = load_jsonl(CYBER_TEST)
refusal_samples     = load_jsonl(REFUSAL)

print(f"=== Gemma cross-task @ L{TARGET_LAYER} ===")
print(f"cyber_train_1500: {len(cyber_train_samples)}")
print(f"cyber_test:       {len(cyber_test_samples)}")
print(f"refusal:          {len(refusal_samples)}  (with split=test/train; we use the dataset-defined splits)")

results = {"layer": TARGET_LAYER, "n_train_per_task": {}, "n_test_per_task": {},
           "matrix": {}}

# Cache per-task per-split features
task_data = {}
for task in ["cyber_1", "cyber_2", "cyber_3", "refusal"]:
    if task == "refusal":
        train_pool = [s for s in refusal_samples if s.get("split") == "train"]
        test_pool  = [s for s in refusal_samples if s.get("split") == "test"]
        train_pool = [{"_label": label_for_task(s, task), **s}
                      for s in train_pool if label_for_task(s, task) is not None]
        test_pool  = [{"_label": label_for_task(s, task), **s}
                      for s in test_pool  if label_for_task(s, task) is not None]
        ext = REFUSAL_EXTRACTS
    else:
        train_pool = [{"_label": label_for_task(s, task), **s}
                      for s in cyber_train_samples
                      if label_for_task(s, task) is not None]
        test_pool  = [{"_label": label_for_task(s, task), **s}
                      for s in cyber_test_samples
                      if label_for_task(s, task) is not None]
        ext = CYBER_EXTRACTS
    print(f"\n--- {task}: train_pool={len(train_pool)} test_pool={len(test_pool)} ---")
    Xtr, ytr, idstr = cache_pool(ext, train_pool, "_label", li,
                                   OUT_DIR / f"crosstask_{task}_train_L{TARGET_LAYER}.npz")
    Xte, yte, idste = cache_pool(ext, test_pool,  "_label", li,
                                   OUT_DIR / f"crosstask_{task}_test_L{TARGET_LAYER}.npz")
    task_data[task] = {"X_train": Xtr, "y_train": ytr,
                        "X_test":  Xte, "y_test":  yte}
    results["n_train_per_task"][task] = int(len(ytr))
    results["n_test_per_task"][task]  = int(len(yte))

# ---- Train per-task probes on train block; refit-on-full and project to all tests ----
TASKS = ["cyber_1", "cyber_2", "cyber_3", "refusal"]
matrix = np.zeros((len(TASKS), len(TASKS)), dtype=np.float64)
for i, train_t in enumerate(TASKS):
    Xtr = task_data[train_t]["X_train"]
    ytr = task_data[train_t]["y_train"]
    # inner C-sweep
    Cs = [0.001, 0.01, 0.1, 1.0, 10.0]
    Xi_tr, Xi_va, yi_tr, yi_va = train_test_split(
        Xtr, ytr, test_size=0.2, stratify=ytr, random_state=0)
    inner = {}
    for C in Cs:
        lr = LogisticRegression(solver="lbfgs", max_iter=4000, C=C,
                                 class_weight="balanced")
        lr.fit(Xi_tr, yi_tr)
        inner[C] = float(roc_auc_score(yi_va,
                            lr.predict_proba(Xi_va)[:, 1]))
    bestC = max(Cs, key=lambda c: inner[c])
    lr = LogisticRegression(solver="lbfgs", max_iter=4000, C=bestC,
                             class_weight="balanced")
    lr.fit(Xtr, ytr)
    for j, test_t in enumerate(TASKS):
        Xte = task_data[test_t]["X_test"]
        yte = task_data[test_t]["y_test"]
        proba = lr.predict_proba(Xte)[:, 1]
        try:
            auc = float(roc_auc_score(yte, proba))
        except Exception:
            auc = float("nan")
        matrix[i, j] = auc
        print(f"  train={train_t:8s} → test={test_t:8s}: AUC {auc:.4f}  "
              f"(bestC={bestC} on {train_t})")

# Save
results["matrix"] = {TASKS[i]: {TASKS[j]: matrix[i, j] for j in range(4)}
                       for i in range(4)}
(OUT_DIR / "gemma_cross_task_results.json").write_text(json.dumps(results, indent=2))
print(f"Saved {OUT_DIR / 'gemma_cross_task_results.json'}")

# Plot heatmap
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
for i in range(4):
    for j in range(4):
        v = matrix[i, j]
        col = "black" if 0.55 < v < 0.85 else "white"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                color=col, fontsize=12, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(TASKS, rotation=20)
ax.set_yticks(range(4)); ax.set_yticklabels(TASKS)
ax.set_xlabel("test task")
ax.set_ylabel("train task")
ax.set_title(f"Gemma cross-task transfer @ L{TARGET_LAYER} (mean-pool, sklearn LR)\n"
             f"Diagonal = honest test-AUC, off-diagonal = train→test transfer")
plt.colorbar(im, ax=ax, label="Test AUC")
fig.tight_layout()
out = OUT_DIR / "gemma_cross_task_matrix.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved {out}")
