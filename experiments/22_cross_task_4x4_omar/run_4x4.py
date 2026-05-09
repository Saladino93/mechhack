"""4×4 cross-task transfer: cyber_1, cyber_2, cyber_3, refusal_gemma.

Extends exp 14 (3×3 cyber-only) by adding refusal_gemma. For each
(train_task, test_task) pair, fits LR on the train_task's labeled features at
LAYER (default 35 — both cyber_3's best and a strong layer for refusal in
exp 18) and predicts on the test_task's samples at the SAME layer.

Cross-domain note: cyber_? predicts refusal means: does the cyber harm
hyperplane learned at L35 separate prompts Gemma refused from prompts Gemma
complied with? A high off-diagonal AUC says the harm representations are
shared across the cyber/refusal domains.

Output: results_4x4.json + matrix.png

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import get_label_for_task, load_dataset  # noqa: E402

CYBER_EXTRACTS = Path("/home/ubuntu/extracts/cyber_all_omar")
REFUSAL_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b")
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"

CACHE = REPO_ROOT / "experiments" / "17_quadratic_probe_omar" / "cache"
SEL_CYBER = {
    "cyber_1": REPO_ROOT / "experiments" / "03_layer_sweep_omar" / "selection.json",
    "cyber_2": REPO_ROOT / "experiments" / "06_cyber2_extract_omar" / "selection.json",
    "cyber_3": REPO_ROOT / "experiments" / "07_cyber3_extract_omar" / "selection.json",
}

# Pick the layer to use for everyone — cyber_3's best (35) is also strong on refusal.
# Could also do L40 (refusal best mean + cyber_1/2 best mean). Keep it L35 to match
# cyber_3 and to be the most "harm-tier-aware" layer.
LAYER = 35
LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def load_cyber_at_layer(task, layer):
    """Returns (X (N,d), y (N,))."""
    cache_p = CACHE / f"{task}_13layer_mean.npz"
    if cache_p.exists():
        z = np.load(cache_p, allow_pickle=True)
        Xall = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        layer_idxs = list(z["layer_idxs"])
        li = layer_idxs.index(layer)
        return Xall[:, li, :], y
    # Fallback: build it
    sel = json.loads(SEL_CYBER[task].read_text())
    selected_ids = [s["sample_id"] for s in sel["samples"]]
    samples_by_id = {s["sample_id"]: s for s in load_dataset("cyber", split="train")}
    Xs, ys = [], []
    layer_idxs = None
    for sid in selected_ids:
        s = samples_by_id.get(sid)
        if s is None: continue
        lbl = get_label_for_task(s, task)
        if lbl is None: continue
        p = CYBER_EXTRACTS / f"{sid}.pt"
        if not p.exists(): continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
        if layer_idxs is None: layer_idxs = list(ex["layer_idxs"])
        li = layer_idxs.index(layer)
        residuals = ex["residuals"][li].float()
        mask = ex["attention_mask"].bool().squeeze()
        n = int(mask.sum().item())
        if n < 2: continue
        m = mask.float().unsqueeze(-1)
        feat = ((residuals * m).sum(dim=0) / n).numpy().astype(np.float32)
        if not np.isfinite(feat).all(): continue
        Xs.append(feat); ys.append(int(lbl))
    return np.stack(Xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def load_refusal_at_layer(layer):
    cache_p = CACHE / "refusal_13layer_mean.npz"
    if cache_p.exists():
        z = np.load(cache_p, allow_pickle=True)
        Xall = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        layer_idxs = list(z["layer_idxs"])
        li = layer_idxs.index(layer)
        return Xall[:, li, :], y
    # Fallback
    attrs = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            attrs[r["sample_id"]] = r
    Xs, ys = [], []
    layer_idxs = None
    pt_paths = sorted(REFUSAL_EXTRACTS.glob("*.pt"))
    for p in pt_paths:
        sid = p.stem
        if sid not in attrs: continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
        if layer_idxs is None: layer_idxs = list(ex["layer_idxs"])
        li = layer_idxs.index(layer)
        residuals = ex["residuals"][li].float().clone()
        mask = ex["attention_mask"].bool().squeeze().clone()
        n = int(mask.sum().item())
        if n < 2: continue
        m = mask.float().unsqueeze(-1)
        feat = ((residuals * m).sum(dim=0) / n).numpy().astype(np.float32)
        if not np.isfinite(feat).all(): continue
        Xs.append(feat); ys.append(int(ex["label"]))
    return np.stack(Xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def main():
    print(f"[4×4 cross-task] layer={LAYER}", flush=True)
    tasks = ["cyber_1", "cyber_2", "cyber_3", "refusal_gemma"]
    data = {}
    t0 = time.time()
    for t in tasks:
        if t.startswith("cyber"):
            X, y = load_cyber_at_layer(t, LAYER)
        else:
            X, y = load_refusal_at_layer(LAYER)
        data[t] = (X, y)
        print(f"  loaded {t}: n={len(y)} pos={int((y==1).sum())} neg={int((y==0).sum())} "
              f"({time.time()-t0:.0f}s)", flush=True)

    matrix = {tr: {} for tr in tasks}
    n_table = {tr: {} for tr in tasks}
    in_task_aucs = {}

    for tr in tasks:
        X_tr_all, y_tr_all = data[tr]
        # In-task: 5-fold CV AUC for diagonal, instead of training-fit (more honest).
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        in_aucs = []
        for fold, (a, b) in enumerate(skf.split(X_tr_all, y_tr_all)):
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(X_tr_all[a], y_tr_all[a])
            in_aucs.append(roc_auc_score(y_tr_all[b], clf.predict_proba(X_tr_all[b])[:, 1]))
        in_task_aucs[tr] = float(np.mean(in_aucs))

        # Cross: fit on FULL train and predict on each test task's full data
        clf_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(X_tr_all, y_tr_all)
        for te in tasks:
            if tr == te:
                matrix[tr][te] = in_task_aucs[tr]
                n_table[tr][te] = int(len(y_tr_all))
                continue
            X_te, y_te = data[te]
            if len(set(y_te.tolist())) < 2:
                matrix[tr][te] = float("nan"); n_table[tr][te] = int(len(y_te)); continue
            p = clf_full.predict_proba(X_te)[:, 1]
            auc = float(roc_auc_score(y_te, p))
            matrix[tr][te] = auc
            n_table[tr][te] = int(len(y_te))
        print(f"\n  train {tr}:", flush=True)
        for te in tasks:
            label = "  (CV)" if tr == te else ""
            print(f"    → {te:>16} AUC={matrix[tr][te]:.4f}  n={n_table[tr][te]}{label}",
                  flush=True)

    out = {
        "tasks": tasks,
        "layer": LAYER,
        "matrix_auc": matrix,
        "n_samples": n_table,
        "in_task_cv_auc": in_task_aucs,
        "wall_seconds": round(time.time() - t0, 1),
    }
    out_path = HERE / "results_4x4.json"
    out_path.write_text(json.dumps(out, indent=2))

    # Plot
    M = np.array([[matrix[tr][te] for te in tasks] for tr in tasks])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            v = M[i, j]
            txt = f"{v:.3f}" + ("\n(CV)" if i == j else "")
            ax.text(j, i, txt, ha="center", va="center",
                    color="black" if 0.6 < v < 0.95 else "white", fontsize=10)
    ax.set_xticks(range(len(tasks))); ax.set_yticks(range(len(tasks)))
    ax.set_xticklabels([f"test\n{t}" for t in tasks])
    ax.set_yticklabels([f"train {t}" for t in tasks])
    ax.set_title(f"4×4 cross-task probe transfer (mean-pool L{LAYER})")
    fig.colorbar(im, ax=ax, label="Test AUC")
    plt.tight_layout()
    fig.savefig(HERE / "matrix_4x4.png", dpi=140)
    print(f"\nwrote {out_path} and {HERE/'matrix_4x4.png'}", flush=True)


if __name__ == "__main__":
    main()
