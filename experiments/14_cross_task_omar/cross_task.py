"""Cross-task generalization: does the cyber_1 probe transfer to cyber_2/cyber_3, and v.v.?

For every (train_task, test_task) ∈ {cyber_1, cyber_2, cyber_3}²:
  - Load mean-pooled features at the train_task's best layer (cyber_1: L40,
    cyber_2: L40, cyber_3: L35) from the shared cyber_all_omar/ extracts.
  - Use the train_task's selection.json to fit LogisticRegression(C=1.0).
  - Use the test_task's selection.json to evaluate.
  - When train_task == test_task, this is the "in-task" CV upper bound from the
    saved exp 03/06/07 results.
  - When train_task != test_task, the probe predicts test_task's labels using
    the direction it learned for train_task. Tells us whether harm tiers share
    a single direction or are orthogonal.

Output:
  - results.json : 3×3 matrix of test AUCs + per-pair n_samples
  - matrix.png   : heatmap visualisation

CPU only.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import get_label_for_task, load_dataset  # noqa: E402

EXTRACTS = Path("/home/ubuntu/extracts/cyber_all_omar")
OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "cyber_1": {
        "selection": REPO_ROOT / "experiments" / "03_layer_sweep_omar" / "selection.json",
        "best_layer": 40,
        "best_pooling": "mean",
    },
    "cyber_2": {
        "selection": REPO_ROOT / "experiments" / "06_cyber2_extract_omar" / "selection.json",
        "best_layer": 40,
        "best_pooling": "mean",
    },
    "cyber_3": {
        "selection": REPO_ROOT / "experiments" / "07_cyber3_extract_omar" / "selection.json",
        "best_layer": 35,
        "best_pooling": "mean",
    },
}


def pool_one(p, layer_idx_position, pooling):
    ex = torch.load(str(p), weights_only=False)
    residuals = ex["residuals"].float()  # (n_layers, n_tok, d)
    mask = ex["attention_mask"].bool()
    n = int(mask.sum().item())
    if n == 0:
        return None
    if pooling == "mean":
        m = mask.float().unsqueeze(0).unsqueeze(-1)
        feat = ((residuals * m).sum(dim=1) / n)[layer_idx_position].numpy()
    else:  # last_token
        last_idx = int(mask.nonzero().max().item())
        feat = residuals[layer_idx_position, last_idx, :].numpy()
    return feat


def load_task(spec, layer):
    """Load features + labels for spec's task at the requested layer."""
    sel = json.loads(spec["selection"].read_text())
    ids = [s["sample_id"] for s in sel["samples"]]
    rows = {s["sample_id"]: s for s in load_dataset("cyber", split="train")}
    layer_idx_position = None  # determined from first extract

    feats, labels, kept = [], [], []
    for sid in ids:
        p = EXTRACTS / f"{sid}.pt"
        if not p.exists():
            continue
        if layer_idx_position is None:
            ex = torch.load(str(p), weights_only=False)
            layer_idx_position = list(ex["layer_idxs"]).index(layer)
        feat = pool_one(p, layer_idx_position, spec["best_pooling"])
        if feat is None or not np.isfinite(feat).all():
            continue
        feats.append(feat)
        kept.append(sid)
    return np.stack(feats).astype(np.float32), kept


def get_labels(sample_ids, task_name):
    samples_by_id = {s["sample_id"]: s for s in load_dataset("cyber", split="train")}
    y = []
    keep_mask = []
    for sid in sample_ids:
        s = samples_by_id.get(sid)
        if s is None:
            keep_mask.append(False)
            y.append(0)
            continue
        lbl = get_label_for_task(s, task_name)
        if lbl is None:
            keep_mask.append(False)
            y.append(0)
        else:
            keep_mask.append(True)
            y.append(int(lbl))
    return np.asarray(y), np.asarray(keep_mask)


def main():
    print("[Cross-task generalization] mean-pool, each task at its own best layer", flush=True)
    cache = {}
    matrix = {tr: {} for tr in TASKS}
    n_samples = {tr: {} for tr in TASKS}

    for tr_name, tr_spec in TASKS.items():
        # Load features at the train task's best layer
        layer = tr_spec["best_layer"]
        if (tr_name, layer) not in cache:
            print(f"\nLoading {tr_name} features (layer {layer}, {tr_spec['best_pooling']})...", flush=True)
            cache[(tr_name, layer)] = load_task(tr_spec, layer)
        X_tr, ids_tr = cache[(tr_name, layer)]

        # In-scope train labels
        y_tr_full, mask_tr = get_labels(ids_tr, tr_name)
        X_tr = X_tr[mask_tr]
        y_tr = y_tr_full[mask_tr]
        print(f"  {tr_name}: train n={len(y_tr)} (pos={int((y_tr==1).sum())}, neg={int((y_tr==0).sum())})", flush=True)

        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(X_tr, y_tr)

        # Predict on each test task — features at the SAME layer
        for te_name, te_spec in TASKS.items():
            if (te_name, layer) not in cache:
                cache[(te_name, layer)] = load_task(te_spec, layer)
            X_te_all, ids_te = cache[(te_name, layer)]
            y_te_full, mask_te = get_labels(ids_te, te_name)
            X_te = X_te_all[mask_te]
            y_te = y_te_full[mask_te]
            if len(set(y_te.tolist())) < 2:
                matrix[tr_name][te_name] = float("nan")
                n_samples[tr_name][te_name] = int(len(y_te))
                continue
            p_te = clf.predict_proba(X_te)[:, 1]
            auc = float(roc_auc_score(y_te, p_te))
            matrix[tr_name][te_name] = auc
            n_samples[tr_name][te_name] = int(len(y_te))
            print(f"    train {tr_name} → test {te_name}: AUC={auc:.4f}  (n={len(y_te)})", flush=True)

    out = {
        "tasks": list(TASKS.keys()),
        "best_layers": {k: v["best_layer"] for k, v in TASKS.items()},
        "matrix_test_auc": matrix,
        "n_samples": n_samples,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT/'results.json'}")

    # Heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(TASKS.keys())
        M = np.array([[matrix[tr][te] for te in names] for tr in names])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        color="black" if 0.6 < M[i,j] < 0.95 else "white", fontsize=11)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels([f"test\n{n}" for n in names])
        ax.set_yticklabels([f"train {n}" for n in names])
        ax.set_title("Cross-task probe transfer (mean-pool, each task's own best layer)")
        fig.colorbar(im, ax=ax, label="Test AUC")
        fig.tight_layout()
        fig.savefig(OUT / "matrix.png", dpi=150)
        plt.close(fig)
        print(f"wrote {OUT/'matrix.png'}")
    except Exception as e:
        print(f"  [warn] matrix plot failed: {e}")


if __name__ == "__main__":
    main()
