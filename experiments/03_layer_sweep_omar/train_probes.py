"""Train one logistic-regression probe per (pooling, layer, cyber-task) with 5-fold CV.

Loads .pt extracts from /home/ubuntu/extracts/03_layer_sweep_omar/, builds two
feature views per (sample, layer):
  - mean: average over attention-masked tokens
  - last_token: residual at the final masked-token position
and sweeps pooling x layers x tasks. For each (pooling, layer, task) runs
5-fold stratified CV and reports per-fold AUC/accuracy plus mean ± std.

Tasks: cyber_1 (dual_use vs benign), cyber_2, cyber_3.
For the cyber_1-only subset, cyber_2 / cyber_3 are `skipped_one_class`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import get_label_for_task, load_dataset

EXTRACTS_DIR = Path("/home/ubuntu/extracts/03_layer_sweep_omar")
SELECTION = Path(__file__).parent / "selection.json"
OUT_PATH = Path(__file__).parent / "results.json"
METRICS_LOG = Path(__file__).parent / "metrics.jsonl"
TASKS = ["cyber_1", "cyber_2", "cyber_3"]
SEED = 0
N_FOLDS = 5


def atomic_write_json(path: Path, obj):
    """Write JSON to a tmp file then rename — never leaves a partial results.json."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def append_jsonl(path: Path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def load_pooled_features():
    """Stream every .pt extract listed in selection.json, build mean + last-token features.

    Returns:
        X_mean:  (N_samples, n_layers, d_model)  float32 — mean over masked tokens
        X_last:  (N_samples, n_layers, d_model)  float32 — final masked-token residual
        sample_ids: list[str]
        layer_idxs: list[int]
    """
    sel = json.loads(SELECTION.read_text())
    selected_ids = [row["sample_id"] for row in sel["samples"]]
    pt_paths = [EXTRACTS_DIR / f"{sid}.pt" for sid in selected_ids]
    missing = [p for p in pt_paths if not p.exists()]
    if missing:
        print(f"  [warn] {len(missing)} selected extracts are missing on disk "
              f"(first: {missing[0].name}). Continuing with the rest.", flush=True)
        pt_paths = [p for p in pt_paths if p.exists()]

    mean_per_sample, last_per_sample, sample_ids = [], [], []
    layer_idxs = None

    for i, p in enumerate(pt_paths):
        ex = torch.load(str(p), weights_only=False)
        residuals = ex["residuals"].float()         # (n_layers, N_tok, d)
        mask = ex["attention_mask"].bool()          # (N_tok,)
        if mask.sum().item() == 0:
            continue
        m_f = mask.float().unsqueeze(0).unsqueeze(-1)  # (1, N_tok, 1)
        mean_pooled = (residuals * m_f).sum(dim=1) / mask.sum().item()  # (n_layers, d)
        last_idx = int(mask.nonzero().max().item())
        last_token = residuals[:, last_idx, :]  # (n_layers, d)

        mean_per_sample.append(mean_pooled.numpy().astype(np.float32))
        last_per_sample.append(last_token.numpy().astype(np.float32))
        sample_ids.append(ex["sample_id"])
        if layer_idxs is None:
            layer_idxs = list(ex["layer_idxs"])
        if (i + 1) % 100 == 0:
            print(f"  pooled {i+1}/{len(pt_paths)}", flush=True)

    X_mean = np.stack(mean_per_sample, axis=0)
    X_last = np.stack(last_per_sample, axis=0)
    return X_mean, X_last, sample_ids, layer_idxs


def get_labels(sample_ids, task):
    """Return (y, mask) where mask=True for samples in scope of `task`."""
    samples_by_id = {
        s["sample_id"]: s for s in load_dataset("cyber", split="train")
    }
    y = np.full(len(sample_ids), -1, dtype=np.int64)
    in_scope = np.zeros(len(sample_ids), dtype=bool)
    for i, sid in enumerate(sample_ids):
        s = samples_by_id.get(sid)
        if s is None:
            continue
        lbl = get_label_for_task(s, task)
        if lbl is None:
            continue
        y[i] = lbl
        in_scope[i] = True
    return y, in_scope


def train_one_cv(X_layer, y, in_scope, n_folds=N_FOLDS, seed=SEED,
                  log_meta=None):
    """5-fold stratified CV for a logistic probe; return per-fold + summary metrics.

    Each completed fold is appended to METRICS_LOG immediately so a crash
    leaves a partial trail to inspect.
    """
    X = X_layer[in_scope]
    y_in = y[in_scope]
    if len(set(y_in.tolist())) < 2:
        return None
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []
    train_aucs, train_accs = [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y_in)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y_in[tr_idx], y_in[te_idx]
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        p_te = clf.predict_proba(X_te)[:, 1]
        p_tr = clf.predict_proba(X_tr)[:, 1]
        auc = float(roc_auc_score(y_te, p_te))
        acc = float(((p_te > 0.5).astype(int) == y_te).mean())
        auc_tr = float(roc_auc_score(y_tr, p_tr))
        acc_tr = float(((p_tr > 0.5).astype(int) == y_tr).mean())
        fm = {
            "fold": fold, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "auc": auc, "acc": acc,
            "train_auc": auc_tr, "train_acc": acc_tr,
        }
        fold_metrics.append(fm)
        train_aucs.append(auc_tr); train_accs.append(acc_tr)
        if log_meta is not None:
            append_jsonl(METRICS_LOG, {**log_meta, **fm})
    aucs = np.array([m["auc"] for m in fold_metrics])
    accs = np.array([m["acc"] for m in fold_metrics])
    return {
        "n_folds": int(n_folds),
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=1)),
        "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()),
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=1)),
        "train_auc_mean": float(np.mean(train_aucs)),
        "train_acc_mean": float(np.mean(train_accs)),
        "fold_metrics": fold_metrics,
    }


def main():
    # Reset the per-fold log on each run (results.json is the durable artifact).
    if METRICS_LOG.exists():
        METRICS_LOG.unlink()
    print(f"Loading + pooling extracts from {EXTRACTS_DIR}...", flush=True)
    X_mean, X_last, sample_ids, layer_idxs = load_pooled_features()
    print(f"X_mean shape: {X_mean.shape} (samples, layers, d_model)", flush=True)
    print(f"X_last shape: {X_last.shape}", flush=True)
    print(f"layer_idxs: {layer_idxs}", flush=True)

    results = {
        "extracts_dir": str(EXTRACTS_DIR),
        "n_samples": len(sample_ids),
        "layer_idxs": layer_idxs,
        "d_model": int(X_mean.shape[-1]),
        "poolings": ["mean", "last_token"],
        "tasks": {},
    }

    pooling_arrays = {"mean": X_mean, "last_token": X_last}

    for task in TASKS:
        y, in_scope = get_labels(sample_ids, task)
        n_in = int(in_scope.sum())
        n_pos = int((y[in_scope] == 1).sum()) if n_in else 0
        n_neg = n_in - n_pos
        print(f"\n=== {task}: {n_in} in scope (pos={n_pos}, neg={n_neg}) ===", flush=True)
        if n_pos == 0 or n_neg == 0:
            results["tasks"][task] = {"status": "skipped_one_class",
                                      "n_in_scope": n_in, "n_pos": n_pos, "n_neg": n_neg,
                                      "per_pooling": {}}
            print("  Only one class present; skipping (expected for cyber_1-only subset).", flush=True)
            continue

        task_block = {"status": "ok", "n_pos": n_pos, "n_neg": n_neg, "per_pooling": {}}
        for pname, X in pooling_arrays.items():
            print(f"  -- pooling = {pname} ({N_FOLDS}-fold CV) --", flush=True)
            per_layer = []
            for li, layer_idx in enumerate(layer_idxs):
                log_meta = {"task": task, "pooling": pname, "layer": int(layer_idx)}
                metrics = train_one_cv(X[:, li, :], y, in_scope, log_meta=log_meta)
                row = {"layer": int(layer_idx), **(metrics or {"auc_mean": float("nan")})}
                per_layer.append(row)
                if metrics is not None:
                    print(f"    layer {layer_idx:>2d}: "
                          f"test AUC={metrics['auc_mean']:.3f}±{metrics['auc_std']:.3f} "
                          f"(min {metrics['auc_min']:.3f}, max {metrics['auc_max']:.3f}) | "
                          f"acc={metrics['acc_mean']:.3f}±{metrics['acc_std']:.3f} | "
                          f"train AUC={metrics['train_auc_mean']:.3f}", flush=True)
                # Snapshot results.json after every layer so a crash mid-run keeps progress.
                task_block["per_pooling"][pname] = per_layer
                results["tasks"][task] = task_block
                atomic_write_json(OUT_PATH, results)
        results["tasks"][task] = task_block

    atomic_write_json(OUT_PATH, results)
    print(f"\nResults written to {OUT_PATH}")
    print(f"Per-fold log: {METRICS_LOG}")


if __name__ == "__main__":
    main()
