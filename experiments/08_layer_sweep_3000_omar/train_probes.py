"""Train logistic-regression probes on the 3190-sample cyber_1 selection.

Two evaluation regimes per (pooling, layer):
  1) 5-fold stratified CV on the TRAIN samples only (the 2267 train rows in
     selection.json) — comparable to exp 03's 5-fold CV.
  2) Single fit on all train samples + evaluation on all 923 cyber-test
     samples — this is the project's official held-out benchmark.

Reported metrics:
  CV: per-fold + mean/std AUC, accuracy, train-AUC (overfit diagnostic)
  Held-out: AUC, acc, bootstrap 95% CI on test predictions

Adapted from experiments/03_layer_sweep_omar/train_probes.py — same probe
config (sklearn LogisticRegression, C=1.0, max_iter=2000, lbfgs, single seed).

Incremental saves:
  - metrics.jsonl: one line per fold + one per held-out eval
  - results.json: re-written atomically after every (pooling, layer) completes
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

EXTRACTS_DIR = Path("/home/ubuntu/extracts/cyber_all_omar")
SELECTION = Path(__file__).parent / "selection.json"
OUT_PATH = Path(__file__).parent / "results.json"
METRICS_LOG = Path(__file__).parent / "metrics.jsonl"
TASK = "cyber_1"
SEED = 0
N_FOLDS = 5
N_BOOTSTRAP = 1000


def atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def append_jsonl(path: Path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def load_pooled_features():
    """Stream every .pt extract listed in selection.json, build mean + last-token features.

    Returns:
        X_mean:  (N_samples, n_layers, d_model)  float32
        X_last:  (N_samples, n_layers, d_model)  float32
        sample_ids: list[str]   (only those present on disk)
        splits:     list[str]   ('train' or 'test') aligned to sample_ids
        layer_idxs: list[int]
    """
    sel = json.loads(SELECTION.read_text())
    rows = sel["samples"]
    split_by_id = {r["sample_id"]: r["split"] for r in rows}

    pt_paths = [(r["sample_id"], EXTRACTS_DIR / f"{r['sample_id']}.pt") for r in rows]
    missing = [sid for sid, p in pt_paths if not p.exists()]
    if missing:
        print(f"  [warn] {len(missing)} selected extracts are missing on disk "
              f"(first: {missing[0]}). Continuing with the rest.", flush=True)
    pt_paths = [(sid, p) for sid, p in pt_paths if p.exists()]

    mean_per_sample, last_per_sample, sample_ids, splits = [], [], [], []
    layer_idxs = None

    for i, (sid, p) in enumerate(pt_paths):
        ex = torch.load(str(p), weights_only=False)
        residuals = ex["residuals"].float()         # (n_layers, N_tok, d)
        mask = ex["attention_mask"].bool()          # (N_tok,)
        if mask.sum().item() == 0:
            continue
        m_f = mask.float().unsqueeze(0).unsqueeze(-1)  # (1, N_tok, 1)
        mean_pooled = (residuals * m_f).sum(dim=1) / mask.sum().item()
        last_idx = int(mask.nonzero().max().item())
        last_token = residuals[:, last_idx, :]

        mean_per_sample.append(mean_pooled.numpy().astype(np.float32))
        last_per_sample.append(last_token.numpy().astype(np.float32))
        sample_ids.append(ex["sample_id"])
        splits.append(split_by_id[sid])
        if layer_idxs is None:
            layer_idxs = list(ex["layer_idxs"])
        if (i + 1) % 200 == 0:
            print(f"  pooled {i+1}/{len(pt_paths)}", flush=True)

    X_mean = np.stack(mean_per_sample, axis=0)
    X_last = np.stack(last_per_sample, axis=0)
    return X_mean, X_last, sample_ids, splits, layer_idxs


def get_labels(sample_ids, task):
    """Return y, in_scope arrays. Joins train+test samples by sample_id."""
    by_id = {}
    for split in ("train", "test"):
        for s in load_dataset("cyber", split=split):
            by_id[s["sample_id"]] = s
    y = np.full(len(sample_ids), -1, dtype=np.int64)
    in_scope = np.zeros(len(sample_ids), dtype=bool)
    for i, sid in enumerate(sample_ids):
        s = by_id.get(sid)
        if s is None:
            continue
        lbl = get_label_for_task(s, task)
        if lbl is None:
            continue
        y[i] = lbl
        in_scope[i] = True
    return y, in_scope


def bootstrap_auc_ci(y_true, p, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(y_true[idx].tolist())) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], p[idx]))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def cv_on_train(X_layer_train, y_train, n_folds=N_FOLDS, seed=SEED, log_meta=None):
    """5-fold stratified CV on the training samples; returns summary metrics."""
    if len(set(y_train.tolist())) < 2:
        return None
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_layer_train, y_train)):
        X_tr, X_te = X_layer_train[tr_idx], X_layer_train[te_idx]
        y_tr, y_te = y_train[tr_idx], y_train[te_idx]
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        p_te = clf.predict_proba(X_te)[:, 1]
        p_tr = clf.predict_proba(X_tr)[:, 1]
        fm = {
            "fold": fold, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "auc": float(roc_auc_score(y_te, p_te)),
            "acc": float(((p_te > 0.5).astype(int) == y_te).mean()),
            "train_auc": float(roc_auc_score(y_tr, p_tr)),
            "train_acc": float(((p_tr > 0.5).astype(int) == y_tr).mean()),
        }
        fold_metrics.append(fm)
        if log_meta is not None:
            append_jsonl(METRICS_LOG, {**log_meta, "phase": "cv_fold", **fm})
    aucs = np.array([m["auc"] for m in fold_metrics])
    accs = np.array([m["acc"] for m in fold_metrics])
    train_aucs = np.array([m["train_auc"] for m in fold_metrics])
    train_accs = np.array([m["train_acc"] for m in fold_metrics])
    return {
        "n_folds": int(n_folds),
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=1)),
        "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()),
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=1)),
        "train_auc_mean": float(train_aucs.mean()),
        "train_acc_mean": float(train_accs.mean()),
        "fold_metrics": fold_metrics,
    }


def heldout_eval(X_layer_train, y_train, X_layer_test, y_test, log_meta=None):
    """Fit on all train, evaluate on all test. Returns summary metrics."""
    if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        return None
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_layer_train, y_train)
    p_te = clf.predict_proba(X_layer_test)[:, 1]
    p_tr = clf.predict_proba(X_layer_train)[:, 1]
    auc = float(roc_auc_score(y_test, p_te))
    acc = float(((p_te > 0.5).astype(int) == y_test).mean())
    auc_lo, auc_hi = bootstrap_auc_ci(y_test, p_te)
    out = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "auc": auc,
        "acc": acc,
        "auc_ci95_lo": auc_lo,
        "auc_ci95_hi": auc_hi,
        "train_auc": float(roc_auc_score(y_train, p_tr)),
        "train_acc": float(((p_tr > 0.5).astype(int) == y_train).mean()),
    }
    if log_meta is not None:
        append_jsonl(METRICS_LOG, {**log_meta, "phase": "heldout", **out})
    return out


def main():
    if METRICS_LOG.exists():
        METRICS_LOG.unlink()

    print(f"Loading + pooling extracts from {EXTRACTS_DIR}...", flush=True)
    X_mean, X_last, sample_ids, splits, layer_idxs = load_pooled_features()
    splits = np.array(splits)
    is_train = splits == "train"
    is_test = splits == "test"
    print(f"X_mean shape: {X_mean.shape}", flush=True)
    print(f"X_last shape: {X_last.shape}", flush=True)
    print(f"layer_idxs:   {layer_idxs}", flush=True)
    print(f"split sizes:  train={int(is_train.sum())}, test={int(is_test.sum())}", flush=True)

    y, in_scope = get_labels(sample_ids, TASK)
    if not in_scope.all():
        print(f"  [warn] {(~in_scope).sum()} samples not in scope of {TASK}", flush=True)

    train_mask = is_train & in_scope
    test_mask = is_test & in_scope
    n_pos_tr = int((y[train_mask] == 1).sum())
    n_neg_tr = int((y[train_mask] == 0).sum())
    n_pos_te = int((y[test_mask] == 1).sum())
    n_neg_te = int((y[test_mask] == 0).sum())
    print(f"\n=== {TASK} ===", flush=True)
    print(f"  train: pos={n_pos_tr} neg={n_neg_tr} (total={n_pos_tr+n_neg_tr})", flush=True)
    print(f"  test:  pos={n_pos_te} neg={n_neg_te} (total={n_pos_te+n_neg_te})", flush=True)

    results = {
        "extracts_dir": str(EXTRACTS_DIR),
        "n_samples_total": int(len(sample_ids)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "train_label_counts": {"pos": n_pos_tr, "neg": n_neg_tr},
        "test_label_counts": {"pos": n_pos_te, "neg": n_neg_te},
        "layer_idxs": layer_idxs,
        "d_model": int(X_mean.shape[-1]),
        "poolings": ["mean", "last_token"],
        "task": TASK,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_bootstrap": N_BOOTSTRAP,
        "cv": {"per_pooling": {}},
        "held_out_test": {"per_pooling": {}},
    }

    pooling_arrays = {"mean": X_mean, "last_token": X_last}

    for pname, X in pooling_arrays.items():
        print(f"\n  -- pooling = {pname} --", flush=True)
        per_layer_cv = []
        per_layer_ho = []
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        for li, layer_idx in enumerate(layer_idxs):
            log_meta = {"task": TASK, "pooling": pname, "layer": int(layer_idx)}
            cv_metrics = cv_on_train(X_train[:, li, :], y_train, log_meta=log_meta)
            ho_metrics = heldout_eval(X_train[:, li, :], y_train,
                                      X_test[:, li, :], y_test, log_meta=log_meta)
            cv_row = {"layer": int(layer_idx), **(cv_metrics or {"auc_mean": float("nan")})}
            ho_row = {"layer": int(layer_idx), **(ho_metrics or {"auc": float("nan")})}
            per_layer_cv.append(cv_row)
            per_layer_ho.append(ho_row)
            if cv_metrics is not None and ho_metrics is not None:
                print(f"    layer {layer_idx:>2d}: "
                      f"CV AUC={cv_metrics['auc_mean']:.3f}±{cv_metrics['auc_std']:.3f} "
                      f"(min {cv_metrics['auc_min']:.3f}, max {cv_metrics['auc_max']:.3f}) | "
                      f"CV acc={cv_metrics['acc_mean']:.3f} | "
                      f"train AUC={cv_metrics['train_auc_mean']:.3f} | "
                      f"HELD-OUT AUC={ho_metrics['auc']:.3f} "
                      f"[{ho_metrics['auc_ci95_lo']:.3f}, {ho_metrics['auc_ci95_hi']:.3f}] "
                      f"acc={ho_metrics['acc']:.3f}",
                      flush=True)
            results["cv"]["per_pooling"][pname] = per_layer_cv
            results["held_out_test"]["per_pooling"][pname] = per_layer_ho
            atomic_write_json(OUT_PATH, results)

    atomic_write_json(OUT_PATH, results)
    print(f"\nResults written to {OUT_PATH}")
    print(f"Per-fold log: {METRICS_LOG}")


if __name__ == "__main__":
    main()
