"""HONEST transfer + combined-data eval (cyber_3 ↔ refusal_gemma).

Train on `split=='train'` only; eval on `split=='test'`.

Three LR probes per layer ∈ {30, 35, 40, 45} and three pooling strategies
(mean L_x, last L_x, multi-concat):

  T1. cyber_3-only   — train on cyber_3 train, eval on cyber_3 test AND
                        refusal_gemma test
  T2. refusal-only   — train on refusal train, eval on refusal test AND
                        cyber_3 test
  T3. combined       — train on UNION (cyber_3_train ∪ refusal_train),
                        labeled by their own task's positive class. Eval
                        on EACH test split with that task's labels.

Output:
  transfer_combined.json — {variant: {pooling, layer, train_set, test_task,
                                       n_train, n_test, train_auc, test_auc}}
  transfer_combined.md   — pretty table
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

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

CYBER_EXTRACTS = Path("/home/ubuntu/extracts/cyber_all_omar")
REFUSAL_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b")
CYBER_TRAIN = REPO_ROOT / "datasets" / "cyber_probes" / "train.jsonl"
CYBER_TEST = REPO_ROOT / "datasets" / "cyber_probes" / "test.jsonl"
REFUSAL_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
LAYERS_TO_TEST = [30, 35, 40, 45]


def cyber_3_label(category):
    if category == "prohibited": return 1
    if category in ("high_risk_dual_use", "dual_use", "benign"): return 0
    return None


def load_cyber_3_features(split_path):
    rows = []
    with split_path.open() as f:
        for line in f:
            r = json.loads(line)
            lbl = cyber_3_label(r["category"])
            if lbl is None: continue
            rows.append((r["sample_id"], lbl))
    means, lasts, labels, sids = [], [], [], []
    for sid, lbl in rows:
        p = CYBER_EXTRACTS / f"{sid}.pt"
        if not p.exists(): continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
        residuals = ex["residuals"].float()
        mask = ex["attention_mask"].bool().squeeze()
        n = int(mask.sum().item())
        if n < 2: continue
        last_idx = int(mask.nonzero().max().item())
        m = mask.float().unsqueeze(0).unsqueeze(-1)
        feat_mean = ((residuals * m).sum(dim=1) / n).numpy().astype(np.float32)
        feat_last = residuals[:, last_idx, :].numpy().astype(np.float32)
        if not np.isfinite(feat_mean).all() or not np.isfinite(feat_last).all(): continue
        means.append(feat_mean); lasts.append(feat_last)
        labels.append(lbl); sids.append(sid)
    return np.stack(means), np.stack(lasts), np.asarray(labels, np.int64), sids


def load_refusal_features(target_split):
    rows = []
    with REFUSAL_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != target_split: continue
            if r.get("is_refusal") is None: continue
            rows.append((r["sample_id"], int(bool(r["is_refusal"]))))
    means, lasts, labels, sids = [], [], [], []
    for sid, lbl in rows:
        p = REFUSAL_EXTRACTS / f"{sid}.pt"
        if not p.exists(): continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
        residuals = ex["residuals"].float()
        mask = ex["attention_mask"].bool().squeeze()
        n = int(mask.sum().item())
        if n < 2: continue
        last_idx = int(mask.nonzero().max().item())
        m = mask.float().unsqueeze(0).unsqueeze(-1)
        feat_mean = ((residuals * m).sum(dim=1) / n).numpy().astype(np.float32)
        feat_last = residuals[:, last_idx, :].numpy().astype(np.float32)
        if not np.isfinite(feat_mean).all() or not np.isfinite(feat_last).all(): continue
        means.append(feat_mean); lasts.append(feat_last)
        labels.append(lbl); sids.append(sid)
    return np.stack(means), np.stack(lasts), np.asarray(labels, np.int64), sids


def main():
    print("[transfer + combined eval]", flush=True)
    print("loading cyber_3 train+test...", flush=True)
    t0 = time.time()
    Xc_tr_m, Xc_tr_l, yc_tr, _ = load_cyber_3_features(CYBER_TRAIN)
    Xc_te_m, Xc_te_l, yc_te, _ = load_cyber_3_features(CYBER_TEST)
    print(f"  cyber_3 train {Xc_tr_m.shape}, test {Xc_te_m.shape} ({time.time()-t0:.0f}s)", flush=True)

    print("loading refusal train+test...", flush=True)
    t0 = time.time()
    Xr_tr_m, Xr_tr_l, yr_tr, _ = load_refusal_features("train")
    Xr_te_m, Xr_te_l, yr_te, _ = load_refusal_features("test")
    print(f"  refusal train {Xr_tr_m.shape}, test {Xr_te_m.shape} ({time.time()-t0:.0f}s)", flush=True)

    results = []  # list of {variant, pooling, layer, train_set, test_task, n_train, n_test, auc}

    for L in LAYERS_TO_TEST:
        li = LAYERS.index(L)
        for pooling in ["mean", "last"]:
            print(f"\n--- L{L} {pooling} ---", flush=True)
            X_c_tr = (Xc_tr_m[:, li, :] if pooling == "mean" else Xc_tr_l[:, li, :])
            X_c_te = (Xc_te_m[:, li, :] if pooling == "mean" else Xc_te_l[:, li, :])
            X_r_tr = (Xr_tr_m[:, li, :] if pooling == "mean" else Xr_tr_l[:, li, :])
            X_r_te = (Xr_te_m[:, li, :] if pooling == "mean" else Xr_te_l[:, li, :])

            # T1. cyber_3-only
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(X_c_tr, yc_tr)
            auc_cc = roc_auc_score(yc_te, clf.predict_proba(X_c_te)[:, 1])
            auc_cr = roc_auc_score(yr_te, clf.predict_proba(X_r_te)[:, 1])
            print(f"  cyber_3-only  → cyber_3 test  AUC={auc_cc:.4f}", flush=True)
            print(f"                 → refusal test  AUC={auc_cr:.4f}  (TRANSFER)", flush=True)
            results.append({"variant": "cyber_3_only", "pooling": pooling, "layer": L,
                            "train_set": "cyber_3_train", "test_task": "cyber_3",
                            "n_train": int(len(yc_tr)), "n_test": int(len(yc_te)), "auc": float(auc_cc)})
            results.append({"variant": "cyber_3_only", "pooling": pooling, "layer": L,
                            "train_set": "cyber_3_train", "test_task": "refusal_gemma",
                            "n_train": int(len(yc_tr)), "n_test": int(len(yr_te)), "auc": float(auc_cr)})

            # T2. refusal-only
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(X_r_tr, yr_tr)
            auc_rr = roc_auc_score(yr_te, clf.predict_proba(X_r_te)[:, 1])
            auc_rc = roc_auc_score(yc_te, clf.predict_proba(X_c_te)[:, 1])
            print(f"  refusal-only  → refusal test  AUC={auc_rr:.4f}", flush=True)
            print(f"                 → cyber_3 test  AUC={auc_rc:.4f}  (TRANSFER)", flush=True)
            results.append({"variant": "refusal_only", "pooling": pooling, "layer": L,
                            "train_set": "refusal_train", "test_task": "refusal_gemma",
                            "n_train": int(len(yr_tr)), "n_test": int(len(yr_te)), "auc": float(auc_rr)})
            results.append({"variant": "refusal_only", "pooling": pooling, "layer": L,
                            "train_set": "refusal_train", "test_task": "cyber_3",
                            "n_train": int(len(yr_tr)), "n_test": int(len(yc_te)), "auc": float(auc_rc)})

            # T3. combined — concat training data, labels = task-1 positive (treat as one
            # 'severe-harm' direction)
            X_combo = np.concatenate([X_c_tr, X_r_tr], axis=0)
            y_combo = np.concatenate([yc_tr, yr_tr], axis=0)
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(X_combo, y_combo)
            auc_cc2 = roc_auc_score(yc_te, clf.predict_proba(X_c_te)[:, 1])
            auc_rr2 = roc_auc_score(yr_te, clf.predict_proba(X_r_te)[:, 1])
            print(f"  combined      → cyber_3 test  AUC={auc_cc2:.4f}", flush=True)
            print(f"                 → refusal test  AUC={auc_rr2:.4f}", flush=True)
            results.append({"variant": "combined", "pooling": pooling, "layer": L,
                            "train_set": "cyber_3_train ∪ refusal_train",
                            "test_task": "cyber_3", "n_train": int(len(y_combo)),
                            "n_test": int(len(yc_te)), "auc": float(auc_cc2)})
            results.append({"variant": "combined", "pooling": pooling, "layer": L,
                            "train_set": "cyber_3_train ∪ refusal_train",
                            "test_task": "refusal_gemma", "n_train": int(len(y_combo)),
                            "n_test": int(len(yr_te)), "auc": float(auc_rr2)})

    # Multi-layer concat (mean over 13 layers)
    print(f"\n--- multi-layer concat (mean, 13 layers) ---", flush=True)
    Xc_tr_concat = Xc_tr_m.reshape(len(Xc_tr_m), -1)
    Xc_te_concat = Xc_te_m.reshape(len(Xc_te_m), -1)
    Xr_tr_concat = Xr_tr_m.reshape(len(Xr_tr_m), -1)
    Xr_te_concat = Xr_te_m.reshape(len(Xr_te_m), -1)

    clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(Xc_tr_concat, yc_tr)
    auc_cc = roc_auc_score(yc_te, clf.predict_proba(Xc_te_concat)[:, 1])
    auc_cr = roc_auc_score(yr_te, clf.predict_proba(Xr_te_concat)[:, 1])
    print(f"  cyber_3-only concat → cyber_3 AUC={auc_cc:.4f}, refusal AUC={auc_cr:.4f}", flush=True)
    results.append({"variant": "cyber_3_only", "pooling": "concat_13_mean",
                    "layer": "all", "train_set": "cyber_3_train", "test_task": "cyber_3",
                    "n_train": int(len(yc_tr)), "n_test": int(len(yc_te)), "auc": float(auc_cc)})
    results.append({"variant": "cyber_3_only", "pooling": "concat_13_mean",
                    "layer": "all", "train_set": "cyber_3_train", "test_task": "refusal_gemma",
                    "n_train": int(len(yc_tr)), "n_test": int(len(yr_te)), "auc": float(auc_cr)})

    clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(Xr_tr_concat, yr_tr)
    auc_rr = roc_auc_score(yr_te, clf.predict_proba(Xr_te_concat)[:, 1])
    auc_rc = roc_auc_score(yc_te, clf.predict_proba(Xc_te_concat)[:, 1])
    print(f"  refusal-only concat → refusal AUC={auc_rr:.4f}, cyber_3 AUC={auc_rc:.4f}", flush=True)
    results.append({"variant": "refusal_only", "pooling": "concat_13_mean",
                    "layer": "all", "train_set": "refusal_train", "test_task": "refusal_gemma",
                    "n_train": int(len(yr_tr)), "n_test": int(len(yr_te)), "auc": float(auc_rr)})
    results.append({"variant": "refusal_only", "pooling": "concat_13_mean",
                    "layer": "all", "train_set": "refusal_train", "test_task": "cyber_3",
                    "n_train": int(len(yr_tr)), "n_test": int(len(yc_te)), "auc": float(auc_rc)})

    X_combo = np.concatenate([Xc_tr_concat, Xr_tr_concat], axis=0)
    y_combo = np.concatenate([yc_tr, yr_tr], axis=0)
    clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(X_combo, y_combo)
    auc_cc2 = roc_auc_score(yc_te, clf.predict_proba(Xc_te_concat)[:, 1])
    auc_rr2 = roc_auc_score(yr_te, clf.predict_proba(Xr_te_concat)[:, 1])
    print(f"  combined concat     → cyber_3 AUC={auc_cc2:.4f}, refusal AUC={auc_rr2:.4f}", flush=True)
    results.append({"variant": "combined", "pooling": "concat_13_mean", "layer": "all",
                    "train_set": "cyber_3_train ∪ refusal_train",
                    "test_task": "cyber_3", "n_train": int(len(y_combo)),
                    "n_test": int(len(yc_te)), "auc": float(auc_cc2)})
    results.append({"variant": "combined", "pooling": "concat_13_mean", "layer": "all",
                    "train_set": "cyber_3_train ∪ refusal_train",
                    "test_task": "refusal_gemma", "n_train": int(len(y_combo)),
                    "n_test": int(len(yr_te)), "auc": float(auc_rr2)})

    # Save
    out_json = HERE / "transfer_combined.json"
    out_json.write_text(json.dumps(results, indent=2))

    # Pretty md
    md = ["# HONEST transfer + combined eval: cyber_3 ↔ refusal_gemma\n"]
    md.append("Train on `split=='train'` only; eval on `split=='test'`.")
    md.append("LR (C=1.0 single-layer, C=0.1 13-layer concat).\n")

    md.append("\n## Single-layer mean-pool transfer\n")
    md.append("| Layer | cyber_3-only → cyber_3 | cyber_3-only → refusal | refusal-only → refusal | refusal-only → cyber_3 | combined → cyber_3 | combined → refusal |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for L in LAYERS_TO_TEST:
        c_self = next(r["auc"] for r in results if r["variant"]=="cyber_3_only" and r["pooling"]=="mean" and r["layer"]==L and r["test_task"]=="cyber_3")
        c_xfer = next(r["auc"] for r in results if r["variant"]=="cyber_3_only" and r["pooling"]=="mean" and r["layer"]==L and r["test_task"]=="refusal_gemma")
        r_self = next(r["auc"] for r in results if r["variant"]=="refusal_only" and r["pooling"]=="mean" and r["layer"]==L and r["test_task"]=="refusal_gemma")
        r_xfer = next(r["auc"] for r in results if r["variant"]=="refusal_only" and r["pooling"]=="mean" and r["layer"]==L and r["test_task"]=="cyber_3")
        co_c = next(r["auc"] for r in results if r["variant"]=="combined" and r["pooling"]=="mean" and r["layer"]==L and r["test_task"]=="cyber_3")
        co_r = next(r["auc"] for r in results if r["variant"]=="combined" and r["pooling"]=="mean" and r["layer"]==L and r["test_task"]=="refusal_gemma")
        md.append(f"| L{L} | {c_self:.4f} | {c_xfer:.4f} | {r_self:.4f} | {r_xfer:.4f} | {co_c:.4f} | {co_r:.4f} |")

    md.append("\n## Single-layer last-token transfer\n")
    md.append("| Layer | cyber_3-only → cyber_3 | cyber_3-only → refusal | refusal-only → refusal | refusal-only → cyber_3 | combined → cyber_3 | combined → refusal |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for L in LAYERS_TO_TEST:
        c_self = next(r["auc"] for r in results if r["variant"]=="cyber_3_only" and r["pooling"]=="last" and r["layer"]==L and r["test_task"]=="cyber_3")
        c_xfer = next(r["auc"] for r in results if r["variant"]=="cyber_3_only" and r["pooling"]=="last" and r["layer"]==L and r["test_task"]=="refusal_gemma")
        r_self = next(r["auc"] for r in results if r["variant"]=="refusal_only" and r["pooling"]=="last" and r["layer"]==L and r["test_task"]=="refusal_gemma")
        r_xfer = next(r["auc"] for r in results if r["variant"]=="refusal_only" and r["pooling"]=="last" and r["layer"]==L and r["test_task"]=="cyber_3")
        co_c = next(r["auc"] for r in results if r["variant"]=="combined" and r["pooling"]=="last" and r["layer"]==L and r["test_task"]=="cyber_3")
        co_r = next(r["auc"] for r in results if r["variant"]=="combined" and r["pooling"]=="last" and r["layer"]==L and r["test_task"]=="refusal_gemma")
        md.append(f"| L{L} | {c_self:.4f} | {c_xfer:.4f} | {r_self:.4f} | {r_xfer:.4f} | {co_c:.4f} | {co_r:.4f} |")

    md.append("\n## Multi-layer concat (mean, 13 layers)\n")
    md.append("| Train set | → cyber_3 test | → refusal test |")
    md.append("|---|---:|---:|")
    cc = next(r["auc"] for r in results if r["variant"]=="cyber_3_only" and r["pooling"]=="concat_13_mean" and r["test_task"]=="cyber_3")
    cr = next(r["auc"] for r in results if r["variant"]=="cyber_3_only" and r["pooling"]=="concat_13_mean" and r["test_task"]=="refusal_gemma")
    rr = next(r["auc"] for r in results if r["variant"]=="refusal_only" and r["pooling"]=="concat_13_mean" and r["test_task"]=="refusal_gemma")
    rc = next(r["auc"] for r in results if r["variant"]=="refusal_only" and r["pooling"]=="concat_13_mean" and r["test_task"]=="cyber_3")
    co_c = next(r["auc"] for r in results if r["variant"]=="combined" and r["pooling"]=="concat_13_mean" and r["test_task"]=="cyber_3")
    co_r = next(r["auc"] for r in results if r["variant"]=="combined" and r["pooling"]=="concat_13_mean" and r["test_task"]=="refusal_gemma")
    md.append(f"| cyber_3 only | {cc:.4f} | {cr:.4f} |")
    md.append(f"| refusal only | {rc:.4f} | {rr:.4f} |")
    md.append(f"| combined | {co_c:.4f} | {co_r:.4f} |")

    (HERE / "transfer_combined.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out_json} and {HERE/'transfer_combined.md'}", flush=True)


if __name__ == "__main__":
    main()
