"""HONEST evaluation per the rules: train on split=='train' only, evaluate
on split=='test'. Report per-task test AUC + mean ± std across the 5 tasks
(skipping Qwen — model not downloaded).

Probes evaluated:
  - LR mean L40
  - LR last-tok L45
  - LR multi-layer concat
  - LR mean-of-layers
  - LR max-of-layers
  - Pleshkov d=16 @ best-task-layer
  - MLP probe (L40 mean)
  - Constitutional probe (multi-layer concat)

Each probe is **re-fit on train rows only**, then scored on test rows.

Output:
  honest_per_task.json : {task: {probe: {train_auc, test_auc, n_train, n_test}}}
  honest_summary.json  : {probe: {mean_test_auc, std_test_auc, per_task: {...}}}
  honest_table.md
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)
REPO_ROOT = HERE.parent.parent

CYBER_EXTRACTS = Path("/home/ubuntu/extracts/cyber_all_omar")
REFUSAL_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b")
CYBER_TRAIN = REPO_ROOT / "datasets" / "cyber_probes" / "train.jsonl"
CYBER_TEST = REPO_ROOT / "datasets" / "cyber_probes" / "test.jsonl"
REFUSAL_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

sys.path.insert(0, str(REPO_ROOT / "experiments" / "17_quadratic_probe_omar"))
from probes import QuadraticProbe  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "experiments" / "30_rewrites_causality_omar"))
from train_extra_probes import MLPProbe, ConstitutionalProbe  # noqa: E402


def cyber_label(category, task):
    if task == "cyber_1":
        if category == "dual_use": return 1
        if category == "benign": return 0
        return None
    if task == "cyber_2":
        if category == "high_risk_dual_use": return 1
        if category in ("dual_use", "benign"): return 0
        return None
    if task == "cyber_3":
        if category == "prohibited": return 1
        if category in ("high_risk_dual_use", "dual_use", "benign"): return 0
        return None


def load_cyber_features(split_path, task):
    rows = []
    with split_path.open() as f:
        for line in f:
            r = json.loads(line)
            lbl = cyber_label(r["category"], task)
            if lbl is None: continue
            rows.append((r["sample_id"], lbl))
    means_per_layer = []
    lasts_per_layer = []
    labels = []
    sids = []
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
        means_per_layer.append(feat_mean)
        lasts_per_layer.append(feat_last)
        labels.append(lbl); sids.append(sid)
    return (np.stack(means_per_layer), np.stack(lasts_per_layer),
            np.asarray(labels, dtype=np.int64), sids)


def load_refusal_features(target_split):
    """target_split: 'train' or 'test'"""
    rows = []
    with REFUSAL_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != target_split: continue
            if r.get("is_refusal") is None: continue
            rows.append((r["sample_id"], int(bool(r["is_refusal"]))))
    means_per_layer = []
    lasts_per_layer = []
    labels = []; sids = []
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
        means_per_layer.append(feat_mean)
        lasts_per_layer.append(feat_last)
        labels.append(lbl); sids.append(sid)
    return (np.stack(means_per_layer), np.stack(lasts_per_layer),
            np.asarray(labels, dtype=np.int64), sids)


def lr_eval(X_tr, y_tr, X_te, y_te, C=1.0):
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(X_tr, y_tr)
    return float(roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1])), \
           float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))


def train_torch_probe(probe_class, X_tr_np, y_tr_np, X_te_np, y_te_np,
                       epochs=30, lr=1e-3, batch=32, l2=1e-4, **kw):
    probe = probe_class(**kw)
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    X_te = torch.tensor(X_te_np, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_np, dtype=torch.float32)
    y_te = torch.tensor(y_te_np, dtype=torch.float32)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=l2)
    n = len(X_tr); rng = np.random.default_rng(0)
    for ep in range(epochs):
        probe.train()
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i:i+batch]
            logits = probe(X_tr[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        s_tr = probe(X_tr).numpy()
        s_te = probe(X_te).numpy()
    return float(roc_auc_score(y_tr_np, s_tr)), float(roc_auc_score(y_te_np, s_te))


def eval_task(name, X_tr_mean, X_tr_last, y_tr, X_te_mean, X_te_last, y_te):
    print(f"\n=== {name}: n_train={len(y_tr)}, n_test={len(y_te)} ===", flush=True)
    print(f"  train pos/neg = {int((y_tr==1).sum())}/{int((y_tr==0).sum())}", flush=True)
    print(f"  test  pos/neg = {int((y_te==1).sum())}/{int((y_te==0).sum())}", flush=True)
    out = {}

    # LR mean L40
    li = LAYERS.index(40)
    tr, te = lr_eval(X_tr_mean[:, li, :], y_tr, X_te_mean[:, li, :], y_te)
    out["LR_mean_L40"] = {"train_auc": tr, "test_auc": te}
    print(f"  LR_mean_L40             train={tr:.4f}, test={te:.4f}", flush=True)

    # LR last-tok L45
    li = LAYERS.index(45)
    tr, te = lr_eval(X_tr_last[:, li, :], y_tr, X_te_last[:, li, :], y_te)
    out["LR_last_L45"] = {"train_auc": tr, "test_auc": te}
    print(f"  LR_last_L45             train={tr:.4f}, test={te:.4f}", flush=True)

    # LR multi-layer concat
    Xtr_concat = X_tr_mean.reshape(len(X_tr_mean), -1)
    Xte_concat = X_te_mean.reshape(len(X_te_mean), -1)
    tr, te = lr_eval(Xtr_concat, y_tr, Xte_concat, y_te, C=0.1)
    out["LR_multi_concat"] = {"train_auc": tr, "test_auc": te}
    print(f"  LR_multi_concat         train={tr:.4f}, test={te:.4f}", flush=True)

    # LR mean-of-layers
    Xtr_mol = X_tr_mean.mean(axis=1)
    Xte_mol = X_te_mean.mean(axis=1)
    tr, te = lr_eval(Xtr_mol, y_tr, Xte_mol, y_te)
    out["LR_mean_of_layers"] = {"train_auc": tr, "test_auc": te}
    print(f"  LR_mean_of_layers       train={tr:.4f}, test={te:.4f}", flush=True)

    # Pleshkov d=16 @ L40 mean
    t0 = time.time()
    li = LAYERS.index(40)
    p = QuadraticProbe(d_pca=16, alpha=10.0, random_state=0).fit(X_tr_mean[:, li, :], y_tr)
    auc_tr = roc_auc_score(y_tr, p.decision_function(X_tr_mean[:, li, :]))
    auc_te = roc_auc_score(y_te, p.decision_function(X_te_mean[:, li, :]))
    out["Pleshkov_d16_L40"] = {"train_auc": float(auc_tr), "test_auc": float(auc_te)}
    print(f"  Pleshkov_d16_L40        train={auc_tr:.4f}, test={auc_te:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # MLP probe @ L40 mean
    t0 = time.time()
    li = LAYERS.index(40)
    tr, te = train_torch_probe(MLPProbe, X_tr_mean[:, li, :], y_tr, X_te_mean[:, li, :], y_te,
                                epochs=50, lr=1e-3, batch=32, d_in=X_tr_mean.shape[2])
    out["MLP_L40"] = {"train_auc": tr, "test_auc": te}
    print(f"  MLP_L40                 train={tr:.4f}, test={te:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # Constitutional probe (multi-layer concat)
    t0 = time.time()
    tr, te = train_torch_probe(ConstitutionalProbe, X_tr_mean, y_tr, X_te_mean, y_te,
                                epochs=30, lr=5e-4, batch=16, l2=1e-3,
                                n_layers=13, d_per_layer=X_tr_mean.shape[2])
    out["Constitutional_concat"] = {"train_auc": tr, "test_auc": te}
    print(f"  Constitutional_concat   train={tr:.4f}, test={te:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    return out


def main():
    print("[honest train/test split eval]", flush=True)
    per_task = {}

    # Cyber 1/2/3
    for task in ["cyber_1", "cyber_2", "cyber_3"]:
        print(f"\nloading cyber {task} train+test features...", flush=True)
        t0 = time.time()
        X_tr_mean, X_tr_last, y_tr, _ = load_cyber_features(CYBER_TRAIN, task)
        print(f"  train: {X_tr_mean.shape} ({time.time()-t0:.0f}s)", flush=True)
        t0 = time.time()
        X_te_mean, X_te_last, y_te, _ = load_cyber_features(CYBER_TEST, task)
        print(f"  test:  {X_te_mean.shape} ({time.time()-t0:.0f}s)", flush=True)
        per_task[task] = eval_task(task, X_tr_mean, X_tr_last, y_tr, X_te_mean, X_te_last, y_te)

    # Refusal Gemma
    print(f"\nloading refusal_gemma train+test features...", flush=True)
    t0 = time.time()
    X_tr_mean, X_tr_last, y_tr, _ = load_refusal_features("train")
    print(f"  train: {X_tr_mean.shape} ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    X_te_mean, X_te_last, y_te, _ = load_refusal_features("test")
    print(f"  test:  {X_te_mean.shape} ({time.time()-t0:.0f}s)", flush=True)
    per_task["refusal_gemma"] = eval_task("refusal_gemma", X_tr_mean, X_tr_last, y_tr,
                                            X_te_mean, X_te_last, y_te)

    # Save per-task
    (HERE / "honest_per_task.json").write_text(json.dumps(per_task, indent=2))

    # Aggregate per probe
    probes = sorted(next(iter(per_task.values())).keys())
    summary = {}
    for probe in probes:
        aucs = [per_task[t][probe]["test_auc"] for t in per_task]
        summary[probe] = {
            "mean_test_auc": float(np.mean(aucs)),
            "std_test_auc": float(np.std(aucs, ddof=1)),
            "min_test_auc": float(np.min(aucs)),
            "max_test_auc": float(np.max(aucs)),
            "per_task": {t: per_task[t][probe]["test_auc"] for t in per_task},
        }
    (HERE / "honest_summary.json").write_text(json.dumps(summary, indent=2))

    # Pretty md table
    md = ["# HONEST evaluation: test-split AUC per probe per task\n"]
    md.append("Train on `split==\"train\"` rows only; eval on `split==\"test\"` rows.")
    md.append("Qwen excluded (model not downloaded). 4-task mean.\n")
    cols = ["cyber_1", "cyber_2", "cyber_3", "refusal_gemma"]
    md.append("| Probe | " + " | ".join(cols) + " | **mean** | std |")
    md.append("|---|" + "|".join(["---:" for _ in cols]) + "|---:|---:|")
    rows = []
    for probe in probes:
        row = [probe] + [f'{per_task[t][probe]["test_auc"]:.4f}' for t in cols]
        s = summary[probe]
        rows.append((s["mean_test_auc"], row + [f'**{s["mean_test_auc"]:.4f}**', f'{s["std_test_auc"]:.4f}']))
    rows.sort(reverse=True)
    for _, r in rows:
        md.append("| " + " | ".join(r) + " |")
    (HERE / "honest_table.md").write_text("\n".join(md) + "\n")

    print(f"\n=== HEADLINE: 4-task mean test AUC (sorted desc) ===")
    for mean_auc, r in rows:
        print(f"  {r[0]:>22}  {mean_auc:.4f}  std={r[-1]}")
    print(f"\nwrote {HERE/'honest_per_task.json'}, {HERE/'honest_summary.json'}, {HERE/'honest_table.md'}")


if __name__ == "__main__":
    main()
