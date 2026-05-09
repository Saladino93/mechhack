"""Pleshkov 2026 quadratic probe — 13-layer sweep.

Loads (N, 13, d_model) features once per task, then runs:
  - linear LR on raw activations (5376-d)
  - linear LR on d_pca PCs (apples-to-apples baseline)
  - quadratic probe (PCA d_pca + degree-2 + ridge)

at each of 13 layers in {0,5,10,...,60} and writes a single sweep JSON +
auc_vs_layer plot per task.

Why this exists: the original train.py was per-layer-per-task. We hadn't
plotted quadratic-probe AUC across the layer axis. Plenty of papers
report quadratic-vs-linear at one layer; the across-layer profile is
the more interesting view.

Usage:
    python sweep_layers.py --task refusal_gemma --d_pca 16
    python sweep_layers.py --task cyber_1     --d_pca 16
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from probes import QuadraticProbe, n_quadratic_features  # noqa: E402
from train import (  # noqa: E402
    cv_linear, cv_linear_on_pcs, cv_quadratic, wilson_ci_auc,
    SELECTIONS, ALPHA_GRID, SEED, N_FOLDS,
)

REFUSAL_13LAYER = Path("/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b")
CYBER_ALL = Path("/home/ubuntu/extracts/cyber_all_omar")
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
RESULTS = HERE / "results"
CACHE = HERE / "cache"
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def load_refusal_13layer():
    """Returns X_mean: (N, 13, d), y, layer_idxs."""
    cache = CACHE / "refusal_13layer_mean.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"].astype(np.float32), z["y"].astype(np.int64), list(z["layer_idxs"])
    attrs = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            attrs[r["sample_id"]] = r
    means, ys = [], []
    layer_idxs = None
    pt_paths = sorted(REFUSAL_13LAYER.glob("*.pt"))
    print(f"loading {len(pt_paths)} refusal 13-layer extracts...", flush=True)
    t0 = time.time()
    for i, p in enumerate(pt_paths):
        sid = p.stem
        if sid not in attrs: continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
        if layer_idxs is None:
            layer_idxs = list(ex["layer_idxs"])
        residuals = ex["residuals"].float().clone()  # (n_layers, n_tok, d)
        mask = ex["attention_mask"].bool().squeeze().clone()
        n = int(mask.sum().item())
        if n < 2: continue
        m = mask.float().unsqueeze(0).unsqueeze(-1)
        feat = ((residuals * m).sum(dim=1) / n).numpy().astype(np.float32)
        if not np.isfinite(feat).all(): continue
        means.append(feat)
        ys.append(int(ex["label"]))
        if (i + 1) % 200 == 0:
            print(f"  loaded {i+1}/{len(pt_paths)} in {time.time()-t0:.1f}s", flush=True)
    X = np.stack(means).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    np.savez(cache, X=X, y=y, layer_idxs=np.array(layer_idxs, dtype=np.int64))
    return X, y, layer_idxs


def load_cyber_13layer(task: str):
    """Returns X_mean: (N, 13, d), y, layer_idxs."""
    cache = CACHE / f"{task}_13layer_mean.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"].astype(np.float32), z["y"].astype(np.int64), list(z["layer_idxs"])
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
    from data import get_label_for_task, load_dataset  # type: ignore
    sel = json.loads(SELECTIONS[task].read_text())
    selected_ids = [row["sample_id"] for row in sel["samples"]]
    samples_by_id = {s["sample_id"]: s for s in load_dataset("cyber", split="train")}
    means, ys = [], []
    layer_idxs = None
    print(f"loading cyber {task} 13-layer extracts...", flush=True)
    t0 = time.time()
    for i, sid in enumerate(selected_ids):
        s = samples_by_id.get(sid)
        if s is None: continue
        lbl = get_label_for_task(s, task)
        if lbl is None: continue
        p = CYBER_ALL / f"{sid}.pt"
        if not p.exists(): continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
        if layer_idxs is None:
            layer_idxs = list(ex["layer_idxs"])
        residuals = ex["residuals"].float().clone()
        mask = ex["attention_mask"].bool().squeeze().clone()
        n = int(mask.sum().item())
        if n < 2: continue
        m = mask.float().unsqueeze(0).unsqueeze(-1)
        feat = ((residuals * m).sum(dim=1) / n).numpy().astype(np.float32)
        if not np.isfinite(feat).all(): continue
        means.append(feat); ys.append(int(lbl))
        if (i + 1) % 200 == 0:
            print(f"  loaded {i+1}/{len(selected_ids)} in {time.time()-t0:.1f}s", flush=True)
    X = np.stack(means).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    np.savez(cache, X=X, y=y, layer_idxs=np.array(layer_idxs, dtype=np.int64))
    return X, y, layer_idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["refusal_gemma", "cyber_1", "cyber_2", "cyber_3"])
    ap.add_argument("--d_pca", type=int, required=True, choices=[16, 32])
    args = ap.parse_args()

    if args.task == "refusal_gemma":
        X_all, y, layer_idxs = load_refusal_13layer()
    else:
        X_all, y, layer_idxs = load_cyber_13layer(args.task)
    n, n_layers, d_model = X_all.shape
    print(f"  X_all shape: {X_all.shape}  layer_idxs={layer_idxs}", flush=True)
    print(f"  n_pos={int((y==1).sum())} n_neg={int((y==0).sum())}", flush=True)

    out = {
        "task": args.task,
        "d_pca": args.d_pca,
        "n_quadratic_features": n_quadratic_features(args.d_pca),
        "n_samples": int(n),
        "d_model": int(d_model),
        "layer_idxs": layer_idxs,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "alpha_grid": ALPHA_GRID,
        "by_layer": {},
    }

    out_path = RESULTS / f"{args.task}_d{args.d_pca}_sweep.json"
    print(f"  output -> {out_path}", flush=True)

    t_start = time.time()
    for li, lyr in enumerate(layer_idxs):
        X = X_all[:, li, :].copy()  # (N, d_model)
        print(f"\n  === L{lyr} ===", flush=True)
        t0 = time.time()
        lin = cv_linear(X, y)
        print(f"    linear:  AUC={lin['auc_mean']:.4f} ± {lin['auc_std']:.4f}", flush=True)
        lin_pcs = cv_linear_on_pcs(X, y, d_pca=args.d_pca)
        print(f"    lin-PCs: AUC={lin_pcs['auc_mean']:.4f} ± {lin_pcs['auc_std']:.4f}", flush=True)
        quad = cv_quadratic(X, y, d_pca=args.d_pca)
        print(f"    quad:    AUC={quad['auc_mean']:.4f} ± {quad['auc_std']:.4f}", flush=True)
        out["by_layer"][str(lyr)] = {
            "linear":         {k: v for k, v in lin.items()     if k != "fold_metrics"},
            "linear_on_pcs":  {k: v for k, v in lin_pcs.items() if k != "fold_metrics"},
            "quadratic":      {k: v for k, v in quad.items()    if k != "fold_metrics"},
            "delta_quad_vs_linear":     float(quad["auc_mean"] - lin["auc_mean"]),
            "delta_quad_vs_lin_on_pcs": float(quad["auc_mean"] - lin_pcs["auc_mean"]),
            "elapsed_s": round(time.time() - t0, 1),
        }
        # Incremental save so we can plot before sweep finishes.
        out_path.write_text(json.dumps(out, indent=2))
        gc.collect()

    out["wall_seconds"] = round(time.time() - t_start, 1)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nfinished. wall={out['wall_seconds']}s. wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
