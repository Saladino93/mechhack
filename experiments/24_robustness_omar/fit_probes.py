"""Fit every (layer, pooling) LR probe on the full 832-sample refusal training set,
plus multi-layer concat — and save the weights to one .npz so the
edit-scoring pass can apply each one without retraining.

Probes saved:
  - lr_mean_L{l}     : mean-pool at each layer in {0,5,...,60}
  - lr_last_L{l}     : last-token at each layer
  - lr_multi_concat  : concat of mean over all 13 layers (69,888-dim)
  - lr_mean_of_layers: single 5376-d "mean across layers" vector
  - lr_max_of_layers : single 5376-d "max across layers" vector

Output:
  experiments/24_robustness_omar/fitted_probes.npz
  experiments/24_robustness_omar/probe_specs.json   (probe metadata)
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

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)
REPO_ROOT = HERE.parent.parent
CACHE_MEAN = REPO_ROOT / "experiments" / "17_quadratic_probe_omar" / "cache" / "refusal_13layer_mean.npz"
REFUSAL_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b")
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def load_last_tok_features():
    """Returns X_last: (N, 13, d), y. Re-pools from .pt because it's
    not part of the cached mean-pool features."""
    cache_path = HERE / "refusal_13layer_last_tok.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return z["X"].astype(np.float32), z["y"].astype(np.int64)
    attrs = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            attrs[r["sample_id"]] = r
    Xs, ys = [], []
    pt_paths = sorted(REFUSAL_EXTRACTS.glob("*.pt"))
    print(f"  loading {len(pt_paths)} .pt files for last-token pool...", flush=True)
    t0 = time.time()
    for i, p in enumerate(pt_paths):
        sid = p.stem
        if sid not in attrs: continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
        residuals = ex["residuals"].float().clone()  # (n_layers, n_tok, d)
        mask = ex["attention_mask"].bool().squeeze().clone()
        last_idx = int(mask.nonzero().max().item())
        feat = residuals[:, last_idx, :].numpy().astype(np.float32)
        if not np.isfinite(feat).all(): continue
        Xs.append(feat); ys.append(int(ex["label"]))
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(pt_paths)} ({time.time()-t0:.1f}s)", flush=True)
    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    np.savez(cache_path, X=X, y=y)
    return X, y


def fit_lr(X, y, C=1.0):
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(X, y)
    return clf.coef_.squeeze().astype(np.float32), float(clf.intercept_[0])


def main():
    print("[fit probes for robustness]", flush=True)
    z = np.load(CACHE_MEAN, allow_pickle=True)
    X_mean = z["X"].astype(np.float32)  # (N, 13, d)
    y = z["y"].astype(np.int64)
    layer_idxs = list(z["layer_idxs"])
    n, n_layers, d = X_mean.shape
    print(f"  refusal mean-pool features: {X_mean.shape}", flush=True)

    X_last, y_last = load_last_tok_features()
    print(f"  refusal last-tok features:  {X_last.shape}", flush=True)
    assert (y == y_last).all(), "label mismatch between mean and last-tok caches"

    weights = {}
    biases = {}
    specs = {"layers": layer_idxs, "n_train_samples": int(n), "d_model": int(d), "probes": []}

    print("\n  fitting per-layer mean-pool LR...", flush=True)
    for li, lyr in enumerate(layer_idxs):
        name = f"lr_mean_L{lyr}"
        coef, bias = fit_lr(X_mean[:, li, :], y)
        weights[name] = coef; biases[name] = bias
        specs["probes"].append({"name": name, "kind": "lr_single_layer", "layer": lyr,
                                "pooling": "mean", "shape": list(coef.shape)})

    print("  fitting per-layer last-tok LR...", flush=True)
    for li, lyr in enumerate(layer_idxs):
        if lyr == 0: continue  # last-tok L0 is degenerate
        name = f"lr_last_L{lyr}"
        coef, bias = fit_lr(X_last[:, li, :], y)
        weights[name] = coef; biases[name] = bias
        specs["probes"].append({"name": name, "kind": "lr_single_layer", "layer": lyr,
                                "pooling": "last_token", "shape": list(coef.shape)})

    print("  fitting multi-layer mean concat LR (69,888d)...", flush=True)
    X_concat = X_mean.reshape(n, -1)  # (N, 13*5376)
    coef, bias = fit_lr(X_concat, y, C=0.1)  # smaller C to fight overfit
    weights["lr_multi_concat"] = coef; biases["lr_multi_concat"] = bias
    specs["probes"].append({"name": "lr_multi_concat", "kind": "lr_multi_layer",
                            "layers": layer_idxs, "pooling": "mean", "shape": list(coef.shape)})

    print("  fitting mean-of-layers LR (5376d)...", flush=True)
    coef, bias = fit_lr(X_mean.mean(axis=1), y)
    weights["lr_mean_of_layers"] = coef; biases["lr_mean_of_layers"] = bias
    specs["probes"].append({"name": "lr_mean_of_layers", "kind": "lr_aggregate",
                            "agg": "mean_over_layers", "shape": list(coef.shape)})

    print("  fitting max-of-layers LR (5376d)...", flush=True)
    coef, bias = fit_lr(X_mean.max(axis=1), y)
    weights["lr_max_of_layers"] = coef; biases["lr_max_of_layers"] = bias
    specs["probes"].append({"name": "lr_max_of_layers", "kind": "lr_aggregate",
                            "agg": "max_over_layers", "shape": list(coef.shape)})

    # Save weights
    out_path = HERE / "fitted_probes.npz"
    save_dict = {}
    for name in weights:
        save_dict[f"coef_{name}"] = weights[name].astype(np.float32)
        save_dict[f"bias_{name}"] = np.float32(biases[name])
    np.savez(out_path, **save_dict)
    (HERE / "probe_specs.json").write_text(json.dumps(specs, indent=2))
    print(f"\n  wrote {out_path} ({len(weights)} probes)", flush=True)
    print(f"  wrote {HERE/'probe_specs.json'}", flush=True)


if __name__ == "__main__":
    main()
