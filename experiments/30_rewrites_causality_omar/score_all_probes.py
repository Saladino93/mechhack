"""Score 648 (=81 originals + 81×7 rewrites) prompts under EVERY probe.

Reads:
  features_rewrites.npz       (mean + last across 13 layers)
  L30_pertoken/<sid>_r<i>.pt  (per-token L30 — for Kramar probes)
  fitted_probes.npz           (28 LR variants — exp 24)
  fitted_extra_probes.pt       (MLP + Constitutional — q23)
  refusal_gemma4_31b_*.pt     (Kramar multimax/rolling/rolling_multimax — exp 16)
  17_quadratic_probe_omar/cache/refusal_13layer_mean.npz  (for re-fitting Pleshkov)

Output:
  rewrites_scored.jsonl: one row per (sample_id, rewrite_idx) with score per probe.
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

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

FEATURES = HERE / "features_rewrites.npz"
PT_DIR = HERE / "L30_pertoken"
LR_PROBES = REPO_ROOT / "experiments" / "24_robustness_omar" / "fitted_probes.npz"
LR_SPECS = REPO_ROOT / "experiments" / "24_robustness_omar" / "probe_specs.json"
EXTRA_PROBES = HERE / "fitted_extra_probes.pt"
REFUSAL_TRAIN_CACHE = REPO_ROOT / "experiments" / "17_quadratic_probe_omar" / "cache" / "refusal_13layer_mean.npz"
KRAMAR_DIR = REPO_ROOT / "experiments" / "16_multimax_probe_omar" / "results"
OUT = HERE / "rewrites_scored.jsonl"

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


# ---- probe classes (mirror experiments/16's probes.py) ----
sys.path.insert(0, str(REPO_ROOT / "experiments" / "16_multimax_probe_omar"))
from probes import (
    AttentionProbe, MultiMaxProbe, RollingAttentionProbe, RollingMultiMaxProbe,
    TransformMLP,
)
sys.path.insert(0, str(REPO_ROOT / "experiments" / "17_quadratic_probe_omar"))
from probes import QuadraticProbe
sys.path.insert(0, str(HERE))
from train_extra_probes import MLPProbe, ConstitutionalProbe  # noqa: E402


def load_lr_probes():
    z = np.load(LR_PROBES, allow_pickle=True)
    specs = json.loads(LR_SPECS.read_text())
    probes = {}
    for s in specs["probes"]:
        n = s["name"]
        probes[n] = {"spec": s,
                     "coef": z[f"coef_{n}"].astype(np.float32),
                     "bias": float(z[f"bias_{n}"])}
    return probes


def score_lr(probes, mean_per_layer, last_per_layer):
    flat_concat = mean_per_layer.reshape(-1)
    mean_of_layers = mean_per_layer.mean(axis=0)
    max_of_layers = mean_per_layer.max(axis=0)
    out = {}
    for name, p in probes.items():
        spec = p["spec"]; coef = p["coef"]; bias = p["bias"]
        if spec["kind"] == "lr_single_layer":
            li = LAYERS.index(spec["layer"])
            feat = mean_per_layer[li] if spec["pooling"] == "mean" else last_per_layer[li]
        elif spec["kind"] == "lr_multi_layer":
            feat = flat_concat
        elif spec["kind"] == "lr_aggregate":
            feat = mean_of_layers if spec["agg"] == "mean_over_layers" else max_of_layers
        else: continue
        if feat.shape != coef.shape:
            out[name] = float("nan"); continue
        logit = float(feat @ coef + bias)
        out[name] = float(1.0 / (1.0 + np.exp(-logit)))
    return out


def fit_pleshkov():
    z = np.load(REFUSAL_TRAIN_CACHE, allow_pickle=True)
    X = z["X"][:, LAYERS.index(40), :].astype(np.float32)
    y = z["y"].astype(np.int64)
    p = QuadraticProbe(d_pca=16, alpha=10.0, random_state=0).fit(X, y)
    return p


def load_kramar(variant: str):
    """variants: rolling, multimax, rolling_multimax. Loads weights + builds probe."""
    pt = torch.load(str(KRAMAR_DIR / f"refusal_gemma4_31b_{variant}.pt"),
                    weights_only=False, map_location="cpu")
    sd = pt["state_dict"]; hp = pt["hyperparameters"]; d = pt["d_model"]
    if variant == "rolling":
        probe = RollingAttentionProbe(d_model=d, d_hidden=hp["d_hidden"],
                                       n_heads=hp["n_heads"], window_size=hp["window_size"])
    elif variant == "multimax":
        probe = MultiMaxProbe(d_model=d, d_hidden=hp["d_hidden"],
                               n_heads=hp["n_heads"])
    elif variant == "rolling_multimax":
        probe = RollingMultiMaxProbe(d_model=d, d_hidden=hp["d_hidden"],
                                      n_heads=hp["n_heads"], window_size=hp["window_size"])
    else:
        raise ValueError(variant)
    probe.load_state_dict(sd)
    probe.eval()
    return probe


def main():
    print("[score all probes on rewrites_k7]", flush=True)
    z = np.load(FEATURES, allow_pickle=True)
    sids = list(z["sample_ids"])
    ridxs = list(z["rewrite_idxs"])
    means = z["mean"]   # (N, 13, d)
    lasts = z["last"]   # (N, 13, d)
    N = len(sids)
    print(f"  loaded features for {N} prompts", flush=True)

    print("  loading LR probes (28)...", flush=True)
    lr_probes = load_lr_probes()

    print("  fitting Pleshkov on full 832 refusal train @ L40...", flush=True)
    pleshkov = fit_pleshkov()

    print("  loading Kramar probes (multimax, rolling, rolling_multimax)...", flush=True)
    kramar_probes = {}
    for v in ["rolling", "multimax", "rolling_multimax"]:
        try:
            kramar_probes[v] = load_kramar(v)
            print(f"    loaded {v}", flush=True)
        except Exception as e:
            print(f"    [skip] {v}: {e}", flush=True)

    print("  loading extra probes (MLP, Constitutional)...", flush=True)
    extra = torch.load(str(EXTRA_PROBES), weights_only=False, map_location="cpu")
    mlp = MLPProbe(d_in=extra["d_model"]); mlp.load_state_dict(extra["mlp_state"]); mlp.eval()
    const = ConstitutionalProbe(n_layers=extra["n_layers"], d_per_layer=extra["d_model"])
    const.load_state_dict(extra["constitutional_state"]); const.eval()

    print("\n  scoring...", flush=True)
    t0 = time.time()
    with OUT.open("w") as fout:
        for i in range(N):
            sid = sids[i]; ridx = int(ridxs[i])
            mean_pl = means[i]  # (13, d)
            last_pl = lasts[i]  # (13, d)
            scores = score_lr(lr_probes, mean_pl, last_pl)
            scores["pleshkov_d16_L40"] = float(pleshkov.decision_function(mean_pl[LAYERS.index(40)][None, :])[0])
            with torch.no_grad():
                scores["mlp_L40"] = float(torch.sigmoid(mlp(torch.tensor(mean_pl[LAYERS.index(40)][None, :], dtype=torch.float32))).item())
                scores["constitutional_concat"] = float(torch.sigmoid(const(torch.tensor(mean_pl[None, :, :], dtype=torch.float32))).item())
            # Kramar probes — need per-token at L30
            pt_path = PT_DIR / f"{sid}_r{ridx}.pt"
            if pt_path.exists() and kramar_probes:
                pt = torch.load(str(pt_path), weights_only=False, map_location="cpu")
                r30 = pt["residuals"].float().unsqueeze(0)  # (1, n_tok, d)
                mask = pt["mask"].unsqueeze(0)
                with torch.no_grad():
                    for v, probe in kramar_probes.items():
                        try:
                            logit = probe(r30, mask).item()
                            scores[f"kramar_{v}_L30"] = float(1.0 / (1.0 + np.exp(-logit)))
                        except Exception as e:
                            scores[f"kramar_{v}_L30"] = float("nan")
            fout.write(json.dumps({"sample_id": sid, "rewrite_idx": ridx, "scores": scores}) + "\n")
            if (i + 1) % 100 == 0 or (i + 1) == N:
                print(f"    [{i+1}/{N}] {time.time()-t0:.0f}s", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
