"""Minimal MultiMax — the simplest possible max-pool linear probe.

  per-token score:  s_t = w · x_t + b
  prompt logit:     z   = max_t s_t  (over masked tokens)
  loss:             BCE(z, label)

No transformation MLP. No multi-head. No attention. ~5377 trainable params
(d_model + 1 bias). One forward pass through PyTorch's Linear is the entire
predictor.

This isolates Kramár's core claim: **max-pool > mean-pool when the signal is
concentrated on one token**. If this beats LR-mean baseline at low cost, the
mechanism is "one critical token wins" not "every token contributes a bit".

Trained on refusal training-split only; evaluated on test split. We also
compare the same head with mean-pool to make the apples-to-apples test of
max vs mean at single-direction-w.
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

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)
REPO_ROOT = HERE.parent.parent

REFUSAL_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b")
REFUSAL_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"

LAYER = 30  # Match exp 16 Rolling probe's layer for direct comparison
TARGET_LAYER_IDX_IN_EXTRACT = 6  # in [0,5,10,...,60] L30 is index 6


class MinimalMultiMax(nn.Module):
    """One linear projection per token, max over tokens."""
    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        # x: (B, T, d), mask: (B, T)
        s = self.lin(x.float()).squeeze(-1)        # (B, T)
        s = s.masked_fill(~mask, float("-inf"))
        return s.max(dim=1).values                  # (B,)


class MinimalMean(nn.Module):
    """Same architecture but mean-pool instead of max — for ablation."""
    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        s = self.lin(x.float()).squeeze(-1)
        s = s.masked_fill(~mask, 0.0)
        return s.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)


def load_split(target_split):
    """Load (per-token L30 fp32 residuals, mask, label) for refusal split."""
    rows = []
    with REFUSAL_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != target_split: continue
            if r.get("is_refusal") is None: continue
            rows.append((r["sample_id"], int(bool(r["is_refusal"]))))
    samples = []
    for sid, lbl in rows:
        p = REFUSAL_EXTRACTS / f"{sid}.pt"
        if not p.exists(): continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
        residuals = ex["residuals"][TARGET_LAYER_IDX_IN_EXTRACT].float()  # (n_tok, d)
        mask = ex["attention_mask"].bool().squeeze()
        n = int(mask.sum().item())
        if n < 2: continue
        if not torch.isfinite(residuals).all(): continue
        samples.append({"sid": sid, "label": lbl, "residuals": residuals.half(),
                         "mask": mask, "n_tokens": n})
    return samples


def collate(batch, max_t=4096):
    """Pad to longest in batch, capped at max_t (truncate from front to keep tail)."""
    Ts = [min(s["residuals"].shape[0], max_t) for s in batch]
    T = max(Ts)
    B = len(batch); d = batch[0]["residuals"].shape[1]
    X = torch.zeros(B, T, d, dtype=torch.float32)
    M = torch.zeros(B, T, dtype=torch.bool)
    Y = torch.zeros(B, dtype=torch.float32)
    for i, s in enumerate(batch):
        n = Ts[i]
        # Tail-keep — refusal signal is at the end (assistant turn)
        X[i, :n] = s["residuals"][-n:].float()
        M[i, :n] = s["mask"][-n:]
        Y[i] = s["label"]
    return X, M, Y


def train_one(probe, train_samples, val_samples, epochs=20, lr=1e-3, batch=8, l2=1e-4):
    from sklearn.metrics import roc_auc_score
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=l2)
    rng = np.random.default_rng(0)
    losses = []
    n = len(train_samples)
    for ep in range(epochs):
        probe.train()
        order = rng.permutation(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            idx = order[i:i+batch]
            bs = [train_samples[j] for j in idx]
            X, M, Y = collate(bs)
            logits = probe(X, M)
            loss = F.binary_cross_entropy_with_logits(logits, Y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
        losses.append(ep_loss / n)
    # eval
    probe.eval()
    def score_split(samples):
        scores = []; labels = []
        for s in samples:
            X, M, Y = collate([s])
            with torch.no_grad():
                z = probe(X, M).item()
            scores.append(z); labels.append(s["label"])
        return roc_auc_score(labels, scores), np.asarray(scores), np.asarray(labels)
    train_auc, _, _ = score_split(train_samples)
    val_auc, val_scores, val_labels = score_split(val_samples)
    return train_auc, val_auc, val_scores, val_labels, losses


def main():
    print(f"[Minimal MultiMax] L{LAYER}, train on train-split, eval on test-split", flush=True)
    print("\nloading refusal train+test...", flush=True)
    t0 = time.time()
    train_samples = load_split("train")
    test_samples = load_split("test")
    print(f"  train={len(train_samples)}, test={len(test_samples)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  train pos/neg = {sum(s['label'] for s in train_samples)}/{sum(1-s['label'] for s in train_samples)}", flush=True)
    print(f"  test  pos/neg = {sum(s['label'] for s in test_samples)}/{sum(1-s['label'] for s in test_samples)}", flush=True)
    d = train_samples[0]["residuals"].shape[1]

    print(f"\n=== MAX-pool minimal probe ===", flush=True)
    t0 = time.time()
    probe_max = MinimalMultiMax(d_model=d)
    n_params = sum(p.numel() for p in probe_max.parameters())
    print(f"  n_params = {n_params}", flush=True)
    tr_auc_max, te_auc_max, sc_max, lab_max, losses_max = train_one(probe_max, train_samples, test_samples,
                                                                    epochs=20, lr=1e-3, batch=8)
    print(f"  TRAIN AUC = {tr_auc_max:.4f}", flush=True)
    print(f"  TEST  AUC = {te_auc_max:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== MEAN-pool minimal probe (ablation: same arch, mean instead of max) ===", flush=True)
    t0 = time.time()
    probe_mean = MinimalMean(d_model=d)
    tr_auc_mean, te_auc_mean, sc_mean, lab_mean, losses_mean = train_one(probe_mean, train_samples, test_samples,
                                                                          epochs=20, lr=1e-3, batch=8)
    print(f"  TRAIN AUC = {tr_auc_mean:.4f}", flush=True)
    print(f"  TEST  AUC = {te_auc_mean:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  Δ (max − mean) = {te_auc_max - te_auc_mean:+.4f}", flush=True)

    # Save
    out = {
        "layer": LAYER, "d_model": int(d), "n_params": int(n_params),
        "n_train": len(train_samples), "n_test": len(test_samples),
        "max_pool": {"train_auc": float(tr_auc_max), "test_auc": float(te_auc_max),
                     "final_loss": float(losses_max[-1])},
        "mean_pool": {"train_auc": float(tr_auc_mean), "test_auc": float(te_auc_mean),
                      "final_loss": float(losses_mean[-1])},
        "delta_max_minus_mean": float(te_auc_max - te_auc_mean),
    }
    (HERE / "minimal_multimax_results.json").write_text(json.dumps(out, indent=2))

    torch.save(probe_max.state_dict(), str(HERE / "weights_max.pt"))
    torch.save(probe_mean.state_dict(), str(HERE / "weights_mean.pt"))
    print(f"\nwrote {HERE/'minimal_multimax_results.json'}", flush=True)


if __name__ == "__main__":
    main()
