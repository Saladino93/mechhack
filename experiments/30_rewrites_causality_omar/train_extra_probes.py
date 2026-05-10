"""Train MLP probe + Constitutional classifier on refusal training set.

Both use cached refusal mean-pool features at L40 (single layer) — keeping
this simple and fast.

MLP probe:
  L40 mean (5376) → Dense(256) → ReLU → Dropout(0.2) → Dense(64) → ReLU →
  Dense(1) sigmoid. Adam lr=1e-3, BCE, 50 epochs, batch 32. CPU.

Constitutional classifier:
  Multi-layer concat (13 × 5376 = 69,888) → Dense(128) → ReLU → Dense(1)
  sigmoid. Same training. Mirrors Anthropic's "Constitutional Classifier"
  philosophy: a small head on top of multi-layer activations, calibrated
  via probability output.

Output:
  fitted_extra_probes.npz — weights for both probes (state_dict in npz form)
  extra_probes_summary.json — train/val AUC per probe
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
CACHE = REPO_ROOT / "experiments" / "17_quadratic_probe_omar" / "cache" / "refusal_13layer_mean.npz"
OUT = HERE / "fitted_extra_probes.pt"
SUMMARY = HERE / "extra_probes_summary.json"

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
L40_IDX = LAYERS.index(40)


class MLPProbe(nn.Module):
    def __init__(self, d_in=5376, d_hidden=256, d_hidden2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, d_hidden2),
            nn.ReLU(),
            nn.Linear(d_hidden2, 1),
        )

    def forward(self, x):
        return self.net(x.float()).squeeze(-1)


class ConstitutionalProbe(nn.Module):
    """Multi-layer concat → small MLP. Anthropic-style "constitutional
    classifier" without the constitutional-AI training data — just the head
    architecture: a calibrated thin MLP over a wide feature stack."""
    def __init__(self, n_layers=13, d_per_layer=5376, d_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_layers * d_per_layer, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        # x: (B, n_layers, d) → flatten
        b = x.shape[0]
        return self.net(x.float().reshape(b, -1)).squeeze(-1)


def train_probe(probe, X_tr, y_tr, X_te, y_te, epochs=30, lr=1e-3, batch=32, l2=1e-4):
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=l2)
    n = len(X_tr)
    rng = np.random.default_rng(0)
    losses = []
    for ep in range(epochs):
        probe.train()
        order = rng.permutation(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            idx = order[i:i+batch]
            x = X_tr[idx]
            y = y_tr[idx].float()
            logits = probe(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
        losses.append(ep_loss / n)
    # Eval AUC
    from sklearn.metrics import roc_auc_score
    probe.eval()
    with torch.no_grad():
        s_tr = probe(X_tr).numpy()
        s_te = probe(X_te).numpy()
    return float(roc_auc_score(y_tr.numpy(), s_tr)), float(roc_auc_score(y_te.numpy(), s_te)), losses


def main():
    print("[train extra probes]", flush=True)
    z = np.load(CACHE, allow_pickle=True)
    X_all = z["X"].astype(np.float32)  # (N, 13, d)
    y = z["y"].astype(np.int64)
    n = len(y)
    print(f"  refusal training data: {X_all.shape}, n_pos={int((y==1).sum())}, n_neg={int((y==0).sum())}",
          flush=True)

    # Train/val split — 80/20
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_tr = int(0.8 * n)
    tr_idx = perm[:n_tr]; te_idx = perm[n_tr:]

    # MLP on L40 mean
    print("\n  training MLP probe (L40 mean)...", flush=True)
    X_l40 = X_all[:, L40_IDX, :]
    X_tr_t = torch.tensor(X_l40[tr_idx], dtype=torch.float32)
    X_te_t = torch.tensor(X_l40[te_idx], dtype=torch.float32)
    y_tr_t = torch.tensor(y[tr_idx], dtype=torch.float32)
    y_te_t = torch.tensor(y[te_idx], dtype=torch.float32)
    mlp = MLPProbe(d_in=X_l40.shape[1])
    t0 = time.time()
    mlp_train_auc, mlp_test_auc, mlp_losses = train_probe(
        mlp, X_tr_t, y_tr_t, X_te_t, y_te_t, epochs=50, lr=1e-3, batch=32)
    print(f"    MLP (L40 mean): train AUC={mlp_train_auc:.4f}, val AUC={mlp_test_auc:.4f}, "
          f"final loss={mlp_losses[-1]:.4f} ({time.time()-t0:.1f}s)", flush=True)

    # Refit on full data for downstream scoring
    mlp_full = MLPProbe(d_in=X_l40.shape[1])
    X_full_t = torch.tensor(X_l40, dtype=torch.float32)
    y_full_t = torch.tensor(y, dtype=torch.float32)
    train_probe(mlp_full, X_full_t, y_full_t, X_full_t, y_full_t, epochs=50, lr=1e-3, batch=32)

    # Constitutional on multi-layer concat
    print("\n  training Constitutional probe (13-layer mean concat)...", flush=True)
    X_concat = X_all  # (N, 13, d)
    X_tr_t = torch.tensor(X_concat[tr_idx], dtype=torch.float32)
    X_te_t = torch.tensor(X_concat[te_idx], dtype=torch.float32)
    y_tr_t = torch.tensor(y[tr_idx], dtype=torch.float32)
    y_te_t = torch.tensor(y[te_idx], dtype=torch.float32)
    const = ConstitutionalProbe(n_layers=13, d_per_layer=X_concat.shape[2])
    t0 = time.time()
    const_train_auc, const_test_auc, const_losses = train_probe(
        const, X_tr_t, y_tr_t, X_te_t, y_te_t, epochs=30, lr=5e-4, batch=16, l2=1e-3)
    print(f"    Constitutional (concat): train AUC={const_train_auc:.4f}, val AUC={const_test_auc:.4f}, "
          f"final loss={const_losses[-1]:.4f} ({time.time()-t0:.1f}s)", flush=True)

    const_full = ConstitutionalProbe(n_layers=13, d_per_layer=X_concat.shape[2])
    X_full_t = torch.tensor(X_concat, dtype=torch.float32)
    y_full_t = torch.tensor(y, dtype=torch.float32)
    train_probe(const_full, X_full_t, y_full_t, X_full_t, y_full_t, epochs=30, lr=5e-4, batch=16, l2=1e-3)

    torch.save({
        "mlp_state": mlp_full.state_dict(),
        "constitutional_state": const_full.state_dict(),
        "d_model": X_l40.shape[1],
        "n_layers": 13,
    }, str(OUT))

    SUMMARY.write_text(json.dumps({
        "MLP_L40": {"train_auc": mlp_train_auc, "val_auc": mlp_test_auc,
                    "final_loss": mlp_losses[-1], "epochs": 50},
        "Constitutional_concat": {"train_auc": const_train_auc, "val_auc": const_test_auc,
                                   "final_loss": const_losses[-1], "epochs": 30},
        "n_train": int(n_tr), "n_val": int(n - n_tr),
    }, indent=2))
    print(f"\nwrote {OUT} and {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
