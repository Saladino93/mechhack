"""Train Gemma refusal probes (linear/MLP/arditi/cc_concat/sklearn) — same protocol
as Qwen exp 31. Per-token heads handled separately if/when GPU frees.

Outputs to gemma_refusal_results.json + auc_by_layer plot scripted later.
"""
import os, sys, json, time, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_IDXS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
SEEDS = [0, 1, 2, 3, 4]

OUT_DIR = Path(__file__).parent
EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal")
SAMPLES_FILE = (OUT_DIR.parent.parent /
                "datasets/refusal_probes/gemma4_31b/attacks_full.jsonl")


# ---------- probe modules ----------
class LinearProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, d, h=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def auc_with_ci(y_true, y_score, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true); aucs = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, size=n)
        if len(set(y_true[ix].tolist())) < 2: continue
        aucs.append(roc_auc_score(y_true[ix], y_score[ix]))
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ---------- data ----------
def load_split(samples_file):
    rows = []
    for line in open(samples_file):
        s = json.loads(line)
        if s.get("is_refusal") is None: continue
        rows.append((s["sample_id"], int(bool(s["is_refusal"])), s.get("split")))
    return rows


def cache_pool_features(extracts_dir, sample_ids, cache_path):
    if cache_path.exists():
        z = np.load(cache_path)
        return z["mean"], z["last"], list(z["sample_ids"])
    print(f"  Caching pool features to {cache_path}")
    mean_feats, last_feats, ok_ids = [], [], []
    skipped = 0
    t0 = time.time()
    for i, sid in enumerate(sample_ids):
        p = extracts_dir / f"{sid}.pt"
        if not p.exists():
            skipped += 1; continue
        ex = torch.load(str(p), weights_only=False, map_location="cpu")
        r = ex["residuals"].float()
        m = ex["attention_mask"].bool()
        if not m.any(): skipped += 1; continue
        last_idx = int(m.nonzero().max().item())
        m_f = m.float().unsqueeze(0).unsqueeze(-1)
        mean = (r * m_f).sum(1) / m_f.sum(1).clamp(min=1)
        last = r[:, last_idx, :]
        mean_feats.append(mean.numpy().astype(np.float32))
        last_feats.append(last.numpy().astype(np.float32))
        ok_ids.append(sid)
        if (i+1) % 100 == 0:
            print(f"    [{i+1}/{len(sample_ids)}] cached, {time.time()-t0:.1f}s")
    mean_arr = np.stack(mean_feats, axis=0)
    last_arr = np.stack(last_feats, axis=0)
    np.savez_compressed(cache_path, mean=mean_arr, last=last_arr,
                         sample_ids=np.array(ok_ids))
    print(f"  cache: mean {mean_arr.shape} last {last_arr.shape}  "
          f"skipped={skipped}  in {time.time()-t0:.1f}s")
    return mean_arr, last_arr, ok_ids


# ---------- train ----------
def train_pooled(arch, X_train, y_train, X_val, y_val, d, seed,
                  epochs=50, batch=32):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = (LinearProbe(d) if arch == "linear" else MLPProbe(d)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(X_val,   dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_val,   dtype=torch.float32, device=DEVICE)
    n = Xt.shape[0]
    best_loss, best_state, patience = float("inf"), None, 5
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(n)
        for st in range(0, n, batch):
            bi = idx[st:st+batch]
            logits = model(Xt[bi])
            loss = bce(logits, yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = bce(model(Xv), yv).item()
        if vl < best_loss - 1e-4:
            best_loss = vl; patience = 5
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0: break
    if best_state is not None: model.load_state_dict(best_state)
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(Xt)).cpu().numpy()


# ---------- main ----------
rows = load_split(SAMPLES_FILE)
print(f"Loaded {len(rows)} labeled samples")
sample_ids_all = [r[0] for r in rows]
label_all = {r[0]: r[1] for r in rows}
split_all = {r[0]: r[2] for r in rows}

cache_path = OUT_DIR / "pool_cache_gemma_refusal.npz"
mean_arr, last_arr, ok_ids = cache_pool_features(EXTRACTS, sample_ids_all, cache_path)

y = np.array([label_all[sid] for sid in ok_ids], dtype=np.int64)
splits = np.array([split_all[sid] for sid in ok_ids])
train_idx = np.where(splits == "train")[0]
test_idx  = np.where(splits == "test")[0]
print(f"  train={len(train_idx)}  test={len(test_idx)}  pos_rate={y.mean():.3f}")

d = mean_arr.shape[-1]
rng = np.random.default_rng(0)
perm = rng.permutation(len(train_idx))
n_val = max(int(len(train_idx) * 0.10), 20)
inner_val_idx = train_idx[perm[:n_val]]
inner_train_idx = train_idx[perm[n_val:]]

results = {"layers": LAYER_IDXS, "n_train": int(len(train_idx)),
           "n_test": int(len(test_idx)), "by_arch": {}}

# --- per-layer pooled heads ---
for arch in ["linear_mean", "linear_last", "mlp_mean"]:
    results["by_arch"][arch] = {"per_layer": {}}
    for L in LAYER_IDXS:
        li = LAYER_IDXS.index(L)
        X = (mean_arr if arch != "linear_last" else last_arr)[:, li, :]
        seed_pred = []
        for seed in SEEDS:
            model = train_pooled(
                "linear" if arch != "mlp_mean" else "mlp",
                X[inner_train_idx], y[inner_train_idx].astype(np.float32),
                X[inner_val_idx],   y[inner_val_idx].astype(np.float32),
                d, seed)
            seed_pred.append(predict(model, X[test_idx]))
        mean_proba = np.mean(np.stack(seed_pred, axis=0), axis=0)
        auc, lo, hi = auc_with_ci(y[test_idx], mean_proba, seed=L)
        results["by_arch"][arch]["per_layer"][str(L)] = {
            "auc_seedavg": auc, "ci95": [lo, hi]}
        print(f"  {arch:12s} L{L:02d}: AUC {auc:.4f} [{lo:.4f},{hi:.4f}]")
    json.dump(results, open(OUT_DIR / "gemma_refusal_results.json", "w"), indent=2)

# --- arditi (no training) ---
results["by_arch"]["arditi"] = {"per_layer": {}}
for L in LAYER_IDXS:
    li = LAYER_IDXS.index(L)
    X = mean_arr[train_idx, li, :]
    mu_pos = X[y[train_idx] == 1].mean(axis=0)
    mu_neg = X[y[train_idx] == 0].mean(axis=0)
    d_dir = mu_pos - mu_neg
    d_dir = d_dir / (np.linalg.norm(d_dir) + 1e-8)
    scores = mean_arr[test_idx, li, :] @ d_dir
    auc, lo, hi = auc_with_ci(y[test_idx], scores, seed=L)
    results["by_arch"]["arditi"]["per_layer"][str(L)] = {
        "auc": auc, "ci95": [lo, hi]}
    print(f"  arditi      L{L:02d}: AUC {auc:.4f} [{lo:.4f},{hi:.4f}]")

# --- cc_concat (sklearn LR with C-sweep) ---
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
Xc = mean_arr.reshape(mean_arr.shape[0], -1)
Xc_train, Xc_test = Xc[train_idx], Xc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
Xc_inner_train, Xc_inner_val, y_inner_train, y_inner_val = train_test_split(
    Xc_train, y_train, test_size=0.2, stratify=y_train, random_state=0)
Cs = [0.01, 0.1, 1.0, 10.0]
inner = {C: float(roc_auc_score(y_inner_val,
            LogisticRegression(solver="lbfgs", max_iter=2000, C=C
                                ).fit(Xc_inner_train, y_inner_train
                                ).predict_proba(Xc_inner_val)[:, 1]))
          for C in Cs}
bestC = max(Cs, key=lambda c: inner[c])
lr = LogisticRegression(solver="lbfgs", max_iter=2000, C=bestC)
lr.fit(Xc_train, y_train)
proba = lr.predict_proba(Xc_test)[:, 1]
auc, lo, hi = auc_with_ci(y_test, proba)
results["by_arch"]["cc_concat"] = {"best_C": bestC, "test_auc": auc,
                                      "ci95": [lo, hi],
                                      "feature_dim": int(Xc.shape[1]),
                                      "C_inner_val_aucs": inner}
print(f"  cc_concat (C={bestC}): AUC {auc:.4f} [{lo:.4f},{hi:.4f}]")
json.dump(results, open(OUT_DIR / "gemma_refusal_results.json", "w"), indent=2)
print(f"\nSaved to {OUT_DIR / 'gemma_refusal_results.json'}")
