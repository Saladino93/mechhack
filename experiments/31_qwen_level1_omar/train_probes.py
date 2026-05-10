"""Probe training for Qwen Refusal — 4 heads × 13 layers.

Inputs:
  --extracts_dir   /home/ubuntu/extracts/qwen36/  (per-sample .pt with key 'residuals' (13, N, d))
  --samples_file   datasets/refusal_probes/qwen36/attacks_full.jsonl
  --out_dir        experiments/30_qwen_level1_omar/

Per-layer (13 layers): linear (mean), linear (last), MLP (mean), attention (full+mask).
Global head: CC++ Head A — concat all 13 mean-pooled vectors, sklearn LR with C-sweep.

Splits: dataset's pre-defined train/test (NOT a random 70/30).
Seeds: 5 per trainable head; C-sweep deterministic.
Metric: test AUC + 1000-bootstrap 95% CI.
"""
import os, sys, json, math, time, random, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Kramár probes (MultiMax etc.) from exp 16
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "16_multimax_probe_omar"))
from probes import MultiMaxProbe  # noqa: E402

DTYPE_NP = np.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_IDXS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
SEEDS = [0, 1, 2, 3, 4]


# ---------------- probe modules ----------------
class LinearProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)
    def forward(self, x_pooled, x_full=None, mask=None):
        return self.fc(x_pooled).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, d, hidden=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 1))
    def forward(self, x_pooled, x_full=None, mask=None):
        return self.net(x_pooled).squeeze(-1)


class AttentionProbe(nn.Module):
    """Single learned-query attention pool."""
    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.head = nn.Linear(d, 1)
    def forward(self, x_pooled, x_full, mask):
        d = x_full.shape[-1]
        logits = (x_full @ self.q) / math.sqrt(d)        # (B, N)
        logits = logits.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(logits, dim=-1)                # (B, N)
        pooled = torch.einsum("bn,bnd->bd", alpha, x_full)
        return self.head(pooled).squeeze(-1)


# ---------------- data ----------------
def load_split(samples_file):
    """Yield (sample_id, label, split). Drop is_refusal=None rows."""
    rows = []
    for line in open(samples_file):
        s = json.loads(line)
        if s.get("is_refusal") is None:
            continue
        rows.append((s["sample_id"], int(bool(s["is_refusal"])), s.get("split")))
    return rows


def cache_pool_features(extracts_dir: Path, sample_ids, cache_path: Path):
    """Build (N_samples, n_layers, d) for both mean and last-token pools.
    Caches to .npz for re-use."""
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
        r = ex["residuals"].float()              # (n_layers=13, N, d)
        m = ex["attention_mask"].bool()           # (N,)
        if not m.any():
            skipped += 1; continue
        last_idx = int(m.nonzero().max().item())
        # mean pool over masked tokens, per layer
        m_f = m.float().unsqueeze(0).unsqueeze(-1)   # (1, N, 1)
        mean = (r * m_f).sum(1) / m_f.sum(1).clamp(min=1)   # (n_layers, d)
        last = r[:, last_idx, :]                       # (n_layers, d)
        mean_feats.append(mean.numpy().astype(DTYPE_NP))
        last_feats.append(last.numpy().astype(DTYPE_NP))
        ok_ids.append(sid)
        if (i+1) % 100 == 0:
            print(f"    [{i+1}/{len(sample_ids)}] cached, skipped={skipped}, {time.time()-t0:.1f}s")
    mean_arr = np.stack(mean_feats, axis=0)         # (N, n_layers, d)
    last_arr = np.stack(last_feats, axis=0)
    np.savez_compressed(cache_path, mean=mean_arr, last=last_arr,
                         sample_ids=np.array(ok_ids))
    print(f"  cache saved: mean {mean_arr.shape} last {last_arr.shape} "
          f"skipped={skipped} in {time.time()-t0:.1f}s")
    return mean_arr, last_arr, ok_ids


# ---------------- train trainable heads (linear/MLP/attn) ----------------
def train_pooled(arch, X_train, y_train, X_val, y_val, d, seed, epochs=50, batch=32):
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


def train_attention(extracts_dir, sample_ids, labels, layer_idx_in_extract,
                    train_idx, val_idx, d, seed, epochs=20, batch=8):
    """Attention probe — needs full per-token features. Lazy-load .pt files."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = AttentionProbe(d).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()

    def fetch(idx_list):
        """Load full per-token features for the given sample indices (one layer)."""
        tens, masks = [], []
        for i in idx_list:
            p = extracts_dir / f"{sample_ids[i]}.pt"
            ex = torch.load(str(p), weights_only=False, map_location="cpu")
            r = ex["residuals"][layer_idx_in_extract].float()    # (N, d)
            m = ex["attention_mask"].bool()
            tens.append(r); masks.append(m)
        T = max(t.shape[0] for t in tens)
        N = len(tens)
        x = torch.zeros(N, T, d, dtype=torch.float32)
        mk = torch.zeros(N, T, dtype=torch.bool)
        for i, (t, m) in enumerate(zip(tens, masks)):
            x[i, :t.shape[0]] = t; mk[i, :m.shape[0]] = m
        return x.to(DEVICE), mk.to(DEVICE)

    yt_all = torch.tensor(labels[train_idx], dtype=torch.float32, device=DEVICE)
    yv_all = torch.tensor(labels[val_idx],   dtype=torch.float32, device=DEVICE)

    rng = np.random.default_rng(seed)
    best_loss, best_state, patience = float("inf"), None, 4
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for st in range(0, len(train_idx), batch):
            bi = order[st:st+batch]
            real_idxs = [int(train_idx[k]) for k in bi]
            x, mk = fetch(real_idxs)
            yb = yt_all[bi]
            logits = model(None, x, mk)
            loss = bce(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # validation in sub-batches
        model.eval()
        vl = 0.0; nb = 0
        with torch.no_grad():
            for st in range(0, len(val_idx), batch):
                real_idxs = [int(val_idx[k]) for k in range(st, min(st+batch, len(val_idx)))]
                x, mk = fetch(real_idxs)
                yb = yv_all[st:st+batch]
                vl += bce(model(None, x, mk), yb).item(); nb += 1
        vl /= max(nb, 1)
        if vl < best_loss - 1e-4:
            best_loss = vl; patience = 4
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0: break
    if best_state is not None: model.load_state_dict(best_state)
    return model


def train_kramar(probe_cls, extracts_dir, sample_ids, labels, layer_idx_in_extract,
                  train_idx, val_idx, d, seed, epochs=20, batch=8):
    """Trainer for Kramár-style probes that take (x_full, mask) — MultiMax/Rolling."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = probe_cls(d_model=d).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3e-3)
    bce = nn.BCEWithLogitsLoss()

    def fetch(idx_list):
        tens, masks = [], []
        for i in idx_list:
            p = extracts_dir / f"{sample_ids[i]}.pt"
            ex = torch.load(str(p), weights_only=False, map_location="cpu")
            r = ex["residuals"][layer_idx_in_extract].float()
            m = ex["attention_mask"].bool()
            tens.append(r); masks.append(m)
        T = max(t.shape[0] for t in tens)
        N = len(tens)
        x = torch.zeros(N, T, d, dtype=torch.float32)
        mk = torch.zeros(N, T, dtype=torch.bool)
        for i, (t, m) in enumerate(zip(tens, masks)):
            x[i, :t.shape[0]] = t; mk[i, :m.shape[0]] = m
        return x.to(DEVICE), mk.to(DEVICE)

    yt_all = torch.tensor(labels[train_idx], dtype=torch.float32, device=DEVICE)
    yv_all = torch.tensor(labels[val_idx],   dtype=torch.float32, device=DEVICE)

    rng = np.random.default_rng(seed)
    best_loss, best_state, patience = float("inf"), None, 4
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for st in range(0, len(train_idx), batch):
            bi = order[st:st+batch]
            real_idxs = [int(train_idx[k]) for k in bi]
            x, mk = fetch(real_idxs)
            yb = yt_all[bi]
            logits = model(x, mk)
            loss = bce(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vl = 0.0; nb = 0
        with torch.no_grad():
            for st in range(0, len(val_idx), batch):
                real_idxs = [int(val_idx[k]) for k in range(st, min(st+batch, len(val_idx)))]
                x, mk = fetch(real_idxs)
                yb = yv_all[st:st+batch]
                vl += bce(model(x, mk), yb).item(); nb += 1
        vl /= max(nb, 1)
        if vl < best_loss - 1e-4:
            best_loss = vl; patience = 4
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0: break
    if best_state is not None: model.load_state_dict(best_state)
    return model


def predict_kramar(model, extracts_dir, sample_ids, idx_list,
                    layer_idx_in_extract, d, batch=8):
    model.eval()
    out = np.zeros(len(idx_list), dtype=np.float32)
    with torch.no_grad():
        for st in range(0, len(idx_list), batch):
            real_idxs = idx_list[st:st+batch]
            tens, masks = [], []
            for i in real_idxs:
                p = extracts_dir / f"{sample_ids[i]}.pt"
                ex = torch.load(str(p), weights_only=False, map_location="cpu")
                r = ex["residuals"][layer_idx_in_extract].float()
                m = ex["attention_mask"].bool()
                tens.append(r); masks.append(m)
            T = max(t.shape[0] for t in tens)
            x = torch.zeros(len(tens), T, d, dtype=torch.float32, device=DEVICE)
            mk = torch.zeros(len(tens), T, dtype=torch.bool, device=DEVICE)
            for i, (t, m) in enumerate(zip(tens, masks)):
                x[i, :t.shape[0]] = t.to(DEVICE)
                mk[i, :m.shape[0]] = m.to(DEVICE)
            logits = model(x, mk)
            out[st:st+len(real_idxs)] = torch.sigmoid(logits).cpu().numpy()
    return out


# ---------------- evaluation ----------------
def auc_with_ci(y_true, y_score, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, size=n)
        if len(set(y_true[ix].tolist())) < 2: continue
        aucs.append(roc_auc_score(y_true[ix], y_score[ix]))
    if aucs:
        lo, hi = float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
    else:
        lo = hi = float("nan")
    return auc, lo, hi


def predict_pooled(model, X):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(Xt)).cpu().numpy()


def predict_attention(model, extracts_dir, sample_ids, idx_list,
                      layer_idx_in_extract, d, batch=8):
    model.eval()
    out = np.zeros(len(idx_list), dtype=np.float32)
    with torch.no_grad():
        for st in range(0, len(idx_list), batch):
            real_idxs = idx_list[st:st+batch]
            tens, masks = [], []
            for i in real_idxs:
                p = extracts_dir / f"{sample_ids[i]}.pt"
                ex = torch.load(str(p), weights_only=False, map_location="cpu")
                r = ex["residuals"][layer_idx_in_extract].float()
                m = ex["attention_mask"].bool()
                tens.append(r); masks.append(m)
            T = max(t.shape[0] for t in tens)
            x = torch.zeros(len(tens), T, d, dtype=torch.float32, device=DEVICE)
            mk = torch.zeros(len(tens), T, dtype=torch.bool, device=DEVICE)
            for i, (t, m) in enumerate(zip(tens, masks)):
                x[i, :t.shape[0]] = t.to(DEVICE)
                mk[i, :m.shape[0]] = m.to(DEVICE)
            logits = model(None, x, mk)
            out[st:st+len(real_idxs)] = torch.sigmoid(logits).cpu().numpy()
    return out


# ---------------- pipeline ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracts_dir", default="/home/ubuntu/extracts/qwen36")
    ap.add_argument("--samples_file",
                    default="/lambda/nfs/jsWnew/mechhack/datasets/refusal_probes/qwen36/attacks_full.jsonl")
    ap.add_argument("--out_dir",
                    default="/lambda/nfs/jsWnew/mechhack/experiments/30_qwen_level1_omar")
    ap.add_argument("--archs",
                    default="arditi,linear_mean,linear_last,mlp_mean,cc_concat,attention,multimax",
                    help="comma-separated subset of {arditi,linear_mean,linear_last,mlp_mean,cc_concat,attention,multimax}")
    ap.add_argument("--layers", default=",".join(map(str, LAYER_IDXS)),
                    help="comma-separated layer indices to sweep for fast heads (linear/MLP)")
    ap.add_argument("--attn_layers", default="0,10,20,30,40,50,60",
                    help="comma-separated layer indices for attention probe (slow). "
                         "Default: every-10 subset.")
    ap.add_argument("--skip_attention_layers", default="",
                    help="(deprecated alias) comma-separated layer indices to skip for attention.")
    args = ap.parse_args()

    extracts_dir = Path(args.extracts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archs = args.archs.split(",")
    sweep_layers = [int(x) for x in args.layers.split(",")]
    skip_attn = set(int(x) for x in args.skip_attention_layers.split(",") if x)

    # Validate layers are in the extracted set
    for L in sweep_layers:
        if L not in LAYER_IDXS:
            raise ValueError(f"layer {L} not in extracted set {LAYER_IDXS}")

    rows = load_split(args.samples_file)
    print(f"Loaded {len(rows)} labeled samples (excluded is_refusal=None)")

    # Pre-cache mean + last pool features
    cache_path = out_dir / "pool_cache.npz"
    sample_ids_all = [r[0] for r in rows]
    label_all = {r[0]: r[1] for r in rows}
    split_all = {r[0]: r[2] for r in rows}
    mean_arr, last_arr, ok_ids = cache_pool_features(extracts_dir, sample_ids_all, cache_path)

    y = np.array([label_all[sid] for sid in ok_ids], dtype=np.int64)
    splits = np.array([split_all[sid] for sid in ok_ids])
    train_idx = np.where(splits == "train")[0]
    test_idx  = np.where(splits == "test")[0]
    print(f"After cache: N={len(ok_ids)}  train={len(train_idx)}  test={len(test_idx)} "
          f"pos_rate={y.mean():.3f}  pos_train={y[train_idx].mean():.3f} "
          f"pos_test={y[test_idx].mean():.3f}")

    d = mean_arr.shape[-1]
    # inner train/val for trainable heads (10% of train as val for early stop)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_idx))
    n_val = max(int(len(train_idx) * 0.10), 20)
    inner_val_idx = train_idx[perm[:n_val]]
    inner_train_idx = train_idx[perm[n_val:]]
    print(f"  inner_train={len(inner_train_idx)}  inner_val={len(inner_val_idx)}")

    # Merge with any existing results.json so partial runs don't clobber
    res_path = out_dir / "results.json"
    if res_path.exists():
        results = json.loads(res_path.read_text())
        if "by_arch" not in results: results["by_arch"] = {}
        # refresh top-level metadata in case data set/split changed
        results["layers"] = LAYER_IDXS
        results["n_train"] = int(len(train_idx))
        results["n_test"]  = int(len(test_idx))
        results["pos_rate_train"] = float(y[train_idx].mean())
        results["pos_rate_test"]  = float(y[test_idx].mean())
        print(f"  Merging into existing results.json (archs already present: "
              f"{list(results['by_arch'].keys())})")
    else:
        results = {"layers": LAYER_IDXS, "n_train": int(len(train_idx)),
                   "n_test": int(len(test_idx)), "pos_rate_train": float(y[train_idx].mean()),
                   "pos_rate_test": float(y[test_idx].mean()),
                   "by_arch": {}}

    # --- per-layer trainable heads ---
    for arch in ["linear_mean", "linear_last", "mlp_mean"]:
        if arch not in archs: continue
        results["by_arch"][arch] = {"per_layer": {}}
        for L in sweep_layers:
            li = LAYER_IDXS.index(L)
            X = (mean_arr if arch != "linear_last" else last_arr)[:, li, :]
            seed_aucs = []; seed_pred = []
            for seed in SEEDS:
                model = train_pooled(
                    "linear" if arch != "mlp_mean" else "mlp",
                    X[inner_train_idx], y[inner_train_idx].astype(np.float32),
                    X[inner_val_idx],   y[inner_val_idx].astype(np.float32),
                    d, seed)
                proba = predict_pooled(model, X[test_idx])
                seed_pred.append(proba)
                seed_aucs.append(auc_with_ci(y[test_idx], proba)[0])
            mean_proba = np.mean(np.stack(seed_pred, axis=0), axis=0)
            auc, lo, hi = auc_with_ci(y[test_idx], mean_proba, seed=L)
            results["by_arch"][arch]["per_layer"][str(L)] = {
                "auc_seedavg": auc, "ci95": [lo, hi],
                "auc_per_seed": [float(x) for x in seed_aucs],
                "auc_seed_mean": float(np.mean(seed_aucs)),
                "auc_seed_std":  float(np.std(seed_aucs)),
            }
            print(f"  {arch:12s} L{L:02d}: AUC {auc:.4f} [{lo:.4f},{hi:.4f}] "
                  f"seed mean±std {np.mean(seed_aucs):.4f}±{np.std(seed_aucs):.4f}")
        # Save partial after every arch
        json.dump(results, open(out_dir / "results.json", "w"), indent=2)

    # --- Arditi mean-difference direction (no training) ---
    if "arditi" in archs:
        results["by_arch"]["arditi"] = {"per_layer": {}}
        for L in sweep_layers:
            li = LAYER_IDXS.index(L)
            X_train_L = mean_arr[train_idx, li, :]
            mu_pos = X_train_L[y[train_idx] == 1].mean(axis=0)
            mu_neg = X_train_L[y[train_idx] == 0].mean(axis=0)
            d_dir = mu_pos - mu_neg
            d_dir = d_dir / (np.linalg.norm(d_dir) + 1e-8)
            scores = mean_arr[test_idx, li, :] @ d_dir
            auc, lo, hi = auc_with_ci(y[test_idx], scores, seed=L)
            results["by_arch"]["arditi"]["per_layer"][str(L)] = {
                "auc": auc, "ci95": [lo, hi],
            }
            print(f"  arditi      L{L:02d}: AUC {auc:.4f} [{lo:.4f},{hi:.4f}]")
        json.dump(results, open(out_dir / "results.json", "w"), indent=2)

    # --- MultiMax probe (Kramár 2026 Architecture C) ---
    if "multimax" in archs:
        results["by_arch"]["multimax"] = {"per_layer": {}}
        attn_sweep = [int(x) for x in args.attn_layers.split(",")]
        for L in attn_sweep:
            if L not in LAYER_IDXS: continue
            li = LAYER_IDXS.index(L)
            seed_aucs = []; seed_pred = []
            for seed in SEEDS[:3]:
                t0 = time.time()
                model = train_kramar(MultiMaxProbe, extracts_dir, ok_ids,
                                      y.astype(np.float32), li, inner_train_idx,
                                      inner_val_idx, d, seed)
                proba = predict_kramar(model, extracts_dir, ok_ids,
                                        list(test_idx), li, d)
                seed_pred.append(proba)
                a = auc_with_ci(y[test_idx], proba)[0]
                seed_aucs.append(a)
                print(f"    multimax L{L:02d} seed{seed}: AUC {a:.4f}  {time.time()-t0:.0f}s")
            mean_proba = np.mean(np.stack(seed_pred, axis=0), axis=0)
            auc, lo, hi = auc_with_ci(y[test_idx], mean_proba, seed=L)
            results["by_arch"]["multimax"]["per_layer"][str(L)] = {
                "auc_seedavg": auc, "ci95": [lo, hi],
                "auc_per_seed": [float(x) for x in seed_aucs],
                "auc_seed_mean": float(np.mean(seed_aucs)),
                "auc_seed_std":  float(np.std(seed_aucs)),
            }
            print(f"  multimax    L{L:02d}: AUC {auc:.4f} [{lo:.4f},{hi:.4f}]")
            json.dump(results, open(out_dir / "results.json", "w"), indent=2)

    # --- attention probe ---
    if "attention" in archs:
        results["by_arch"]["attention"] = {"per_layer": {}}
        for L in sweep_layers:
            if L in skip_attn:
                print(f"  attention   L{L:02d}: SKIPPED")
                continue
            li = LAYER_IDXS.index(L)
            seed_aucs = []; seed_pred = []
            for seed in SEEDS[:3]:  # 3 seeds for the slow head, not 5
                t0 = time.time()
                model = train_attention(extracts_dir, ok_ids, y.astype(np.float32),
                                         li, inner_train_idx, inner_val_idx,
                                         d, seed)
                proba = predict_attention(model, extracts_dir, ok_ids,
                                           list(test_idx), li, d)
                seed_pred.append(proba)
                a = auc_with_ci(y[test_idx], proba)[0]
                seed_aucs.append(a)
                print(f"    attn L{L:02d} seed{seed}: AUC {a:.4f}  {time.time()-t0:.0f}s")
            mean_proba = np.mean(np.stack(seed_pred, axis=0), axis=0)
            auc, lo, hi = auc_with_ci(y[test_idx], mean_proba, seed=L)
            results["by_arch"]["attention"]["per_layer"][str(L)] = {
                "auc_seedavg": auc, "ci95": [lo, hi],
                "auc_per_seed": [float(x) for x in seed_aucs],
                "auc_seed_mean": float(np.mean(seed_aucs)),
                "auc_seed_std":  float(np.std(seed_aucs)),
            }
            print(f"  attention   L{L:02d}: AUC {auc:.4f} [{lo:.4f},{hi:.4f}]")
            json.dump(results, open(out_dir / "results.json", "w"), indent=2)

    # --- CC++ Head A: concat all 13 mean-pools, sklearn LR with C-sweep ---
    if "cc_concat" in archs:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        Xc = mean_arr.reshape(mean_arr.shape[0], -1)   # (N, 13*d)
        Xc_train, Xc_test = Xc[train_idx], Xc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # inner val for C-sweep
        Xc_inner_train, Xc_inner_val, y_inner_train, y_inner_val = train_test_split(
            Xc_train, y_train, test_size=0.2, stratify=y_train, random_state=0)
        Cs = [0.01, 0.1, 1.0, 10.0]
        c_aucs = {}
        for C in Cs:
            lr = LogisticRegression(solver="lbfgs", max_iter=2000, C=C)
            lr.fit(Xc_inner_train, y_inner_train)
            p = lr.predict_proba(Xc_inner_val)[:, 1]
            a = roc_auc_score(y_inner_val, p)
            c_aucs[C] = float(a)
            print(f"  cc_concat C={C:>5}: inner-val AUC {a:.4f}")
        best_C = max(Cs, key=lambda c: c_aucs[c])
        lr = LogisticRegression(solver="lbfgs", max_iter=2000, C=best_C)
        lr.fit(Xc_train, y_train)
        proba = lr.predict_proba(Xc_test)[:, 1]
        auc, lo, hi = auc_with_ci(y_test, proba)
        results["by_arch"]["cc_concat"] = {
            "best_C": best_C, "C_inner_val_aucs": c_aucs,
            "test_auc": auc, "ci95": [lo, hi],
            "feature_dim": int(Xc.shape[1]),
        }
        print(f"  cc_concat (best C={best_C}): test AUC {auc:.4f} [{lo:.4f},{hi:.4f}]  "
              f"feat_dim={Xc.shape[1]}")
        json.dump(results, open(out_dir / "results.json", "w"), indent=2)

    print("\nResults saved to", out_dir / "results.json")


if __name__ == "__main__":
    main()
