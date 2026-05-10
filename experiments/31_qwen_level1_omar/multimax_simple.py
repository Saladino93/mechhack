"""Minimal MultiMax: per-token linear probe, max-over-tokens.

Strictly simpler than Kramár 2026 Architecture C (which adds TransformMLP
d_model→100 + 10 heads + sharp softmax τ=0.1). This version keeps the core
'replace softmax with max' insight and drops everything else:

    score_t = w · x_t + b                  # one linear projection
    pred    = max_t score_t  (over masked t)
    loss    = BCE(pred, label)

Runs at the same 4 layers (30, 40, 50, 60) the heavy variant uses.
~30 s/seed, 3 seeds → ~6 min total per layer (12 min × 4 = ~25 min).
"""
import json, math, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_IDXS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
SEEDS = [0, 1, 2]
SWEEP_LAYERS = [30, 40, 50, 60]

OUT_DIR = Path(__file__).parent
EXTRACTS = Path("/home/ubuntu/extracts/qwen36")
SAMPLES_FILE = (OUT_DIR.parent.parent /
                "datasets/refusal_probes/qwen36/attacks_full.jsonl")


class MaxLinearProbe(nn.Module):
    """Per-token linear, max-over-tokens. The minimal MultiMax."""
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)
    def forward(self, x_full, mask):
        # x_full: (B, T, d), mask: (B, T) bool
        logits = self.fc(x_full.float()).squeeze(-1)            # (B, T)
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits.max(dim=1).values                          # (B,)


# Load samples + ids
all_s = {json.loads(l)["sample_id"]: json.loads(l) for l in open(SAMPLES_FILE)}
labeled = [s for s in all_s.values() if s.get("is_refusal") is not None]
ids_all = [s["sample_id"] for s in labeled
           if (EXTRACTS / f"{s['sample_id']}.pt").exists()]
y = np.array([int(all_s[sid]["is_refusal"]) for sid in ids_all], dtype=np.int64)
splits = np.array([all_s[sid]["split"] for sid in ids_all])
train_idx = np.where(splits == "train")[0]
test_idx  = np.where(splits == "test")[0]
print(f"Loaded {len(ids_all)} samples (train={len(train_idx)}, test={len(test_idx)})")

# Inner val for early stop
rng = np.random.default_rng(0)
perm = rng.permutation(len(train_idx))
n_val = max(int(len(train_idx) * 0.10), 20)
val_idx = train_idx[perm[:n_val]]
inner_train_idx = train_idx[perm[n_val:]]


def fetch(idx_list, layer_idx_in_extract, d):
    tens, masks = [], []
    for i in idx_list:
        ex = torch.load(str(EXTRACTS / f"{ids_all[i]}.pt"),
                         weights_only=False, map_location="cpu")
        r = ex["residuals"][layer_idx_in_extract].float()
        m = ex["attention_mask"].bool()
        tens.append(r); masks.append(m)
    T = max(t.shape[0] for t in tens)
    N = len(tens)
    x  = torch.zeros(N, T, d, dtype=torch.float32, device=DEVICE)
    mk = torch.zeros(N, T, dtype=torch.bool, device=DEVICE)
    for i, (t, m) in enumerate(zip(tens, masks)):
        x[i, :t.shape[0]] = t.to(DEVICE)
        mk[i, :m.shape[0]] = m.to(DEVICE)
    return x, mk


def auc_with_ci(y_true, y_score, n_boot=1000, seed=0):
    auc = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true); aucs = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, size=n)
        if len(set(y_true[ix].tolist())) < 2: continue
        aucs.append(roc_auc_score(y_true[ix], y_score[ix]))
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def train_one(layer, seed, epochs=10, batch=8):
    li = LAYER_IDXS.index(layer)
    # Inspect d_model from one extract
    ex = torch.load(str(EXTRACTS / f"{ids_all[0]}.pt"), weights_only=False,
                     map_location="cpu")
    d = ex["residuals"].shape[-1]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = MaxLinearProbe(d).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y[inner_train_idx], dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y[val_idx], dtype=torch.float32, device=DEVICE)
    rng = np.random.default_rng(seed)
    best_loss, best_state, patience = float("inf"), None, 3
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(inner_train_idx))
        for st in range(0, len(inner_train_idx), batch):
            bi = order[st:st+batch]
            real = [int(inner_train_idx[k]) for k in bi]
            x, m = fetch(real, li, d)
            logits = model(x, m)
            loss = bce(logits, yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        vl = 0.0; nb = 0
        with torch.no_grad():
            for st in range(0, len(val_idx), batch):
                real = [int(val_idx[k]) for k in range(st, min(st+batch, len(val_idx)))]
                x, m = fetch(real, li, d)
                yb = yv[st:st+batch]
                vl += bce(model(x, m), yb).item(); nb += 1
        vl /= max(nb, 1)
        if vl < best_loss - 1e-4:
            best_loss = vl; patience = 3
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, d


def predict_test(model, layer, d, batch=8):
    li = LAYER_IDXS.index(layer)
    out = np.zeros(len(test_idx), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for st in range(0, len(test_idx), batch):
            real = [int(test_idx[k]) for k in range(st, min(st+batch, len(test_idx)))]
            x, m = fetch(real, li, d)
            out[st:st+len(real)] = torch.sigmoid(model(x, m)).cpu().numpy()
    return out


results = {"per_layer": {}}
for L in SWEEP_LAYERS:
    seed_aucs = []; seed_pred = []
    for seed in SEEDS:
        t0 = time.time()
        m, d = train_one(L, seed)
        proba = predict_test(m, L, d)
        seed_pred.append(proba)
        a = auc_with_ci(y[test_idx], proba)[0]
        seed_aucs.append(a)
        print(f"    multimax_simple L{L:02d} seed{seed}: AUC {a:.4f}  {time.time()-t0:.0f}s")
    mean_p = np.mean(np.stack(seed_pred, axis=0), axis=0)
    auc, lo, hi = auc_with_ci(y[test_idx], mean_p, seed=L)
    results["per_layer"][str(L)] = {
        "auc_seedavg": auc, "ci95": [lo, hi],
        "auc_per_seed": [float(x) for x in seed_aucs],
        "auc_seed_mean": float(np.mean(seed_aucs)),
        "auc_seed_std":  float(np.std(seed_aucs)),
    }
    print(f"  multimax_simple L{L:02d}: AUC {auc:.4f} [{lo:.4f},{hi:.4f}]  "
          f"seed mean±std {np.mean(seed_aucs):.4f}±{np.std(seed_aucs):.4f}")

# Merge into results.json
res_path = OUT_DIR / "results.json"
all_res = json.loads(res_path.read_text()) if res_path.exists() else {}
all_res.setdefault("by_arch", {})["multimax_simple"] = results
res_path.write_text(json.dumps(all_res, indent=2))
print(f"Saved to {res_path}")
