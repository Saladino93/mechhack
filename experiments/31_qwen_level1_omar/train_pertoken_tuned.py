"""Tuned per-token probes — paper-exact wd, bigger batch, class-balanced pos_weight,
more seeds, more patience.

Outputs into results.json under archs `multimax_simple_tuned`,
`multimax_kramar_tuned`, `attention_deepmind_tuned` so they coexist with the
non-tuned variants.
"""
import json, math, time, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_IDXS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
SEEDS = [0, 1, 2, 3, 4]
SWEEP_LAYERS = [30, 40, 50, 60]

OUT_DIR = Path(__file__).parent
EXTRACTS = Path("/home/ubuntu/extracts/qwen36")
SAMPLES_FILE = (OUT_DIR.parent.parent /
                "datasets/refusal_probes/qwen36/attacks_full.jsonl")

sys.path.insert(0, str(OUT_DIR.parent / "16_multimax_probe_omar"))
from probes import MultiMaxProbe  # noqa: E402


class MultiMaxSimple(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)
    def forward(self, x_full, mask):
        s = self.fc(x_full.float()).squeeze(-1)
        s = s.masked_fill(~mask, float("-inf"))
        return s.max(dim=1).values


class AttentionProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.head = nn.Linear(d, 1)
    def forward(self, x_full, mask):
        d = x_full.shape[-1]
        x_full = x_full.float()
        logits = (x_full @ self.q) / math.sqrt(d)
        logits = logits.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(logits, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", alpha, x_full)
        return self.head(pooled).squeeze(-1)


# Paper-aligned tuning
HP = {
    "multimax_simple_tuned":    {"factory": MultiMaxSimple,
                                  "lr": 1e-3, "wd": 1e-3},
    "multimax_kramar_tuned":    {"factory": lambda d: MultiMaxProbe(d_model=d),
                                  "lr": 1e-4, "wd": 3e-3},   # paper-exact
    "attention_deepmind_tuned": {"factory": AttentionProbe,
                                  "lr": 5e-4, "wd": 1e-3},
}
EPOCHS = 25
BATCH  = 32
PATIENCE = 5


# ---- data ----
all_s = {json.loads(l)["sample_id"]: json.loads(l) for l in open(SAMPLES_FILE)}
labeled = [s for s in all_s.values() if s.get("is_refusal") is not None]
ids_all = [s["sample_id"] for s in labeled
           if (EXTRACTS / f"{s['sample_id']}.pt").exists()]
y = np.array([int(all_s[sid]["is_refusal"]) for sid in ids_all], dtype=np.int64)
splits = np.array([all_s[sid]["split"] for sid in ids_all])
train_idx = np.where(splits == "train")[0]
test_idx  = np.where(splits == "test")[0]
print(f"Loaded {len(ids_all)} (train={len(train_idx)} test={len(test_idx)})")

# Class-balanced pos_weight = n_neg / n_pos on the train set
n_pos = int(y[train_idx].sum()); n_neg = int((1-y[train_idx]).sum())
pos_weight = n_neg / max(n_pos, 1)
print(f"  pos_weight = n_neg/n_pos = {n_neg}/{n_pos} = {pos_weight:.3f}")

rng = np.random.default_rng(0)
perm = rng.permutation(len(train_idx))
n_val = max(int(len(train_idx) * 0.10), 20)
val_idx = train_idx[perm[:n_val]]
inner_train_idx = train_idx[perm[n_val:]]


def cache_layer(L):
    li = LAYER_IDXS.index(L)
    print(f"  Caching L{L:02d}...")
    t0 = time.time()
    tens, masks = [], []
    for i, sid in enumerate(ids_all):
        ex = torch.load(str(EXTRACTS / f"{sid}.pt"),
                         weights_only=False, map_location="cpu")
        tens.append(ex["residuals"][li].half())
        masks.append(ex["attention_mask"].bool())
    d = tens[0].shape[-1]
    print(f"  L{L:02d} cached in {time.time()-t0:.1f}s")
    return tens, masks, d


def pad_batch(idx_list, tens, masks, d):
    sub_t = [tens[i] for i in idx_list]
    sub_m = [masks[i] for i in idx_list]
    T = max(t.shape[0] for t in sub_t)
    N = len(sub_t)
    x  = torch.zeros(N, T, d, dtype=torch.float32, device=DEVICE)
    mk = torch.zeros(N, T, dtype=torch.bool, device=DEVICE)
    for i, (t, m) in enumerate(zip(sub_t, sub_m)):
        x[i, :t.shape[0]] = t.to(DEVICE).float()
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


def train_one(arch, d, tens, masks, seed):
    hp = HP[arch]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = hp["factory"](d).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    pw = torch.tensor(pos_weight, device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.tensor(y[inner_train_idx], dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y[val_idx], dtype=torch.float32, device=DEVICE)
    rng = np.random.default_rng(seed)
    best_loss, best_state, patience = float("inf"), None, PATIENCE
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(len(inner_train_idx))
        for st in range(0, len(inner_train_idx), BATCH):
            bi = order[st:st+BATCH]
            real = [int(inner_train_idx[k]) for k in bi]
            x, m = pad_batch(real, tens, masks, d)
            logits = model(x, m)
            loss = bce(logits, yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vl = 0.0; nb = 0
        with torch.no_grad():
            for st in range(0, len(val_idx), BATCH):
                end = min(st+BATCH, len(val_idx))
                real = [int(val_idx[k]) for k in range(st, end)]
                x, m = pad_batch(real, tens, masks, d)
                vl += bce(model(x, m), yv[st:end]).item(); nb += 1
        vl /= max(nb, 1)
        if vl < best_loss - 1e-4:
            best_loss = vl; patience = PATIENCE
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0: break
    if best_state is not None: model.load_state_dict(best_state)
    return model


def predict_test(model, d, tens, masks):
    model.eval()
    out = np.zeros(len(test_idx), dtype=np.float32)
    with torch.no_grad():
        for st in range(0, len(test_idx), BATCH):
            end = min(st+BATCH, len(test_idx))
            real = [int(test_idx[k]) for k in range(st, end)]
            x, m = pad_batch(real, tens, masks, d)
            out[st:end] = torch.sigmoid(model(x, m)).cpu().numpy()
    return out


results = {arch: {"per_layer": {}, "hp": {**HP[arch], "factory": HP[arch]["factory"].__name__
                                             if hasattr(HP[arch]["factory"], '__name__')
                                             else "lambda"}}
           for arch in HP}
for L in SWEEP_LAYERS:
    tens, masks, d = cache_layer(L)
    for arch in HP:
        seed_aucs = []; seed_pred = []
        for seed in SEEDS:
            t0 = time.time()
            m = train_one(arch, d, tens, masks, seed)
            p = predict_test(m, d, tens, masks)
            seed_pred.append(p)
            a = auc_with_ci(y[test_idx], p)[0]
            seed_aucs.append(a)
            print(f"    {arch:25s} L{L:02d} seed{seed}: AUC {a:.4f}  {time.time()-t0:.0f}s")
        mean_p = np.mean(np.stack(seed_pred, axis=0), axis=0)
        auc, lo, hi = auc_with_ci(y[test_idx], mean_p, seed=L)
        results[arch]["per_layer"][str(L)] = {
            "auc_seedavg": auc, "ci95": [lo, hi],
            "auc_per_seed": [float(x) for x in seed_aucs],
            "auc_seed_mean": float(np.mean(seed_aucs)),
            "auc_seed_std":  float(np.std(seed_aucs)),
        }
        print(f"  {arch:25s} L{L:02d}: AUC {auc:.4f} [{lo:.4f},{hi:.4f}]  "
              f"seed mean±std {np.mean(seed_aucs):.4f}±{np.std(seed_aucs):.4f}")
        # Save partial
        res_path = OUT_DIR / "results.json"
        all_res = json.loads(res_path.read_text()) if res_path.exists() else {}
        # only persist auc data (drop the lambda factory which won't json-serialize)
        save_results = {arch: {"per_layer": v["per_layer"], "hp_lr": v["hp"]["lr"],
                                "hp_wd": v["hp"]["wd"]} for arch, v in results.items()}
        all_res.setdefault("by_arch", {}).update(save_results)
        res_path.write_text(json.dumps(all_res, indent=2))
    del tens, masks

print(f"\nSaved tuned results to {OUT_DIR / 'results.json'}")
