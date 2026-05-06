"""Phase 2 probe training. Adds MLP and 4-head attention probe; layer sweep
for linear (using cached final_token_all_layers); 5 seeds; saves probe weights.

For Qwen refusal task, switches to the new chat-template extracts at
extracts/qwen36_ct/ — every other task uses Phase 1 extracts.
"""
import os, sys, json, time, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PHASE1 = Path(os.environ.get("PHASE1_DIR", "/mloscratch/protsenk-kaniko/probes/phase1"))
PHASE2 = Path(os.environ.get("PHASE2_DIR", "/mloscratch/protsenk-kaniko/probes/phase2"))
RESULTS = PHASE2 / "results"
WEIGHTS = PHASE2 / "weights"
RESULTS.mkdir(parents=True, exist_ok=True)
WEIGHTS.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
SEEDS = [0, 1, 2, 3, 4]
ARCHS = ["linear", "mlp", "attention", "attention_4h"]
REGIMES = ["batch", "incremental"]


# ---------------- probe modules ----------------
class LinearProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)
    def forward(self, x_final, x_full=None, mask=None):
        return self.fc(x_final).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, d, hidden=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 1))
    def forward(self, x_final, x_full=None, mask=None):
        return self.net(x_final).squeeze(-1)


class AttentionProbe(nn.Module):
    """Single learned-query attention over tokens."""
    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.head = nn.Linear(d, 1)
    def forward(self, x_final, x_full, mask):
        d = x_full.shape[-1]
        logits = (x_full @ self.q) / math.sqrt(d)
        logits = logits.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(logits, dim=-1)                    # (B, N)
        pooled = torch.einsum("bn,bnd->bd", alpha, x_full)   # (B, d)
        return self.head(pooled).squeeze(-1), alpha


class MultiHeadAttentionProbe(nn.Module):
    """K learned queries; concat per-head pooled vectors."""
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_heads, d) / math.sqrt(d))
        self.head = nn.Linear(d * n_heads, 1)
        self.n_heads = n_heads
    def forward(self, x_final, x_full, mask):
        B, N, d = x_full.shape
        logits = torch.einsum("bnd,kd->bnk", x_full, self.q) / math.sqrt(d)  # (B,N,K)
        logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = F.softmax(logits, dim=1)                       # (B, N, K)
        pooled = torch.einsum("bnk,bnd->bkd", alpha, x_full)   # (B, K, d)
        pooled = pooled.reshape(B, -1)                          # (B, K*d)
        return self.head(pooled).squeeze(-1), alpha


def make_probe(arch, d):
    if arch == "linear":      return LinearProbe(d)
    if arch == "mlp":         return MLPProbe(d)
    if arch == "attention":   return AttentionProbe(d)
    if arch == "attention_4h":return MultiHeadAttentionProbe(d, n_heads=4)
    raise ValueError(arch)


# ---------------- data loading ----------------
def extracts_root_for(model_key, task):
    """Refusal_qwen36 uses chat-template extracts; everything else uses Phase 1."""
    if task == "refusal_qwen36":
        return PHASE2 / "extracts" / "qwen36_ct"
    return PHASE1 / "extracts" / model_key


def load_extract(extracts_dir, sample_id):
    return torch.load(str(extracts_dir / f"{sample_id}.pt"), weights_only=False)


def build_dataset(samples, label_fn, extracts_dir):
    x_final, x_full_list, mask_list, y, ids = [], [], [], [], []
    skipped = 0
    for s in samples:
        try:
            ex = load_extract(extracts_dir, s["sample_id"])
        except FileNotFoundError:
            skipped += 1; continue
        full = ex["middle_layer_all_tokens"].to(DTYPE)
        m = ex["attention_mask"]
        if not m.any():
            skipped += 1; continue
        last = int(m.nonzero().max().item())
        x_final.append(full[last])
        x_full_list.append(full)
        mask_list.append(m)
        y.append(label_fn(s))
        ids.append(s["sample_id"])
    if skipped: print(f"    [warn] skipped {skipped}")
    if not x_final: return None
    return {
        "x_final": torch.stack(x_final, dim=0).to(DEVICE),
        "x_full_list": x_full_list,
        "mask_list": mask_list,
        "y": torch.tensor(y, dtype=torch.float32, device=DEVICE),
        "ids": ids,
    }


def pad_full(x_full_list, mask_list):
    N = len(x_full_list)
    T_max = max(t.shape[0] for t in x_full_list)
    d = x_full_list[0].shape[1]
    px = torch.zeros(N, T_max, d, dtype=DTYPE, device=DEVICE)
    pm = torch.zeros(N, T_max, dtype=torch.bool, device=DEVICE)
    for i, (t, m) in enumerate(zip(x_full_list, mask_list)):
        T_i = t.shape[0]
        px[i, :T_i] = t.to(DEVICE)
        pm[i, :T_i] = m.to(DEVICE)
    return px, pm


# ---------------- training ----------------
def train(arch, regime, ds, train_idx, test_idx, seed, d):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = make_probe(arch, d).to(DEVICE).to(DTYPE)
    bce = nn.BCEWithLogitsLoss()
    lr = 5e-4 if "attention" in arch else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    train_ix = np.array(train_idx).copy()
    np.random.default_rng(seed).shuffle(train_ix)
    best_loss, best_state, patience = float("inf"), None, 5

    if regime == "batch":
        for epoch in range(50):
            np.random.default_rng(seed + epoch).shuffle(train_ix)
            model.train()
            tl = 0; nb = 0
            for st in range(0, len(train_ix), 32):
                bi = train_ix[st:st+32]
                yb = ds["y"][bi]
                if arch in ("linear", "mlp"):
                    logits = model(ds["x_final"][bi])
                else:
                    px, pm = pad_full([ds["x_full_list"][i] for i in bi],
                                       [ds["mask_list"][i] for i in bi])
                    out = model(None, px, pm)
                    logits = out[0] if isinstance(out, tuple) else out
                loss = bce(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item(); nb += 1
            ev = evaluate(model, arch, ds, test_idx)
            if ev["loss"] < best_loss - 1e-4:
                best_loss = ev["loss"]; best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 5
            else:
                patience -= 1
                if patience <= 0: break
    else:  # incremental
        model.train()
        for i, idx in enumerate(train_ix):
            yb = ds["y"][idx:idx+1]
            if arch in ("linear", "mlp"):
                logits = model(ds["x_final"][idx:idx+1])
            else:
                px, pm = pad_full([ds["x_full_list"][idx]], [ds["mask_list"][idx]])
                out = model(None, px, pm)
                logits = out[0] if isinstance(out, tuple) else out
            loss = bce(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    return evaluate(model, arch, ds, test_idx), model


def evaluate(model, arch, ds, idx):
    model.eval()
    with torch.no_grad():
        if arch in ("linear", "mlp"):
            logits = model(ds["x_final"][idx])
        else:
            px, pm = pad_full([ds["x_full_list"][i] for i in idx], [ds["mask_list"][i] for i in idx])
            out = model(None, px, pm)
            logits = out[0] if isinstance(out, tuple) else out
        y = ds["y"][idx]
        loss = F.binary_cross_entropy_with_logits(logits, y).item()
        probs = torch.sigmoid(logits).cpu().numpy()
        y_np = y.cpu().numpy().astype(int)
        preds = (probs > 0.5).astype(int)
        acc = (preds == y_np).mean()
        try:
            from sklearn.metrics import f1_score, roc_auc_score
            f1 = f1_score(y_np, preds, zero_division=0)
            auc = roc_auc_score(y_np, probs) if len(set(y_np.tolist())) > 1 else float("nan")
        except ImportError:
            f1, auc = float("nan"), float("nan")
    return {"loss": loss, "acc": float(acc), "f1": float(f1), "auc": float(auc)}


def task_specs(manifest):
    cyber = manifest["cyber_samples"]
    for model_key in ["gemma4_31b", "qwen36"]:
        ref = manifest["refusal_samples"][model_key]
        yield (f"refusal_{model_key}", model_key, ref, lambda s: 1.0 if s["is_refusal"] else 0.0)
        for cls in ["prohibited", "high_risk_dual_use", "dual_use", "benign"]:
            pos = [s for s in cyber if s["label"] == cls]
            neg_all = [s for s in cyber if s["label"] != cls]
            other = sorted({s["label"] for s in neg_all})
            n_per = len(pos) // len(other)
            rng = random.Random(42 + hash(cls) % 1000)
            neg = []
            for c in other:
                pool = [s for s in neg_all if s["label"] == c]
                neg += rng.sample(pool, min(n_per, len(pool)))
            samples = pos + neg
            short = {"prohibited": "prohib", "high_risk_dual_use": "hdu",
                     "dual_use": "du", "benign": "ben"}[cls]
            yield (f"cyber_{short}_vs_rest_{model_key}", model_key, samples,
                   (lambda s, _cls=cls: 1.0 if s["label"] == _cls else 0.0))


def main():
    manifest = json.load(open(PHASE1 / "phase1_manifest.json"))
    rows = []
    log_path = RESULTS / "phase2_metrics.jsonl"
    log_f = open(log_path, "w")

    for task_name, model_key, samples, label_fn in task_specs(manifest):
        print(f"\n=== {task_name}  ({model_key}, {len(samples)}) ===")
        extracts = extracts_root_for(model_key, task_name)
        ds = build_dataset(samples, label_fn, extracts)
        if ds is None:
            print("  no data"); continue
        d = ds["x_final"].shape[1]
        N = len(ds["ids"])
        y_np = ds["y"].cpu().numpy().astype(int)
        pos_idx = np.where(y_np == 1)[0]; neg_idx = np.where(y_np == 0)[0]
        rng = np.random.default_rng(0)
        rng.shuffle(pos_idx); rng.shuffle(neg_idx)
        n_pt = int(len(pos_idx) * 0.3); n_nt = int(len(neg_idx) * 0.3)
        test_idx = np.concatenate([pos_idx[:n_pt], neg_idx[:n_nt]])
        train_idx = np.concatenate([pos_idx[n_pt:], neg_idx[n_nt:]])
        print(f"  d={d} N={N} train={len(train_idx)} test={len(test_idx)} extracts={extracts.name}")

        best_attn_state = None  # save for dashboard
        best_attn_auc = -1
        for arch in ARCHS:
            for reg in REGIMES:
                metrics_seeds = []
                for seed in SEEDS:
                    t0 = time.time()
                    metrics, model = train(arch, reg, ds, train_idx, test_idx, seed, d)
                    elapsed = time.time() - t0
                    rec = {"task": task_name, "model_key": model_key, "arch": arch,
                           "regime": reg, "seed": seed, "elapsed_s": round(elapsed, 2),
                           "N_train": int(len(train_idx)), "N_test": int(len(test_idx)),
                           "extracts": extracts.name, **metrics}
                    metrics_seeds.append(metrics)
                    rows.append(rec)
                    log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                    # Save best attention probe (single-head, batch) per task for dashboard
                    if arch == "attention" and reg == "batch" and metrics["auc"] > best_attn_auc:
                        best_attn_auc = metrics["auc"]
                        best_attn_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                accs = [m["acc"] for m in metrics_seeds]
                aucs = [m["auc"] for m in metrics_seeds if not math.isnan(m["auc"])]
                print(f"    {arch:13s} | {reg:11s} | acc {np.mean(accs):.3f}±{np.std(accs):.3f} | auc {(np.mean(aucs) if aucs else float('nan')):.3f}±{(np.std(aucs) if aucs else float('nan')):.3f}")
        # Save attention probe weights
        if best_attn_state is not None:
            torch.save({
                "state": best_attn_state,
                "task": task_name, "model_key": model_key,
                "extracts_dir": extracts.name,
                "test_idx": test_idx.tolist(),
                "train_idx": train_idx.tolist(),
                "d_model": d,
                "best_auc_seed_max": best_attn_auc,
            }, str(WEIGHTS / f"{task_name}_attention.pt"))

    log_f.close()
    print(f"\nFull metrics: {log_path}")
    print(f"Probe weights: {WEIGHTS}")


if __name__ == "__main__":
    main()
