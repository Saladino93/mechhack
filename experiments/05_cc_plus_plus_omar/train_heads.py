"""Constitutional Classifiers++ probe-head comparison.

Implements and compares four probe heads on Gemma 4-31B-it residuals
(extracted in experiment 03), all evaluated on identical 80/20 stratified
splits of the same balanced 1000-sample cyber_1 (dual_use vs benign) selection.

Heads:
  A. concat   -- mean-pool every layer, concat (13 x d), sklearn LR + C sweep
  B. swim     -- per-token linear probe + sliding-window-mean smoothing then max
  C. softbce  -- per-token linear probe with softmax-weighted BCE training loss
                 (eval reported with both swim aggregation and plain max-over-tokens)
  D. baseline -- sklearn LR on mean-pooled residuals at the best single layer

CPU only. Reads .pt extracts from /home/ubuntu/extracts/03_layer_sweep_omar/.
Writes:
  - results.json        -- all numbers for plotting & docs
  - swim_traces.npz     -- per-token logits before/after smoothing for two examples

Polls until all 1000 extracts in selection.json are present (extraction may
still be running on GPU in another process when this script is launched).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import get_label_for_task, load_dataset

# ── paths ─────────────────────────────────────────────────────
EXTRACTS_DIR = Path("/home/ubuntu/extracts/03_layer_sweep_omar")
SELECTION = REPO_ROOT / "experiments" / "03_layer_sweep_omar" / "selection.json"
OUT_PATH = Path(__file__).parent / "results.json"
TRACES_PATH = Path(__file__).parent / "swim_traces.npz"

# ── hyperparameters ──────────────────────────────────────────
TASK = "cyber_1"
SPLIT_SEED = 0
TEST_FRAC = 0.2
N_BOOT = 1000
SEEDS = [0, 1, 2, 3, 4]
SWIM_WINDOW = 16
BEST_LAYER = 40                        # from exp 03
C_GRID = [0.01, 0.1, 1.0, 10.0]
TORCH_EPOCHS = 50
TORCH_BATCH = 32
TORCH_LR = 1e-3
DEVICE = torch.device("cpu")
torch.set_num_threads(min(16, torch.get_num_threads()))


# ── extract polling ──────────────────────────────────────────
def wait_for_extracts():
    sel = json.loads(SELECTION.read_text())
    needed = {row["sample_id"] for row in sel["samples"]}
    last_have = -1
    while True:
        have = {p.stem for p in EXTRACTS_DIR.glob("*.pt")}
        missing = needed - have
        if not missing:
            print(f"All {len(needed)} extracts present.", flush=True)
            return sel
        if len(have) != last_have:
            print(f"Waiting for extracts: have {len(have & needed)}/{len(needed)} "
                  f"({len(missing)} missing). Sleep 20s.", flush=True)
            last_have = len(have)
        time.sleep(20)


# ── feature loading ──────────────────────────────────────────
def load_features(sel):
    """Load per-sample features and per-token tensors at BEST_LAYER.

    Returns dict with:
        X_concat       : (N, n_layers * d)        fp32  (multi-layer pooled, concat)
        X_pool_best    : (N, d)                    fp32  (pooled at BEST_LAYER)
        per_token      : list of (N_tok_i, d)      fp32 tensors at BEST_LAYER
                         (already masked: only valid tokens kept)
        sample_ids     : list[str]
        labels_str     : list[str] (raw category)
        layer_idxs     : list[int]
    """
    selected_ids = [row["sample_id"] for row in sel["samples"]]
    layer_idxs = None
    X_concat = []
    X_pool_best = []
    per_token = []
    sample_ids = []
    labels_str = []

    for i, sid in enumerate(selected_ids):
        p = EXTRACTS_DIR / f"{sid}.pt"
        ex = torch.load(str(p), weights_only=False)
        residuals = ex["residuals"].float()              # (n_layers, N_tok, d)
        mask = ex["attention_mask"].bool()
        if mask.sum().item() == 0:
            continue
        if layer_idxs is None:
            layer_idxs = list(ex["layer_idxs"])
        # Mean pool per layer over masked tokens
        valid = mask.float().unsqueeze(0).unsqueeze(-1)   # (1, N_tok, 1)
        pooled = (residuals * valid).sum(dim=1) / mask.sum().item()  # (n_layers, d)
        X_concat.append(pooled.flatten().numpy().astype(np.float32))
        # Best-layer pooled
        best_li = layer_idxs.index(BEST_LAYER)
        X_pool_best.append(pooled[best_li].numpy().astype(np.float32))
        # Per-token at best layer, masked-only
        tok = residuals[best_li][mask].contiguous()       # (N_valid, d)
        per_token.append(tok)
        sample_ids.append(ex["sample_id"])
        labels_str.append(ex["label"])
        if (i + 1) % 100 == 0:
            print(f"  loaded {i+1}/{len(selected_ids)}", flush=True)

    return {
        "X_concat": np.stack(X_concat),
        "X_pool_best": np.stack(X_pool_best),
        "per_token": per_token,
        "sample_ids": sample_ids,
        "labels_str": labels_str,
        "layer_idxs": layer_idxs,
    }


def get_labels(sample_ids, task):
    samples_by_id = {s["sample_id"]: s for s in load_dataset("cyber", split="train")}
    y = np.full(len(sample_ids), -1, dtype=np.int64)
    in_scope = np.zeros(len(sample_ids), dtype=bool)
    for i, sid in enumerate(sample_ids):
        s = samples_by_id.get(sid)
        if s is None:
            continue
        lbl = get_label_for_task(s, task)
        if lbl is None:
            continue
        y[i] = lbl
        in_scope[i] = True
    return y, in_scope


# ── metric helpers ───────────────────────────────────────────
def bootstrap_auc_ci(y_true, y_score, n_boot=N_BOOT, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(y_true[idx].tolist())) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def metric_pack(y_true, score):
    """Compute AUC, accuracy (0.5 thresh on prob), F1, BCE, plus AUC bootstrap CI."""
    auc = float(roc_auc_score(y_true, score))
    pred = (score > 0.5).astype(int)
    acc = float((pred == y_true).mean())
    f1 = float(f1_score(y_true, pred))
    # log_loss expects probabilities; clip score to [eps, 1-eps]
    eps = 1e-7
    p = np.clip(score, eps, 1 - eps)
    bce = float(log_loss(y_true, p, labels=[0, 1]))
    auc_lo, auc_hi = bootstrap_auc_ci(y_true, score)
    return {
        "auc": auc, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
        "acc": acc, "f1": f1, "bce": bce,
    }


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ── HEAD A: multi-layer concat probe ─────────────────────────
def head_concat(X, y, idx_tr, idx_te):
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_te, y_te = X[idx_te], y[idx_te]

    # Hold-out validation split inside train for C selection (random_state fixed).
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=SPLIT_SEED, stratify=y_tr
    )
    best = (-np.inf, None, None)
    cs = {}
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X_fit, y_fit)
        p_val = clf.predict_proba(X_val)[:, 1]
        auc_val = float(roc_auc_score(y_val, p_val))
        cs[C] = auc_val
        if auc_val > best[0]:
            best = (auc_val, C, None)

    # Refit on full train at chosen C
    C_star = best[1]
    final = LogisticRegression(C=C_star, max_iter=2000, solver="lbfgs")
    final.fit(X_tr, y_tr)
    p_te = final.predict_proba(X_te)[:, 1]
    out = metric_pack(y_te, p_te)
    out["C_chosen"] = C_star
    out["C_grid_val_auc"] = {str(k): v for k, v in cs.items()}
    return out


# ── HEAD D: linear baseline at best single layer ─────────────
def head_baseline(X, y, idx_tr, idx_te):
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_te, y_te = X[idx_te], y[idx_te]
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]
    return metric_pack(y_te, p_te)


# ── per-token torch probes (HEAD B + C) ──────────────────────
class PerTokenLinearProbe(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, 1, bias=True)

    def forward(self, h):
        # h: (N_tok, d) -> logits (N_tok,)
        return self.lin(h).squeeze(-1)


def make_batches(indices, batch_size, rng):
    order = rng.permutation(len(indices))
    for i in range(0, len(order), batch_size):
        yield [indices[j] for j in order[i:i + batch_size]]


def swim_smooth(logits, window=SWIM_WINDOW):
    """Centred 1D mean filter, length-preserving (pads with edge values)."""
    if len(logits) <= 1:
        return logits.copy()
    half = window // 2
    pad_left = max(0, half)
    pad_right = max(0, window - 1 - half)
    padded = np.pad(logits, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    # smoothed length = len(padded) - window + 1 = len(logits)
    return smoothed.astype(logits.dtype)


def aggregate_predictions(probe, per_token, indices, agg, device=DEVICE):
    """Score a set of samples with a torch probe and a chosen aggregation.

    agg: "swim_max"  -> sliding-window mean (window=16) then max
         "max"       -> plain max over tokens
         "mean"      -> mean over tokens
    Returns probabilities (sigmoid of aggregated logit) of shape (len(indices),).
    """
    probe.eval()
    out = np.zeros(len(indices), dtype=np.float32)
    with torch.no_grad():
        for k, i in enumerate(indices):
            h = per_token[i].to(device)
            l = probe(h).cpu().numpy().astype(np.float64)
            if agg == "max":
                agg_l = float(l.max()) if len(l) else 0.0
            elif agg == "mean":
                agg_l = float(l.mean()) if len(l) else 0.0
            elif agg == "swim_max":
                s = swim_smooth(l, SWIM_WINDOW)
                agg_l = float(s.max()) if len(s) else 0.0
            else:
                raise ValueError(agg)
            out[k] = sigmoid(agg_l)
    return out


def train_torch_probe(per_token, y, idx_tr, idx_val, d, seed,
                      loss_kind="mean_bce", agg_for_val="swim_max"):
    """Train a per-token linear probe; early-stop on validation AUC.

    loss_kind:
      "mean_bce"  -> mean BCE over masked tokens, sample-level label broadcast
      "softw_bce" -> softmax(per-token logits)-weighted BCE
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    probe = PerTokenLinearProbe(d).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=TORCH_LR)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    best_val_auc = -np.inf
    best_state = None
    history = []

    for epoch in range(TORCH_EPOCHS):
        probe.train()
        epoch_loss = 0.0
        n_seen = 0
        for batch in make_batches(idx_tr, TORCH_BATCH, rng):
            opt.zero_grad()
            losses = []
            for i in batch:
                h = per_token[i].to(DEVICE)
                if h.shape[0] == 0:
                    continue
                yi = float(y[i])
                logits = probe(h)                               # (N_tok,)
                target = torch.full_like(logits, yi)
                per_tok_loss = bce(logits, target)              # (N_tok,)
                if loss_kind == "mean_bce":
                    sample_loss = per_tok_loss.mean()
                elif loss_kind == "softw_bce":
                    w = torch.softmax(logits.detach(), dim=0)   # weights as in paper
                    sample_loss = (w * per_tok_loss).sum()
                else:
                    raise ValueError(loss_kind)
                losses.append(sample_loss)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * len(losses)
            n_seen += len(losses)
        avg_loss = epoch_loss / max(1, n_seen)

        # Validation AUC with target aggregator
        val_score = aggregate_predictions(probe, per_token, idx_val, agg_for_val)
        val_auc = float(roc_auc_score(y[idx_val], val_score))
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_auc": val_auc})
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    return probe, best_val_auc, history


def head_swim(per_token, y, idx_tr, idx_te, d, seed):
    # Inner train/val split for early stopping
    y_tr_arr = y[idx_tr]
    fit_idx, val_idx = train_test_split(
        idx_tr, test_size=0.2, random_state=seed, stratify=y_tr_arr
    )
    probe, best_val, history = train_torch_probe(
        per_token, y, fit_idx, val_idx, d, seed,
        loss_kind="mean_bce", agg_for_val="swim_max"
    )
    p_te = aggregate_predictions(probe, per_token, idx_te, "swim_max")
    res = metric_pack(y[idx_te], p_te)
    res["best_val_auc"] = float(best_val)
    res["history"] = history
    return res, probe


def head_softbce(per_token, y, idx_tr, idx_te, d, seed):
    y_tr_arr = y[idx_tr]
    fit_idx, val_idx = train_test_split(
        idx_tr, test_size=0.2, random_state=seed, stratify=y_tr_arr
    )
    probe, best_val, history = train_torch_probe(
        per_token, y, fit_idx, val_idx, d, seed,
        loss_kind="softw_bce", agg_for_val="swim_max"
    )
    # Two aggregations
    p_te_swim = aggregate_predictions(probe, per_token, idx_te, "swim_max")
    p_te_max = aggregate_predictions(probe, per_token, idx_te, "max")
    res_swim = metric_pack(y[idx_te], p_te_swim)
    res_max = metric_pack(y[idx_te], p_te_max)
    return {
        "best_val_auc": float(best_val),
        "history": history,
        "swim_max": res_swim,
        "max_only": res_max,
    }, probe


# ── plotting traces helper ───────────────────────────────────
def save_swim_traces(probe, per_token, y, idx_te):
    """Pick one positive and one negative test sample and save raw/smoothed logits."""
    pos = next((i for i in idx_te if y[i] == 1), None)
    neg = next((i for i in idx_te if y[i] == 0), None)
    traces = {}
    probe.eval()
    with torch.no_grad():
        for label, i in [("pos", pos), ("neg", neg)]:
            if i is None:
                continue
            h = per_token[i].to(DEVICE)
            l = probe(h).cpu().numpy().astype(np.float32)
            s = swim_smooth(l.astype(np.float64), SWIM_WINDOW).astype(np.float32)
            traces[f"{label}_raw"] = l
            traces[f"{label}_smooth"] = s
            traces[f"{label}_index"] = np.array([i], dtype=np.int64)
    np.savez(TRACES_PATH, **traces)
    print(f"Saved traces to {TRACES_PATH}", flush=True)


# ── main ─────────────────────────────────────────────────────
def main():
    sel = wait_for_extracts()

    print("Loading + featurising 1000 extracts (CPU)...", flush=True)
    feats = load_features(sel)
    sample_ids = feats["sample_ids"]
    layer_idxs = feats["layer_idxs"]
    d = feats["X_pool_best"].shape[1]
    print(f"N_samples={len(sample_ids)} d_model={d} n_layers={len(layer_idxs)} "
          f"X_concat shape={feats['X_concat'].shape}", flush=True)

    y, in_scope = get_labels(sample_ids, TASK)
    keep = np.where(in_scope)[0]
    y = y[keep]
    X_concat = feats["X_concat"][keep]
    X_pool_best = feats["X_pool_best"][keep]
    per_token = [feats["per_token"][i] for i in keep]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"In-scope: n={len(y)}, pos={n_pos}, neg={n_neg}", flush=True)

    # Single 80/20 stratified split, mirroring exp 03
    all_idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(
        all_idx, test_size=TEST_FRAC, random_state=SPLIT_SEED, stratify=y
    )
    print(f"Train: {len(idx_tr)}  Test: {len(idx_te)}", flush=True)

    results = {
        "task": TASK,
        "extracts_dir": str(EXTRACTS_DIR),
        "n_samples_in_scope": int(len(y)),
        "n_pos": n_pos, "n_neg": n_neg,
        "best_layer_for_pertoken": BEST_LAYER,
        "layer_idxs": layer_idxs,
        "d_model": int(d),
        "split_seed": SPLIT_SEED,
        "test_frac": TEST_FRAC,
        "swim_window": SWIM_WINDOW,
        "torch_seeds": SEEDS,
        "torch_epochs": TORCH_EPOCHS,
        "torch_lr": TORCH_LR,
        "torch_batch": TORCH_BATCH,
        "C_grid": C_GRID,
        "heads": {},
    }

    # HEAD D: baseline (single seed; sklearn solver is deterministic)
    print("\n=== HEAD D: linear + mean-pool @ layer 40 (baseline) ===", flush=True)
    res_base = head_baseline(X_pool_best, y, idx_tr, idx_te)
    print(f"  AUC={res_base['auc']:.3f} CI=[{res_base['auc_ci_lo']:.3f},"
          f"{res_base['auc_ci_hi']:.3f}] acc={res_base['acc']:.3f} "
          f"f1={res_base['f1']:.3f} bce={res_base['bce']:.3f}", flush=True)
    results["heads"]["D_baseline"] = res_base

    # HEAD A: multi-layer concat
    print("\n=== HEAD A: multi-layer concat (sklearn LR + C sweep) ===", flush=True)
    res_concat = head_concat(X_concat, y, idx_tr, idx_te)
    print(f"  C_chosen={res_concat['C_chosen']}  AUC={res_concat['auc']:.3f} "
          f"CI=[{res_concat['auc_ci_lo']:.3f},{res_concat['auc_ci_hi']:.3f}] "
          f"acc={res_concat['acc']:.3f} f1={res_concat['f1']:.3f} "
          f"bce={res_concat['bce']:.3f}", flush=True)
    print(f"  val AUC by C: {res_concat['C_grid_val_auc']}", flush=True)
    results["heads"]["A_concat"] = res_concat

    # Free big array
    del X_concat
    feats["X_concat"] = None

    # HEAD B + C: per-token torch probes, 5 seeds
    swim_runs = []
    softbce_runs = []
    swim_probe_for_traces = None
    for seed in SEEDS:
        print(f"\n=== seed {seed}: HEAD B (SWiM, mean BCE) ===", flush=True)
        res_b, probe_b = head_swim(per_token, y, idx_tr, idx_te, d, seed)
        print(f"  best_val_auc={res_b['best_val_auc']:.3f}  "
              f"test AUC={res_b['auc']:.3f} "
              f"CI=[{res_b['auc_ci_lo']:.3f},{res_b['auc_ci_hi']:.3f}] "
              f"acc={res_b['acc']:.3f} f1={res_b['f1']:.3f} "
              f"bce={res_b['bce']:.3f}", flush=True)
        swim_runs.append(res_b)
        if seed == SEEDS[0]:
            swim_probe_for_traces = probe_b

        print(f"\n=== seed {seed}: HEAD C (softmax-weighted BCE) ===", flush=True)
        res_c, _ = head_softbce(per_token, y, idx_tr, idx_te, d, seed)
        rs = res_c["swim_max"]
        rm = res_c["max_only"]
        print(f"  swim_max  AUC={rs['auc']:.3f} acc={rs['acc']:.3f} f1={rs['f1']:.3f}", flush=True)
        print(f"  max_only  AUC={rm['auc']:.3f} acc={rm['acc']:.3f} f1={rm['f1']:.3f}", flush=True)
        softbce_runs.append(res_c)

    # Aggregate seed runs
    def summarise(runs, key_fn=lambda r: r):
        aucs = [key_fn(r)["auc"] for r in runs]
        accs = [key_fn(r)["acc"] for r in runs]
        f1s = [key_fn(r)["f1"] for r in runs]
        bces = [key_fn(r)["bce"] for r in runs]
        return {
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)),   "f1_std": float(np.std(f1s)),
            "bce_mean": float(np.mean(bces)), "bce_std": float(np.std(bces)),
            "auc_ci_lo_mean": float(np.mean([key_fn(r)["auc_ci_lo"] for r in runs])),
            "auc_ci_hi_mean": float(np.mean([key_fn(r)["auc_ci_hi"] for r in runs])),
        }

    results["heads"]["B_swim"] = {
        "per_seed": [{k: v for k, v in r.items() if k != "history"} for r in swim_runs],
        "summary": summarise(swim_runs),
    }
    results["heads"]["C_softbce_swim_max"] = {
        "per_seed": [{k: v for k, v in r["swim_max"].items()} for r in softbce_runs],
        "summary": summarise(softbce_runs, key_fn=lambda r: r["swim_max"]),
    }
    results["heads"]["C_softbce_max_only"] = {
        "per_seed": [{k: v for k, v in r["max_only"].items()} for r in softbce_runs],
        "summary": summarise(softbce_runs, key_fn=lambda r: r["max_only"]),
    }

    # Save SWiM traces (using seed-0 probe) for the smoothing-effect plot
    if swim_probe_for_traces is not None:
        save_swim_traces(swim_probe_for_traces, per_token, y, idx_te)

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
