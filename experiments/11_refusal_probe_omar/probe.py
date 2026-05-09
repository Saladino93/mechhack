"""Refusal probe + Arditi direction for Gemma 4-31B-it (layer 32).

Reads pre-extracted residuals from /home/ubuntu/extracts/gemma_refusal_L32/gemma4_31b/.
Each .pt file has: residuals shape (1, n_tokens, d_model) at fp16, attention_mask,
input_ids, label (0=complied, 1=refusal), layer_idxs=[32].

Outputs:
  - results.json      : per-pooling 5-fold CV AUC + Arditi direction stats
  - arditi_direction.npy : the unit-normed (mean refused − mean complied) vector
                          (mean-pooled features), the Level 2 attribution source
  - metrics.jsonl     : per-fold log

CPU only.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = ""

EXTRACTS_DIR = Path("/home/ubuntu/extracts/gemma_refusal_L32/gemma4_31b")
OUT_DIR = Path(__file__).parent
RESULTS_PATH = OUT_DIR / "results.json"
METRICS_LOG = OUT_DIR / "metrics.jsonl"
ARDITI_PATH = OUT_DIR / "arditi_direction.npy"

SEED = 0
N_FOLDS = 5


def atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def append_jsonl(path: Path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def pool_one(p):
    """Return (mean_pooled, last_pooled, label, n_tokens) at layer 32. fp16 → fp32.

    Returns ``None`` if the extract is unusable (zero tokens, NaN/Inf in
    pooled features). Bad files are silently skipped — the count is reported
    in the main loop.
    """
    ex = torch.load(str(p), weights_only=False)
    residuals = ex["residuals"]  # (1, n_tok, d) fp16
    if residuals.dim() == 3 and residuals.shape[0] == 1:
        residuals = residuals.squeeze(0)  # (n_tok, d)
    # Upcast to fp32 before summing — refusal prompts are long (up to ~8k
    # tokens), and fp16 sums overflow ±65504 with high probability.
    residuals = residuals.float()
    mask = ex["attention_mask"].bool().squeeze()  # (n_tok,)
    n = int(mask.sum().item())
    if n == 0:
        return None
    m_f = mask.float().unsqueeze(-1)
    mean_p = (residuals * m_f).sum(dim=0) / n  # (d,)
    last_idx = int(mask.nonzero().max().item())
    last_p = residuals[last_idx, :]  # (d,)
    mean_np = mean_p.numpy()
    last_np = last_p.numpy()
    if (not np.isfinite(mean_np).all()) or (not np.isfinite(last_np).all()):
        return None
    return mean_np, last_np, int(ex["label"]), n


def main():
    print("[refusal probe + Arditi direction] Gemma layer 32", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pt_paths = sorted(EXTRACTS_DIR.glob("*.pt"))
    print(f"  found {len(pt_paths)} extracts in {EXTRACTS_DIR}", flush=True)

    means, lasts, labels, ntoks, sample_ids = [], [], [], [], []
    t0 = time.time()
    for i, p in enumerate(pt_paths):
        try:
            r = pool_one(p)
        except Exception as e:
            print(f"  [warn] {p.stem}: {type(e).__name__}: {e}", flush=True)
            continue
        if r is None:
            continue
        mean_p, last_p, lbl, n = r
        means.append(mean_p); lasts.append(last_p); labels.append(lbl); ntoks.append(n)
        sample_ids.append(p.stem)
        if (i + 1) % 100 == 0:
            print(f"  pooled {i+1}/{len(pt_paths)} in {time.time()-t0:.1f}s", flush=True)

    X_mean = np.stack(means, axis=0).astype(np.float32)
    X_last = np.stack(lasts, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"  loaded {len(y)} samples (refusal={n_pos}, complied={n_neg}) "
          f"in {time.time()-t0:.1f}s", flush=True)

    # ---- Arditi refusal direction (mean-pooled features) ----
    mu_pos = X_mean[y == 1].mean(axis=0)
    mu_neg = X_mean[y == 0].mean(axis=0)
    raw_dir = mu_pos - mu_neg
    norm = float(np.linalg.norm(raw_dir))
    arditi_dir = raw_dir / norm
    np.save(ARDITI_PATH, arditi_dir)
    print(f"  saved Arditi direction to {ARDITI_PATH}  "
          f"(L2 norm of raw diff-of-means = {norm:.4f})", flush=True)

    # Quick sanity: project onto direction → AUC (should be high; this is the
    # 1-D version of the linear probe).
    proj_mean = X_mean @ arditi_dir
    proj_auc = float(roc_auc_score(y, proj_mean))
    print(f"  Arditi-direction projection AUC (mean-pool, no fitting): {proj_auc:.4f}", flush=True)

    # ---- 5-fold CV LR probe per pooling ----
    results = {
        "n_samples": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "layer": 32,
        "model": "Gemma-4-31B-it",
        "n_folds": N_FOLDS,
        "seed": SEED,
        "arditi_direction_norm": norm,
        "arditi_projection_auc": proj_auc,
        "per_pooling": {},
    }

    pool_arrays = {"mean": X_mean, "last_token": X_last}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for pname, X in pool_arrays.items():
        print(f"\n  === pooling = {pname} ===", flush=True)
        fold_metrics = []
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            t1 = time.time()
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            clf.fit(X[tr_idx], y[tr_idx])
            p_te = clf.predict_proba(X[te_idx])[:, 1]
            p_tr = clf.predict_proba(X[tr_idx])[:, 1]
            auc = float(roc_auc_score(y[te_idx], p_te))
            acc = float(((p_te > 0.5).astype(int) == y[te_idx]).mean())
            train_auc = float(roc_auc_score(y[tr_idx], p_tr))
            fm = {
                "fold": fold,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "auc": auc, "acc": acc, "train_auc": train_auc,
                "elapsed_s": round(time.time() - t1, 2),
            }
            fold_metrics.append(fm)
            append_jsonl(METRICS_LOG, {"pooling": pname, **fm})
            print(f"    fold {fold}: AUC={auc:.4f} acc={acc:.4f} "
                  f"train_auc={train_auc:.4f} | {fm['elapsed_s']:.1f}s", flush=True)

        aucs = np.asarray([m["auc"] for m in fold_metrics])
        accs = np.asarray([m["acc"] for m in fold_metrics])
        train_aucs = np.asarray([m["train_auc"] for m in fold_metrics])
        summary = {
            "auc_mean": float(aucs.mean()),
            "auc_std": float(aucs.std(ddof=1)),
            "auc_min": float(aucs.min()),
            "auc_max": float(aucs.max()),
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std(ddof=1)),
            "train_auc_mean": float(train_aucs.mean()),
            "fold_metrics": fold_metrics,
        }
        results["per_pooling"][pname] = summary
        print(f"  {pname:>10}: AUC={summary['auc_mean']:.4f} ± {summary['auc_std']:.4f} "
              f"| acc={summary['acc_mean']:.4f}", flush=True)

    atomic_write_json(RESULTS_PATH, results)
    print(f"\nwritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
