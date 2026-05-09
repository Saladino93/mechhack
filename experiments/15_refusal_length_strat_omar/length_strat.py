"""Length-stratified refusal probe AUC — does Gemma-31B refusal probe degrade
on long prompts?

If yes, that motivates implementing Kramar-2026 MultiMax + Rolling Attention
(TODO 9c). If no, Kramar is less critical for this dataset.

Method:
  - Load 832 Gemma refusal extracts at L32 (single-layer .pt files in
    /home/ubuntu/extracts/gemma_refusal_L32/gemma4_31b/).
  - Mean-pool over attended tokens → fp32 feature.
  - Bucket samples by prompt char-length (from attacks_full.jsonl) into tertiles.
  - 5-fold StratifiedKFold:
      * fit LR(C=1.0) on the train fold;
      * for each length bucket within the test fold, compute AUC and accuracy.
  - Aggregate across folds: mean ± std AUC per (length bucket).
  - Also report overall (across all lengths).

Output:
  - results.json   : per-bucket AUC mean ± std + bucket boundaries (chars + tokens)
  - auc_by_length.png : bar chart with error bars
  - feature_summary.json : pos/neg counts per bucket

CPU only.
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_L32/gemma4_31b")
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)
N_FOLDS = 5
SEED = 0


def main():
    print("[refusal length-strat] mean-pool L32 LR probe per length tertile", flush=True)

    # Load attack metadata for char lengths and labels
    attrs = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            attrs[r["sample_id"]] = r

    feats, labels, char_lens, ntoks, ids_kept = [], [], [], [], []
    skipped = 0
    pt_paths = sorted(EXTRACTS.glob("*.pt"))
    print(f"  loading {len(pt_paths)} extracts...", flush=True)
    for p in pt_paths:
        sid = p.stem
        meta = attrs.get(sid)
        if meta is None:
            skipped += 1
            continue
        ex = torch.load(str(p), weights_only=False)
        residuals = ex["residuals"]  # (1, n_tok, d) fp16
        if residuals.dim() == 3 and residuals.shape[0] == 1:
            residuals = residuals.squeeze(0)
        residuals = residuals.float()
        mask = ex["attention_mask"].bool().squeeze()
        n = int(mask.sum().item())
        if n == 0:
            skipped += 1
            continue
        m_f = mask.float().unsqueeze(-1)
        feat = ((residuals * m_f).sum(dim=0) / n).numpy()
        if not np.isfinite(feat).all():
            skipped += 1
            continue
        feats.append(feat)
        labels.append(int(ex["label"]))
        char_lens.append(len(meta["attack_prompt"]))
        ntoks.append(n)
        ids_kept.append(sid)

    X = np.stack(feats).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    char_lens = np.asarray(char_lens)
    ntoks = np.asarray(ntoks)
    print(f"  loaded {len(y)} samples ({skipped} skipped); pos={int((y==1).sum())} neg={int((y==0).sum())}", flush=True)

    cuts = np.percentile(char_lens, [33.33, 66.67])
    bucket = np.digitize(char_lens, cuts)  # 0=short, 1=medium, 2=long
    bucket_names = ["short", "medium", "long"]

    # report bucket boundaries + counts
    print(f"  char-length cuts: short ≤ {cuts[0]:.0f}, medium ≤ {cuts[1]:.0f}, long > {cuts[1]:.0f}", flush=True)
    feature_summary = {"buckets": {}}
    for b, name in enumerate(bucket_names):
        idx = np.where(bucket == b)[0]
        feature_summary["buckets"][name] = {
            "n_samples": int(len(idx)),
            "n_pos": int((y[idx] == 1).sum()),
            "n_neg": int((y[idx] == 0).sum()),
            "char_min": int(char_lens[idx].min()) if len(idx) else None,
            "char_max": int(char_lens[idx].max()) if len(idx) else None,
            "char_median": int(np.median(char_lens[idx])) if len(idx) else None,
            "ntok_min": int(ntoks[idx].min()) if len(idx) else None,
            "ntok_max": int(ntoks[idx].max()) if len(idx) else None,
            "ntok_median": int(np.median(ntoks[idx])) if len(idx) else None,
        }
        print(f"  {name:>6}: n={len(idx)} (pos={(y[idx]==1).sum()}, neg={(y[idx]==0).sum()}), "
              f"chars∈[{char_lens[idx].min()},{char_lens[idx].max()}] tokens∈[{ntoks[idx].min()},{ntoks[idx].max()}]", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    per_bucket_auc = defaultdict(list)
    per_bucket_acc = defaultdict(list)
    per_fold_global_auc = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr_idx], y[tr_idx])
        p_te = clf.predict_proba(X[te_idx])[:, 1]
        try:
            per_fold_global_auc.append(float(roc_auc_score(y[te_idx], p_te)))
        except Exception:
            pass
        for b, name in enumerate(bucket_names):
            te_b = te_idx[bucket[te_idx] == b]
            if len(te_b) < 5 or len(set(y[te_b].tolist())) < 2:
                continue
            p_b = clf.predict_proba(X[te_b])[:, 1]
            auc_b = float(roc_auc_score(y[te_b], p_b))
            acc_b = float(((p_b > 0.5).astype(int) == y[te_b]).mean())
            per_bucket_auc[name].append(auc_b)
            per_bucket_acc[name].append(acc_b)

    print(f"\n=== per-length AUC (5-fold mean ± 1σ) ===", flush=True)
    out = {
        "n_samples": int(len(y)),
        "char_length_cuts": cuts.tolist(),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "buckets": {},
    }
    for name in bucket_names:
        aucs = np.array(per_bucket_auc[name])
        accs = np.array(per_bucket_acc[name])
        if len(aucs) == 0:
            continue
        out["buckets"][name] = {
            "auc_mean": float(aucs.mean()),
            "auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "fold_aucs": aucs.tolist(),
            "fold_accs": accs.tolist(),
        }
        print(f"  {name:>6}: AUC={aucs.mean():.4f} ± {aucs.std(ddof=1) if len(aucs)>1 else 0:.4f}  "
              f"acc={accs.mean():.4f}", flush=True)

    ga = np.array(per_fold_global_auc)
    out["overall"] = {
        "auc_mean": float(ga.mean()) if len(ga) else None,
        "auc_std": float(ga.std(ddof=1)) if len(ga) > 1 else 0.0,
    }
    out["feature_summary"] = feature_summary
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    (OUT / "feature_summary.json").write_text(json.dumps(feature_summary, indent=2))
    print(f"  overall: AUC={out['overall']['auc_mean']:.4f}  (vs exp 11 mean-pool 0.9265 ± 0.0134)", flush=True)
    print(f"\nwrote {OUT/'results.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = bucket_names
        means = [out["buckets"][n]["auc_mean"] for n in names if n in out["buckets"]]
        stds  = [out["buckets"][n]["auc_std"]  for n in names if n in out["buckets"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=6, alpha=0.85,
                      color=["tab:green", "tab:olive", "tab:red"], edgecolor="black")
        for i, m in enumerate(means):
            ax.text(x[i], m + max(stds[i], 0.005) + 0.005, f"{m:.4f}", ha="center", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(
            [f"{n}\n({feature_summary['buckets'][n]['n_samples']} samples,\n"
             f"chars≤{feature_summary['buckets'][n]['char_max']:,})" for n in names])
        ax.set_ylabel("ROC-AUC (5-fold CV, ±1σ)")
        ax.set_ylim(0.5, 1.0)
        ax.set_title("Gemma 4-31B-it refusal probe (mean-pool L32) — AUC by prompt-length tertile\n"
                     "(Kramar 2026 prediction: long should drop. Result: ?)")
        if out["overall"]["auc_mean"]:
            ax.axhline(out["overall"]["auc_mean"], ls=":", color="black",
                       label=f"overall AUC={out['overall']['auc_mean']:.4f}")
            ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "auc_by_length.png", dpi=150)
        plt.close(fig)
        print(f"wrote {OUT/'auc_by_length.png'}")
    except Exception as e:
        print(f"  [warn] plot failed: {e}")


if __name__ == "__main__":
    main()
