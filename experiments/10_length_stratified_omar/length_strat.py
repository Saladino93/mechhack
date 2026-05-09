"""TODO 9a / Phase 10c — does probe AUC depend on prompt length?

For each task in {cyber_1, cyber_2, cyber_3}:
  - Use the existing selection + extracts at the task's best mean-pool layer.
  - Compute prompt char-length per sample (from the cyber-train jsonl).
  - Bucket prompts into tertiles: short / medium / long (within-task, since
    different tasks have different length distributions).
  - Run 5-fold StratifiedKFold; for each fold:
      * fit LR on the train rows;
      * predict on the test rows;
      * for each length bucket within the test fold, compute AUC over that bucket.
  - Aggregate across folds: mean ± std AUC per (task, length bucket).

Output:
  - results.json : per (task, bucket) AUC mean ± std + bucket boundaries (chars)
  - plot.png     : per-task AUC ± 1σ vs length-bucket

Headline question: does probe accuracy degrade on long prompts? Answer
motivates 9c (Kramar MultiMax + Rolling for long-prompt OOD).

CPU only.
"""
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
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import get_label_for_task, load_dataset  # noqa: E402

EXTRACTS = Path("/home/ubuntu/extracts/cyber_all_omar")
OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "cyber_1": {
        "selection": REPO_ROOT / "experiments" / "03_layer_sweep_omar" / "selection.json",
        "best_layer": 40,
    },
    "cyber_2": {
        "selection": REPO_ROOT / "experiments" / "06_cyber2_extract_omar" / "selection.json",
        "best_layer": 40,
    },
    "cyber_3": {
        "selection": REPO_ROOT / "experiments" / "07_cyber3_extract_omar" / "selection.json",
        "best_layer": 35,
    },
}
N_FOLDS = 5
SEED = 0


def pool_mean_at_layer(p, layer):
    ex = torch.load(str(p), weights_only=False)
    layer_idx_position = list(ex["layer_idxs"]).index(layer)
    residuals = ex["residuals"].float()
    mask = ex["attention_mask"].bool()
    n = int(mask.sum().item())
    if n == 0:
        return None
    m = mask.float().unsqueeze(0).unsqueeze(-1)
    feat = ((residuals * m).sum(dim=1) / n)[layer_idx_position].numpy()
    if not np.isfinite(feat).all():
        return None
    return feat


def main():
    train_samples = {s["sample_id"]: s for s in load_dataset("cyber", split="train")}

    out = {}
    for task_name, spec in TASKS.items():
        print(f"\n=== {task_name} (mean L{spec['best_layer']}) ===", flush=True)
        sel = json.loads(spec["selection"].read_text())
        ids = [s["sample_id"] for s in sel["samples"]]

        feats, labels, lengths, kept = [], [], [], []
        for sid in ids:
            p = EXTRACTS / f"{sid}.pt"
            if not p.exists():
                continue
            s = train_samples.get(sid)
            if s is None:
                continue
            lbl = get_label_for_task(s, task_name)
            if lbl is None:
                continue
            feat = pool_mean_at_layer(p, spec["best_layer"])
            if feat is None:
                continue
            feats.append(feat)
            labels.append(int(lbl))
            lengths.append(len(s["prompt"]))
            kept.append(sid)
        X = np.stack(feats).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        lengths = np.asarray(lengths)
        print(f"  loaded {len(y)} samples (pos={int((y==1).sum())}, neg={int((y==0).sum())})")
        print(f"  prompt char-length: min={lengths.min()}, p33={np.percentile(lengths,33):.0f}, "
              f"p67={np.percentile(lengths,67):.0f}, max={lengths.max()}")

        cuts = np.percentile(lengths, [33.33, 66.67])
        bucket = np.digitize(lengths, cuts)  # 0=short, 1=medium, 2=long

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        per_bucket = defaultdict(list)  # bucket -> list of (auc, n)
        global_aucs = []
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            clf.fit(X[tr_idx], y[tr_idx])
            p_te = clf.predict_proba(X[te_idx])[:, 1]
            try:
                global_aucs.append(float(roc_auc_score(y[te_idx], p_te)))
            except Exception:
                pass
            for b in (0, 1, 2):
                te_b = te_idx[bucket[te_idx] == b]
                if len(te_b) < 5 or len(set(y[te_b].tolist())) < 2:
                    continue
                p_b = clf.predict_proba(X[te_b])[:, 1]
                auc_b = float(roc_auc_score(y[te_b], p_b))
                per_bucket[b].append((auc_b, len(te_b)))

        bucket_summary = {}
        for b, name in [(0, "short"), (1, "medium"), (2, "long")]:
            entries = per_bucket[b]
            if not entries:
                continue
            aucs = np.array([a for a, _ in entries])
            ns = np.array([n for _, n in entries])
            bucket_summary[name] = {
                "auc_mean": float(aucs.mean()),
                "auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
                "n_per_fold_mean": float(ns.mean()),
                "fold_aucs": aucs.tolist(),
            }
            print(f"  {name:>7}: AUC={aucs.mean():.4f} ± {aucs.std(ddof=1) if len(aucs)>1 else 0:.4f}  "
                  f"(n/fold ≈ {ns.mean():.0f})")
        ga = np.array(global_aucs)
        bucket_summary["overall"] = {
            "auc_mean": float(ga.mean()),
            "auc_std": float(ga.std(ddof=1)) if len(ga) > 1 else 0.0,
        }
        out[task_name] = {
            "best_layer": spec["best_layer"],
            "n_samples": int(len(y)),
            "length_cuts_chars": cuts.tolist(),
            "buckets": bucket_summary,
        }

    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT/'results.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        x_labels = ["short", "medium", "long"]
        x = np.arange(3)
        width = 0.25
        for i, (task_name, t) in enumerate(out.items()):
            means = [t["buckets"].get(b, {}).get("auc_mean", np.nan) for b in x_labels]
            stds  = [t["buckets"].get(b, {}).get("auc_std", 0)       for b in x_labels]
            ax.errorbar(x + (i - 1) * width, means, yerr=stds, marker="o", capsize=4,
                        label=f"{task_name} (L{t['best_layer']})")
        ax.set_xticks(x); ax.set_xticklabels(x_labels)
        ax.set_ylabel("Test AUC (per length-bucket, 5-fold mean ± 1σ)")
        ax.set_title("Probe AUC by prompt-length tertile")
        ax.grid(alpha=0.3); ax.legend()
        ax.set_ylim(0.6, 1.02)
        fig.tight_layout()
        fig.savefig(OUT / "auc_by_length.png", dpi=150)
        plt.close(fig)
        print(f"wrote {OUT/'auc_by_length.png'}")
    except Exception as e:
        print(f"  [warn] plot failed: {e}")


if __name__ == "__main__":
    main()
