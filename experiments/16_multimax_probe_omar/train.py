"""Train Kramar 2026 Selected Probe (`RollingAttentionProbe`) end-to-end.

Usage:
    python experiments/16_multimax_probe_omar/train.py --task cyber_1 [--layer 40]
    python experiments/16_multimax_probe_omar/train.py --task cyber_2
    python experiments/16_multimax_probe_omar/train.py --task cyber_3
    python experiments/16_multimax_probe_omar/train.py --task refusal_gemma4_31b

Outputs (in --out_dir, default ./results/):
    <task>_rolling.pt   — best-fold model state_dict + metadata
    <task>_rolling.json — overall + per-fold AUC, length-tertile AUC, bootstrap CIs

This is a 5-fold CV training loop with full-batch (mini-batch) AdamW, paper
defaults from Appendix C: lr=1e-4, wd=3e-3, 1000 steps, d_hidden=100, H=10, w=10.

Length-tertile AUC uses the same percentile cuts (33.33 / 66.67 of char-length)
as exp 10 (cyber) and exp 15 (refusal).

CPU-only by default (the GPUs are busy with rollouts).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only

# Device selection: prefer CUDA if available. The unfold-based forward is
# ~10× faster on a single H100 than on this 26-core CPU box, and the model
# is tiny (5376 → 100 MLP + 10 heads × 100-dim q,v) so memory is not a concern.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CPU thread cap (only relevant if we fall back to CPU).
torch.set_num_threads(8)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
from data import get_label_for_task, load_dataset  # noqa: E402

from probes import RollingAttentionProbe, build_probe  # noqa: E402

# ----------------------------------------------------------------------
# Task registry
# ----------------------------------------------------------------------
CYBER_EXTRACTS = Path("/home/ubuntu/extracts/cyber_all_omar")
REFUSAL_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b")

TASKS = {
    "cyber_1": {
        "extracts_dir": CYBER_EXTRACTS,
        "selection": REPO_ROOT / "experiments" / "03_layer_sweep_omar" / "selection.json",
        "default_layer": 40,
        "dataset": "cyber",
        "is_refusal": False,
    },
    "cyber_2": {
        "extracts_dir": CYBER_EXTRACTS,
        "selection": REPO_ROOT / "experiments" / "06_cyber2_extract_omar" / "selection.json",
        "default_layer": 40,
        "dataset": "cyber",
        "is_refusal": False,
    },
    "cyber_3": {
        "extracts_dir": CYBER_EXTRACTS,
        "selection": REPO_ROOT / "experiments" / "07_cyber3_extract_omar" / "selection.json",
        "default_layer": 40,
        "dataset": "cyber",
        "is_refusal": False,
    },
    "refusal_gemma4_31b": {
        "extracts_dir": REFUSAL_EXTRACTS,
        "selection": None,
        "default_layer": 32,
        "dataset": "refusal_gemma4_31b",
        "is_refusal": True,
    },
}

# Hyperparameters from Kramar Appendix C (with the n_steps lowered to fit a
# CPU-only wallclock budget — we measured loss saturation by step ~300 on
# held-out smoke tests, so 600 is safely past convergence on the BCE loss).
HP = dict(
    d_hidden=100,
    n_heads=10,
    window_size=10,
    lr=1e-4,
    weight_decay=3e-3,
    n_steps=600,
    batch_size=6,
    seed=0,
    n_folds=5,
)

# Maximum tokens per sample. Paper handles up to 8189 tokens, but on CPU we
# truncate the front of very long prompts to keep wall-time reasonable. The
# refusal signal lives at the end of the prompt (final assistant turn), so
# tail-truncation preserves the discriminative content.
# We keep this large enough (4096) that the long-tertile bucket still
# meaningfully tests "long prompt" behaviour: ~half of refusal long-bucket
# samples are <= 4096 tokens (median 4272).
MAX_TOKENS = 4096


def log(msg: str):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_task_data(task: str, layer: int, max_samples: int | None = None):
    """Load (residuals, mask, label, char_len, sample_id) per sample.

    Returns a list of dicts. Filters extracts whose pooled features are non-finite,
    matching the cyber/refusal LR baselines (exp 10 / exp 15).
    """
    spec = TASKS[task]
    is_refusal = spec["is_refusal"]
    extracts_dir = spec["extracts_dir"]

    if is_refusal:
        # Refusal: load all 832 extracts, get labels and char-lengths from the
        # attacks_full.jsonl file (for length tertile cuts).
        attacks = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
        attrs = {json.loads(l)["sample_id"]: json.loads(l) for l in open(attacks)}
        sample_ids = sorted(p.stem for p in extracts_dir.glob("*.pt"))
        log(f"  found {len(sample_ids)} refusal extracts")
    else:
        # Cyber: read the per-task selection.json to know which sample_ids and
        # which binary label-mapping to use; cross-reference with cyber/train.jsonl
        # for the prompt char-length.
        sel = json.loads(spec["selection"].read_text())
        sample_ids = [s["sample_id"] for s in sel["samples"]]
        train_samples = {s["sample_id"]: s for s in load_dataset("cyber", split="train")}
        log(f"  task={task} selection has {len(sample_ids)} samples; loading prompts...")

    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]

    out = []
    skipped = 0
    n_truncated = 0
    t_load = time.time()
    for i_sid, sid in enumerate(sample_ids):
        p = extracts_dir / f"{sid}.pt"
        if not p.exists():
            skipped += 1
            continue
        # mmap=True is ~100x faster than full torch.load on these big files —
        # we only need 1/13 of the residual stream (cyber) or 1/1 (refusal),
        # so reading the whole 100MB file into RAM was pure overhead.
        ex = torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
        layer_idxs = list(ex["layer_idxs"])
        if layer not in layer_idxs:
            skipped += 1
            continue
        layer_pos = layer_idxs.index(layer)
        residuals = ex["residuals"]  # (n_layers, n_tok, d) fp16
        if residuals.dim() != 3:
            skipped += 1
            continue
        # .clone() forces the mmap'd slice into RAM so we don't keep the file
        # mapping alive forever (and so subsequent ops aren't IO-bound).
        residuals = residuals[layer_pos].clone()  # (n_tok, d) still fp16
        mask = ex["attention_mask"].bool().squeeze().clone()
        n_tok = int(mask.sum().item())
        if n_tok < 2:
            skipped += 1
            continue

        # Quick finiteness check on a mean pool (matches LR baseline filter).
        m_f = mask.float().unsqueeze(-1)
        feat_check = (residuals.float() * m_f).sum(dim=0) / max(n_tok, 1)
        if not torch.isfinite(feat_check).all().item():
            skipped += 1
            continue

        # Truncate to the LAST MAX_TOKENS attended tokens (refusal signal is at
        # the end; cyber prompts rarely exceed 4k tokens).
        if residuals.shape[0] > MAX_TOKENS:
            n_truncated += 1
            attended_idx = mask.nonzero(as_tuple=False).squeeze(-1)
            keep_idx = attended_idx[-MAX_TOKENS:]
            residuals = residuals[keep_idx]
            mask = torch.ones(residuals.shape[0], dtype=torch.bool)

        # Get label + char length
        if is_refusal:
            meta = attrs.get(sid)
            if meta is None:
                skipped += 1
                continue
            label = int(ex["label"])
            char_len = len(meta["attack_prompt"])
        else:
            sample = train_samples.get(sid)
            if sample is None:
                skipped += 1
                continue
            label = get_label_for_task(sample, task)
            if label is None:
                skipped += 1
                continue
            char_len = len(sample["prompt"])

        out.append(
            {
                "sample_id": sid,
                "residuals": residuals,  # fp16, (n_tok, d)
                "mask": mask,            # bool, (n_tok,)
                "label": int(label),
                "char_len": int(char_len),
            }
        )
        if (i_sid + 1) % 100 == 0:
            log(
                f"  ... loaded {i_sid+1}/{len(sample_ids)} ({time.time()-t_load:.1f}s, "
                f"trunc={n_truncated}, skip={skipped})"
            )

    log(
        f"  loaded {len(out)} samples (skipped {skipped}, truncated {n_truncated} to {MAX_TOKENS} tokens)"
    )
    n_pos = sum(1 for s in out if s["label"] == 1)
    n_neg = len(out) - n_pos
    log(f"  pos={n_pos}, neg={n_neg}")
    return out


# ----------------------------------------------------------------------
# Mini-batch padding
# ----------------------------------------------------------------------
def collate(batch):
    """Pad a list of samples to the max length in the batch."""
    Tmax = max(s["residuals"].shape[0] for s in batch)
    d = batch[0]["residuals"].shape[1]
    B = len(batch)
    x = torch.zeros(B, Tmax, d, dtype=torch.float32)
    mask = torch.zeros(B, Tmax, dtype=torch.bool)
    y = torch.zeros(B, dtype=torch.float32)
    for i, s in enumerate(batch):
        T = s["residuals"].shape[0]
        x[i, :T] = s["residuals"].float()
        mask[i, :T] = s["mask"]
        y[i] = float(s["label"])
    return x, mask, y


# ----------------------------------------------------------------------
# Training one fold
# ----------------------------------------------------------------------
def train_one_fold(
    train_data,
    val_data,
    d_model: int,
    fold: int,
    seed: int = 0,
    variant: str = "rolling",
):
    """Train the Selected Probe on a single fold; return (probe, val_logits).

    Uses mini-batch AdamW with the paper's HP. We cycle through `n_steps`
    randomly-sampled mini-batches from the training set (the "1000 steps"
    in the paper is over mini-batches of size 4-8; cyber selection has
    ~800 train rows, so we go ~6 epochs).
    """
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)

    probe = build_probe(
        variant=variant,
        d_model=d_model,
        d_hidden=HP["d_hidden"],
        n_heads=HP["n_heads"],
        window_size=HP["window_size"],
        tau=HP.get("tau", 0.1),
    ).to(DEVICE)
    opt = torch.optim.AdamW(
        probe.parameters(),
        lr=HP["lr"],
        weight_decay=HP["weight_decay"],
        betas=(0.9, 0.999),
    )
    probe.train()

    rng = np.random.default_rng(seed + fold)
    train_idx = np.arange(len(train_data))

    # Pre-sort training data by length so that mini-batches drawn from nearby
    # length-buckets have low padding waste. We still randomize batch picks,
    # but within a "length stratum" of ~50 samples — see below.
    train_lens = np.asarray([s["residuals"].shape[0] for s in train_data])
    order = np.argsort(train_lens)
    # Stratum size: large enough to retain randomness, small enough that the
    # ratio of max(T)/min(T) within a batch is ~1.5x.
    stratum_size = max(64, HP["batch_size"] * 6)
    strata = [order[i : i + stratum_size] for i in range(0, len(order), stratum_size)]

    losses = []
    for step in range(HP["n_steps"]):
        # Pick a stratum (uniformly), then a mini-batch from within it.
        stratum = strata[rng.integers(0, len(strata))]
        batch_ids = rng.choice(stratum, size=min(HP["batch_size"], len(stratum)), replace=False)
        batch = [train_data[i] for i in batch_ids]
        x, mask, y = collate(batch)
        x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        logits = probe(x, mask)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        # Clip — the max-over-windows aggregator can produce big gradients on
        # long prompts.
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % 200 == 0:
            recent = np.mean(losses[-50:])
            log(f"    fold {fold} step {step+1}/{HP['n_steps']}: loss(last50)={recent:.4f}")

    # Predict on validation set (no shuffle, batch-size 1 to avoid big padding).
    probe.eval()
    val_logits = np.zeros(len(val_data), dtype=np.float32)
    with torch.no_grad():
        bs = 4
        for i in range(0, len(val_data), bs):
            batch = val_data[i : i + bs]
            x, mask, _ = collate(batch)
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            l = probe(x, mask).cpu().numpy()
            val_logits[i : i + bs] = l
    return probe, val_logits, losses


# ----------------------------------------------------------------------
# Bootstrap AUC CIs (1000x)
# ----------------------------------------------------------------------
def bootstrap_auc_ci(y_true, y_score, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys, ps = y_true[idx], y_score[idx]
        if len(set(ys.tolist())) < 2:
            continue
        try:
            aucs.append(roc_auc_score(ys, ps))
        except Exception:
            continue
    if not aucs:
        return None, None
    aucs = np.asarray(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run(task: str, layer: int, out_dir: Path, max_samples: int = None,
        variant: str = "rolling"):
    spec = TASKS[task]
    log(f"\n=== task={task} layer={layer} ===")
    data = load_task_data(task, layer, max_samples=max_samples)
    if not data:
        raise RuntimeError(f"no data loaded for {task}")

    d_model = data[0]["residuals"].shape[1]
    y_all = np.asarray([s["label"] for s in data], dtype=np.int64)
    char_lens = np.asarray([s["char_len"] for s in data])
    cuts = np.percentile(char_lens, [33.33, 66.67])
    bucket = np.digitize(char_lens, cuts)  # 0=short, 1=medium, 2=long
    log(f"  d_model={d_model}, char-length cuts: {cuts.tolist()}")

    skf = StratifiedKFold(n_splits=HP["n_folds"], shuffle=True, random_state=HP["seed"])
    fold_aucs = []
    bucket_aucs = defaultdict(list)
    bucket_names = ["short", "medium", "long"]

    all_val_logits = np.zeros(len(data), dtype=np.float32)
    all_val_targets = np.zeros(len(data), dtype=np.int64)
    all_val_buckets = np.zeros(len(data), dtype=np.int64)

    best_fold_auc = -1.0
    best_state = None
    best_meta = None

    t_start = time.time()
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(data)), y_all)):
        log(f"\n  --- fold {fold+1}/{HP['n_folds']} ---")
        t0 = time.time()
        train_data = [data[i] for i in tr_idx]
        val_data = [data[i] for i in te_idx]

        probe, val_logits, losses = train_one_fold(
            train_data, val_data, d_model=d_model, fold=fold, seed=HP["seed"],
            variant=variant,
        )

        y_te = y_all[te_idx]
        try:
            auc = float(roc_auc_score(y_te, val_logits))
        except Exception:
            auc = float("nan")
        fold_aucs.append(auc)
        all_val_logits[te_idx] = val_logits
        all_val_targets[te_idx] = y_te
        all_val_buckets[te_idx] = bucket[te_idx]

        for b, name in enumerate(bucket_names):
            te_b = te_idx[bucket[te_idx] == b]
            if len(te_b) < 5 or len(set(y_all[te_b].tolist())) < 2:
                continue
            scores_b = all_val_logits[te_b]
            try:
                auc_b = float(roc_auc_score(y_all[te_b], scores_b))
            except Exception:
                continue
            bucket_aucs[name].append(auc_b)

        elapsed = time.time() - t0
        log(f"  fold {fold+1}: AUC={auc:.4f}  (val n={len(te_idx)}, {elapsed:.1f}s)")

        if auc > best_fold_auc:
            best_fold_auc = auc
            best_state = {k: v.detach().cpu() for k, v in probe.state_dict().items()}
            best_meta = {
                "fold": fold,
                "auc": auc,
                "tr_idx": tr_idx.tolist(),
                "te_idx": te_idx.tolist(),
            }

    total_elapsed = time.time() - t_start

    # Pooled AUC across all folds (using the concatenated test predictions).
    try:
        overall_auc_pooled = float(roc_auc_score(all_val_targets, all_val_logits))
    except Exception:
        overall_auc_pooled = float("nan")

    overall_auc_mean = float(np.mean(fold_aucs))
    overall_auc_std = float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0

    # Bootstrap 95% CI on the *pooled* CV predictions (matches the pooled AUC).
    ci_lo, ci_hi = bootstrap_auc_ci(all_val_targets, all_val_logits, n_boot=1000, seed=0)

    # Per-bucket pooled AUC + bootstrap CI.
    per_bucket_summary = {}
    for b, name in enumerate(bucket_names):
        idx = np.where(all_val_buckets == b)[0]
        if len(idx) < 5 or len(set(all_val_targets[idx].tolist())) < 2:
            continue
        try:
            auc_pooled_b = float(roc_auc_score(all_val_targets[idx], all_val_logits[idx]))
        except Exception:
            auc_pooled_b = float("nan")
        ci_lo_b, ci_hi_b = bootstrap_auc_ci(
            all_val_targets[idx], all_val_logits[idx], n_boot=1000, seed=0
        )
        # Per-fold mean+std for this bucket
        fold_aucs_b = bucket_aucs.get(name, [])
        per_bucket_summary[name] = {
            "n_samples": int(len(idx)),
            "auc_pooled": auc_pooled_b,
            "auc_pooled_ci95": [ci_lo_b, ci_hi_b],
            "auc_fold_mean": float(np.mean(fold_aucs_b)) if fold_aucs_b else None,
            "auc_fold_std": float(np.std(fold_aucs_b, ddof=1)) if len(fold_aucs_b) > 1 else 0.0,
            "fold_aucs": fold_aucs_b,
        }

    log("\n=== summary ===")
    log(f"  overall AUC (5-fold mean ± 1σ): {overall_auc_mean:.4f} ± {overall_auc_std:.4f}")
    log(f"  overall AUC (pooled CV preds):  {overall_auc_pooled:.4f}  CI95=[{ci_lo:.4f}, {ci_hi:.4f}]")
    for name in bucket_names:
        if name in per_bucket_summary:
            r = per_bucket_summary[name]
            log(
                f"  {name:>6}: AUC_pooled={r['auc_pooled']:.4f} "
                f"CI95=[{r['auc_pooled_ci95'][0]:.4f}, {r['auc_pooled_ci95'][1]:.4f}]  "
                f"(n={r['n_samples']})"
            )
    log(f"  total wall-time: {total_elapsed:.1f}s")

    # ----- save weights -----
    weights_path = out_dir / f"{task}_{variant}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "task": task,
            "layer": layer,
            "d_model": d_model,
            "hyperparameters": HP,
            "best_fold_meta": best_meta,
        },
        weights_path,
    )
    log(f"  saved weights -> {weights_path}")

    # ----- save metrics -----
    out = {
        "task": task,
        "layer": layer,
        "n_samples": len(data),
        "d_model": d_model,
        "hyperparameters": HP,
        "char_length_cuts": cuts.tolist(),
        "wall_time_seconds": float(total_elapsed),
        "overall": {
            "auc_fold_mean": overall_auc_mean,
            "auc_fold_std": overall_auc_std,
            "auc_pooled": overall_auc_pooled,
            "auc_pooled_ci95": [ci_lo, ci_hi],
            "fold_aucs": fold_aucs,
        },
        "by_length_tertile": per_bucket_summary,
    }
    metrics_path = out_dir / f"{task}_{variant}.json"
    metrics_path.write_text(json.dumps(out, indent=2))
    log(f"  saved metrics -> {metrics_path}")

    return out


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=list(TASKS) + ["all"],
        required=True,
        help="Task to train on. 'all' runs all 4 tasks sequentially.",
    )
    parser.add_argument("--layer", type=int, default=None, help="Override default layer.")
    parser.add_argument(
        "--variant",
        choices=["attention_kramar", "multimax", "rolling", "rolling_multimax"],
        default="rolling",
        help="Kramar architecture A/C/B/D (default: rolling = arch B = Selected Probe)",
    )
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--max_samples", type=int, default=None, help="(debug) cap to first N samples")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(TASKS) if args.task == "all" else [args.task]
    summary = {}
    for t in tasks:
        layer = args.layer if args.layer is not None else TASKS[t]["default_layer"]
        try:
            r = run(t, layer, args.out_dir, max_samples=args.max_samples,
                    variant=args.variant)
            summary[t] = {
                "auc_fold_mean": r["overall"]["auc_fold_mean"],
                "auc_pooled": r["overall"]["auc_pooled"],
                "auc_pooled_ci95": r["overall"]["auc_pooled_ci95"],
                "by_length_tertile": {
                    k: v["auc_pooled"] for k, v in r["by_length_tertile"].items()
                },
                "wall_time_seconds": r["wall_time_seconds"],
            }
        except Exception as e:
            log(f"[ERROR] task {t} failed: {type(e).__name__}: {e}")
            summary[t] = {"error": str(e)}

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"\n=== final summary written to {summary_path} ===")
    for t, r in summary.items():
        if "error" in r:
            log(f"  {t}: ERROR — {r['error']}")
            continue
        log(
            f"  {t}: overall AUC pooled={r['auc_pooled']:.4f} "
            f"CI95=[{r['auc_pooled_ci95'][0]:.4f}, {r['auc_pooled_ci95'][1]:.4f}]  "
            f"({r['wall_time_seconds']:.0f}s)"
        )


if __name__ == "__main__":
    cli()
