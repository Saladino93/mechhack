"""Train a Constitutional Classifier++ style streaming linear probe.

Patched version for activation extracts produced by experiments/02_extract_activations.

Key behavior:
  - Reads activations from .pt files, not from extraction_metadata.json.
  - Uses the raw label saved in each .pt file.
  - Uses experiments/02_extract_activations/data.py to recover official train/test
    membership and cyber task mappings.
  - Supports --task cyber_1, cyber_2, cyber_3, or cyber/all/cyber_all.

Example:
    python experiments/03_constitutional_probe/train_constitutional_probe.py \
      --extracts_dir ./extracts/gemma4_31b_cyber_4layers_combined \
      --dataset cyber \
      --task cyber_3 \
      --layer_mode all \
      --window_size 16 \
      --tau 1.0 \
      --batch_size 1 \
      --eval_batch_size 1 \
      --out_dir ./probes/constitutional
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

# Local probe implementation from the same experiment/starter directory.
from constitutional_probe import (
    ProbeConfig,
    StreamingLinearProbe,
    pad_residual_batch,
    resolve_layer_positions,
    softmax_weighted_bce_loss,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _repo_root() -> Path:
    """Best-effort repo-root discovery from this script location."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "experiments" / "02_extract_activations" / "data.py").exists():
            return parent
        if (parent / "datasets").exists() and (parent / "experiments").exists():
            return parent
    # Fallback for starter_code layout.
    return here.parent.parent


def import_extraction_data_module():
    root = _repo_root()
    candidates = [
        root / "experiments" / "02_extract_activations",
        Path.cwd() / "experiments" / "02_extract_activations",
    ]
    for c in candidates:
        if (c / "data.py").exists():
            sys.path.insert(0, str(c))
            from data import load_dataset, get_label_for_task, get_tasks_for_dataset  # type: ignore
            return load_dataset, get_label_for_task, get_tasks_for_dataset
    raise ImportError(
        "Could not find experiments/02_extract_activations/data.py. "
        "Run from repo root or add that directory to PYTHONPATH."
    )


def load_extract_from_path(path: Path) -> dict:
    return torch.load(str(path), weights_only=False, map_location="cpu")


def load_extract(extracts_dir: Path, sample_id: str) -> dict:
    return load_extract_from_path(extracts_dir / f"{sample_id}.pt")


def residual_tensor(ex: dict) -> torch.Tensor:
    if "residuals" in ex:
        r = ex["residuals"]
    elif "middle_layer_all_tokens" in ex:
        r = ex["middle_layer_all_tokens"].unsqueeze(0)
    else:
        raise KeyError("extract missing `residuals` or `middle_layer_all_tokens`")
    if r.dim() == 2:
        r = r.unsqueeze(0)
    return r.float()


def iter_pt_files(extracts_dirs: list[Path]):
    seen: set[str] = set()
    for d in extracts_dirs:
        for p in sorted(d.glob("*.pt")):
            if p.name in seen:
                continue
            seen.add(p.name)
            yield p


def parse_extracts_dirs(arg: str) -> list[Path]:
    return [Path(x).expanduser() for x in arg.split(",") if x.strip()]


def expand_tasks(dataset_name: str | None, task: str | None, legacy_manifest: dict | None = None):
    """Return task names to run."""
    if dataset_name == "cyber":
        all_tasks = ["cyber_1", "cyber_2", "cyber_3"]
        if task is None or task in {"cyber", "all", "cyber_all"}:
            return all_tasks
        aliases = {
            "cyber1": "cyber_1",
            "cyber2": "cyber_2",
            "cyber3": "cyber_3",
            "cyber_1": "cyber_1",
            "cyber_2": "cyber_2",
            "cyber_3": "cyber_3",
        }
        if task not in aliases:
            raise ValueError(f"Unknown cyber task {task!r}. Use cyber_1, cyber_2, cyber_3, or cyber.")
        return [aliases[task]]

    if dataset_name and dataset_name.startswith("refusal"):
        return [task or dataset_name]

    # Legacy manifest mode keeps old exact task names.
    if task:
        return [task]
    return []


def build_dataset_rows_from_extracts(
    extracts_dirs: list[Path],
    dataset_name: str,
    task_name: str,
) -> tuple[list[dict], list[int], list[int]]:
    """Build rows from .pt extracts and official train/test membership from data.py."""
    load_dataset, get_label_for_task, _ = import_extraction_data_module()

    train_samples = load_dataset(dataset_name, split="train")
    test_samples = load_dataset(dataset_name, split="test")
    train_ids = {s["sample_id"] for s in train_samples}
    test_ids = {s["sample_id"] for s in test_samples}

    rows: list[dict] = []
    train_idx: list[int] = []
    test_idx: list[int] = []
    skipped_scope = 0
    skipped_split = 0
    skipped_bad = 0

    for pt in iter_pt_files(extracts_dirs):
        try:
            ex = load_extract_from_path(pt)
            sid = ex.get("sample_id", pt.stem)
            raw_label = ex.get("label")
            if raw_label is None:
                skipped_bad += 1
                continue
            # Reuse get_label_for_task by constructing the uniform sample dict it expects.
            y = get_label_for_task({"sample_id": sid, "label": raw_label}, task_name)
            if y is None:
                skipped_scope += 1
                continue
            row = {
                "sample_id": sid,
                "label": float(y),
                "path": str(pt),
                "source": {"sample_id": sid, "label": raw_label},
            }
            idx = len(rows)
            rows.append(row)
            if sid in train_ids:
                train_idx.append(idx)
            elif sid in test_ids:
                test_idx.append(idx)
            else:
                skipped_split += 1
        except Exception as e:
            skipped_bad += 1
            print(f"  [warn] skipping {pt}: {type(e).__name__}: {str(e)[:160]}", flush=True)

    if skipped_scope:
        print(f"  [info] skipped {skipped_scope} extracts outside task scope", flush=True)
    if skipped_split:
        print(f"  [warn] {skipped_split} rows had sample_id not present in official train/test", flush=True)
    if skipped_bad:
        print(f"  [warn] skipped {skipped_bad} unreadable/unlabeled extracts", flush=True)

    return rows, train_idx, test_idx


# ---- Legacy manifest mode, retained for old starter_code extracts ----

def legacy_task_specs(manifest: dict):
    cyber = manifest.get("cyber_samples", [])
    for model_key in ["gemma4_31b", "qwen36"]:
        ref = manifest.get("refusal_samples", {}).get(model_key, [])
        if ref:
            yield (f"refusal_{model_key}", model_key, ref, lambda s: 1.0 if s["is_refusal"] else 0.0)
        if cyber:
            for cls in ["prohibited", "high_risk_dual_use", "dual_use", "benign"]:
                pos = [s for s in cyber if s.get("label") == cls or s.get("category") == cls]
                neg_all = [s for s in cyber if (s.get("label") or s.get("category")) != cls]
                other = sorted({s.get("label") or s.get("category") for s in neg_all})
                if not pos or not other:
                    continue
                n_per = max(1, len(pos) // len(other))
                rng = random.Random(42 + abs(hash(cls)) % 1000)
                neg = []
                for c in other:
                    pool = [s for s in neg_all if (s.get("label") or s.get("category")) == c]
                    neg += rng.sample(pool, min(n_per, len(pool)))
                short = {"prohibited": "prohib", "high_risk_dual_use": "hdu", "dual_use": "du", "benign": "ben"}[cls]
                yield (f"cyber_{short}_vs_rest_{model_key}", model_key, pos + neg,
                       (lambda s, _cls=cls: 1.0 if (s.get("label") or s.get("category")) == _cls else 0.0))


def build_rows(samples: list[dict], label_fn: Callable[[dict], float], extracts_dir: Path) -> list[dict]:
    rows = []
    skipped = 0
    for s in samples:
        sid = s["sample_id"]
        path = extracts_dir / f"{sid}.pt"
        if not path.exists():
            skipped += 1
            continue
        rows.append({"sample_id": sid, "label": float(label_fn(s)), "path": str(path), "source": s})
    if skipped:
        print(f"  [warn] skipped {skipped} samples without extracts")
    return rows


def make_split(rows: list[dict], seed: int, test_frac: float = 0.3):
    y = np.array([int(r["label"]) for r in rows])
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_test = max(1, int(len(pos) * test_frac)) if len(pos) else 0
    n_neg_test = max(1, int(len(neg) * test_frac)) if len(neg) else 0
    test = np.concatenate([pos[:n_pos_test], neg[:n_neg_test]])
    train = np.concatenate([pos[n_pos_test:], neg[n_neg_test:]])
    rng.shuffle(train)
    rng.shuffle(test)
    return train.tolist(), test.tolist()


def make_val_split(train_idx: list[int], rows: list[dict], seed: int, val_frac: float):
    if val_frac <= 0 or len(train_idx) < 4:
        return train_idx, []
    y = np.array([int(rows[i]["label"]) for i in train_idx])
    pos = np.array([i for i in train_idx if int(rows[i]["label"]) == 1])
    neg = np.array([i for i in train_idx if int(rows[i]["label"]) == 0])
    rng = np.random.default_rng(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_val = max(1, int(len(pos) * val_frac)) if len(pos) > 1 else 0
    n_neg_val = max(1, int(len(neg) * val_frac)) if len(neg) > 1 else 0
    val = np.concatenate([pos[:n_pos_val], neg[:n_neg_val]])
    new_train = np.concatenate([pos[n_pos_val:], neg[n_neg_val:]])
    rng.shuffle(new_train)
    rng.shuffle(val)
    return new_train.tolist(), val.tolist()


def infer_shapes(rows: list[dict], layer_mode: str):
    for r in rows:
        ex = load_extract_from_path(Path(r["path"]))
        residuals = residual_tensor(ex)
        layer_positions = resolve_layer_positions(residuals.shape[0], layer_mode)
        if not layer_positions:
            raise ValueError(f"layer_mode={layer_mode!r} selected no layers")
        bad = [p for p in layer_positions if p < 0 or p >= residuals.shape[0]]
        if bad:
            raise ValueError(f"layer positions {bad} out of bounds for {residuals.shape[0]} extracted layers")
        source_layer_idxs = ex.get("layer_idxs", list(range(residuals.shape[0])))
        selected_source_layer_idxs = [source_layer_idxs[p] if p < len(source_layer_idxs) else p for p in layer_positions]
        return residuals.shape[0], residuals.shape[-1], layer_positions, selected_source_layer_idxs
    raise ValueError("no rows with extracts")


def load_batch(rows, indices, layer_positions):
    examples = []
    labels = []
    for i in indices:
        ex = load_extract_from_path(Path(rows[i]["path"]))
        residuals = residual_tensor(ex)
        mask = ex.get("attention_mask")
        if mask is None:
            mask = torch.ones(residuals.shape[1], dtype=torch.bool)
        examples.append((residuals, mask.bool()))
        labels.append(rows[i]["label"])
    x, mask = pad_residual_batch(examples, layer_positions)
    return x.to(DEVICE), mask.to(DEVICE), torch.tensor(labels, dtype=torch.float32, device=DEVICE)


def evaluate(model: StreamingLinearProbe, rows, indices, layer_positions, batch_size: int):
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for st in range(0, len(indices), batch_size):
            bi = indices[st:st + batch_size]
            if not bi:
                continue
            x, mask, y = load_batch(rows, bi, layer_positions)
            token_logits = model.forward_token_logits(x)
            loss = softmax_weighted_bce_loss(
                token_logits, y, mask,
                window_size=model.config.window_size,
                tau=model.config.tau,
            )
            seq_logits = model.sequence_logits(x, mask)
            total_loss += float(loss.item())
            n_batches += 1
            ys.extend(y.detach().cpu().numpy().astype(int).tolist())
            ps.extend(torch.sigmoid(seq_logits).detach().cpu().numpy().tolist())
    y_np = np.asarray(ys)
    p_np = np.asarray(ps)
    preds = (p_np > 0.5).astype(int)
    acc = float((preds == y_np).mean()) if len(y_np) else float("nan")
    try:
        from sklearn.metrics import roc_auc_score, f1_score
        auc = float(roc_auc_score(y_np, p_np)) if len(set(y_np.tolist())) > 1 else float("nan")
        f1 = float(f1_score(y_np, preds, zero_division=0))
    except Exception:
        auc, f1 = float("nan"), float("nan")
    return {"loss": total_loss / max(n_batches, 1), "acc": acc, "f1": f1, "auc": auc}


def train_one(args, rows, train_idx, val_idx, test_idx, d_model, layer_positions, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cfg = ProbeConfig(
        n_layers=len(layer_positions),
        d_model=d_model,
        window_size=args.window_size,
        tau=args.tau,
        score_mode=args.score_mode,
        ema_decay=args.ema_decay,
        layer_indices=tuple(layer_positions),
    )
    model = StreamingLinearProbe(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_metric = -float("inf")
    patience = args.patience
    rng = np.random.default_rng(seed)
    train_idx = list(train_idx)

    # Use validation for early stopping when available. Otherwise fall back to test, but warn.
    monitor_idx = val_idx if val_idx else test_idx
    monitor_name = "val" if val_idx else "test"
    if not val_idx:
        print("  [warn] no validation split; early stopping will monitor test metrics", flush=True)

    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        model.train()
        train_loss = 0.0
        n_batches = 0
        for st in range(0, len(train_idx), args.batch_size):
            bi = train_idx[st:st + args.batch_size]
            x, mask, y = load_batch(rows, bi, layer_positions)
            token_logits = model.forward_token_logits(x)
            loss = softmax_weighted_bce_loss(token_logits, y, mask, args.window_size, args.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_loss += float(loss.item())
            n_batches += 1
        ev = evaluate(model, rows, monitor_idx, layer_positions, args.eval_batch_size)
        metric = ev["auc"] if not math.isnan(ev["auc"]) else -ev["loss"]
        print(
            f"    seed={seed} epoch={epoch:02d} train_loss={train_loss/max(n_batches,1):.4f} "
            f"{monitor_name}_loss={ev['loss']:.4f} auc={ev['auc']:.4f} acc={ev['acc']:.4f}",
            flush=True,
        )
        if metric > best_metric + args.min_delta:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    final = evaluate(model, rows, test_idx, layer_positions, args.eval_batch_size)
    return final, model


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extracts_dir", required=True, help="Directory of .pt extracts, or comma-separated directories")
    ap.add_argument("--manifest", default=None, help="Legacy labeled JSON manifest. Not needed with --dataset.")
    ap.add_argument("--dataset", default=None, choices=["cyber", "refusal_gemma4_31b", "refusal_qwen36"],
                    help="Dataset name. Preferred mode for experiments/02_extract_activations outputs.")
    ap.add_argument("--out_dir", default="./probes/constitutional")
    ap.add_argument("--task", default=None, help="cyber_1, cyber_2, cyber_3, cyber/all, or refusal task")
    ap.add_argument("--layer_mode", default="all", help="all, middle, early, late, every2, every4, 0:65:4, or comma list")
    ap.add_argument("--window_size", type=int, default=16)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--score_mode", choices=["max_swim", "max_ema", "final_swim"], default="max_swim")
    ap.add_argument("--ema_decay", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--seeds", default="0", help="comma-separated seeds; e.g. 0,1,2")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Fraction of official train split used for validation")
    return ap.parse_args()


def main():
    args = parse_args()
    extracts_dirs = parse_extracts_dirs(args.extracts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    best_overall = None

    task_jobs = []
    if args.dataset:
        for task_name in expand_tasks(args.dataset, args.task):
            rows, train_idx, test_idx = build_dataset_rows_from_extracts(extracts_dirs, args.dataset, task_name)
            model_key = "unknown"
            # Try to infer model_key from first extract metadata.
            if rows:
                ex0 = load_extract_from_path(Path(rows[0]["path"]))
                model_key = ex0.get("model_key") or "unknown"
            task_jobs.append((task_name, model_key, rows, train_idx, test_idx))
    else:
        if not args.manifest:
            raise ValueError("Pass --dataset cyber/refusal_* for extracted .pt mode, or --manifest for legacy mode.")
        manifest = json.load(open(args.manifest))
        if "samples" in manifest and "cyber_samples" not in manifest and "refusal_samples" not in manifest:
            raise ValueError(
                "You passed extraction_metadata.json. That file has extraction run metadata only. "
                "Use --dataset cyber/refusal_* so the trainer reads labels from .pt files and splits from data.py."
            )
        for task_name, model_key, samples, label_fn in legacy_task_specs(manifest):
            if args.task and task_name != args.task:
                continue
            rows = build_rows(samples, label_fn, extracts_dirs[0])
            train_idx, test_idx = make_split(rows, seed=0)
            task_jobs.append((task_name, model_key, rows, train_idx, test_idx))

    if not task_jobs:
        raise ValueError("No tasks matched. For cyber use --dataset cyber --task cyber_1/cyber_2/cyber_3/cyber.")

    with metrics_path.open("w") as log_f:
        for task_name, model_key, rows, train_idx, test_idx in task_jobs:
            print(f"\n=== {task_name} ({model_key}, rows={len(rows)}) ===", flush=True)
            if not rows:
                print("  no rows with extracts")
                continue
            if not train_idx or not test_idx:
                raise ValueError(
                    f"Task {task_name} has train={len(train_idx)} test={len(test_idx)} rows. "
                    "Extract both train and test splits, or check sample_id matching."
                )

            y = np.array([int(r["label"]) for r in rows])
            train_y = np.array([int(rows[i]["label"]) for i in train_idx])
            test_y = np.array([int(rows[i]["label"]) for i in test_idx])
            print(
                f"  total={len(rows)} pos={int(y.sum())} neg={int((1-y).sum())} | "
                f"train={len(train_idx)} pos={int(train_y.sum())} neg={int((1-train_y).sum())} | "
                f"test={len(test_idx)} pos={int(test_y.sum())} neg={int((1-test_y).sum())}",
                flush=True,
            )

            train_idx2, val_idx = make_val_split(train_idx, rows, seed=0, val_frac=args.val_frac)
            n_input_layers, d_model, layer_positions, selected_source_layer_idxs = infer_shapes(rows, args.layer_mode)
            print(
                f"  extract_layers={n_input_layers} selected={layer_positions} "
                f"source_layers={selected_source_layer_idxs} d={d_model} val={len(val_idx)}",
                flush=True,
            )

            for seed in seeds:
                t0 = time.time()
                metrics, model = train_one(args, rows, train_idx2, val_idx, test_idx, d_model, layer_positions, seed)
                rec = {
                    "task": task_name,
                    "model_key": model_key,
                    "seed": seed,
                    "elapsed_s": round(time.time() - t0, 2),
                    "N_train": len(train_idx2),
                    "N_val": len(val_idx),
                    "N_test": len(test_idx),
                    "layer_mode": args.layer_mode,
                    "layer_positions": layer_positions,
                    "source_layer_idxs": selected_source_layer_idxs,
                    "window_size": args.window_size,
                    "tau": args.tau,
                    "score_mode": args.score_mode,
                    **metrics,
                }
                print(f"  RESULT seed={seed}: test_auc={metrics['auc']:.4f} test_acc={metrics['acc']:.4f} test_f1={metrics['f1']:.4f}", flush=True)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                rank_metric = metrics["auc"] if not math.isnan(metrics["auc"]) else -metrics["loss"]
                if best_overall is None or rank_metric > best_overall[0]:
                    best_overall = (rank_metric, task_name, model_key, seed, metrics, model, layer_positions, selected_source_layer_idxs, d_model)

    if best_overall is not None:
        _, task_name, model_key, seed, metrics, model, layer_positions, selected_source_layer_idxs, d_model = best_overall
        ckpt = {
            "format": "constitutional_streaming_linear_probe_v1",
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": {
                "n_layers": len(layer_positions),
                "d_model": d_model,
                "window_size": args.window_size,
                "tau": args.tau,
                "score_mode": args.score_mode,
                "ema_decay": args.ema_decay,
                "layer_positions": layer_positions,
                "source_layer_idxs": selected_source_layer_idxs,
                "layer_mode": args.layer_mode,
            },
            "task": task_name,
            "model_key": model_key,
            "seed": seed,
            "metrics": metrics,
        }
        out_path = out_dir / "constitutional_probe.pt"
        torch.save(ckpt, str(out_path))
        print(f"\nSaved best checkpoint: {out_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
