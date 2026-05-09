#!/usr/bin/env python3
"""Score a cyber-trained probe on the Gemma/Qwen refusal dataset.

Purpose
-------
This is the transfer experiment:

    train probe on cyber labels  -> freeze it -> score refusal prompts

The resulting score should be interpreted as "cyber-positive-like" or
"prohibited-like" depending on the cyber task used for training, while the
refusal label is the model behavior label (is_refusal). This lets us find:

    high cyber score + compliance  = harmfulness visible, model complied
    low cyber score  + refusal     = model refused for reasons not captured by cyber probe

Supported checkpoints
---------------------
1. Constitutional trainer checkpoint:
      format == "constitutional_streaming_linear_probe_v1"
      saved by experiments/03_constitutional_probe/train_constitutional_probe.py

2. Starter/default attention checkpoint:
      saved by starter_code/train_probe.py as *_attention.pt
      contains keys: state, d_model, task, model_key
      Note: that old trainer used the first residual layer in each extract.

Inputs
------
--refusal_extracts_dir can be one dir or comma-separated dirs containing .pt
activation extracts. Each .pt should have residuals, attention_mask, sample_id,
and label where label is is_refusal (bool/int).

Outputs
-------
- transfer_scores.csv
- metrics.json
- mismatch_high_cyber_compliance.csv
- mismatch_low_cyber_refusal.csv
- figures/*.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_dirs(s: str) -> list[Path]:
    return [Path(x).expanduser() for x in s.split(",") if x.strip()]


def iter_pt_files(dirs: Sequence[Path]) -> Iterable[Path]:
    seen: set[str] = set()
    for d in dirs:
        for p in sorted(d.glob("*.pt")):
            if p.name in seen:
                continue
            seen.add(p.name)
            yield p


def load_extract(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def residual_tensor(ex: dict) -> torch.Tensor:
    if "residuals" in ex:
        r = ex["residuals"]
    elif "middle_layer_all_tokens" in ex:
        r = ex["middle_layer_all_tokens"].unsqueeze(0)
    else:
        raise KeyError("extract missing residuals or middle_layer_all_tokens")
    if r.dim() == 2:
        r = r.unsqueeze(0)
    return r.float()


def get_mask(ex: dict, T: int) -> torch.Tensor:
    if "attention_mask" in ex:
        m = ex["attention_mask"].bool().view(-1)
        return m[:T]
    return torch.ones(T, dtype=torch.bool)


def bool_label(x) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(bool(x))
    if isinstance(x, str):
        y = x.strip().lower()
        if y in {"true", "1", "yes", "refusal", "refused"}:
            return 1
        if y in {"false", "0", "no", "compliance", "complied"}:
            return 0
    raise ValueError(f"cannot interpret refusal label: {x!r}")


# ---------------- Constitutional checkpoint scoring ----------------

@dataclass
class ConstitutionalConfig:
    n_layers: int
    d_model: int
    window_size: int = 16
    score_mode: str = "max_swim"
    ema_decay: float | None = None
    source_layer_idxs: list[int] | None = None
    layer_positions: list[int] | None = None


def sliding_window_mean(token_logits: torch.Tensor, mask: torch.Tensor, window_size: int):
    """token_logits: (B,T), mask: (B,T). Returns zbar, score_mask."""
    mask = mask.to(device=token_logits.device, dtype=torch.bool)
    B, T = token_logits.shape
    M = max(int(window_size), 1)
    z = token_logits.masked_fill(~mask, 0.0)
    counts_raw = mask.float()
    csum = F.pad(torch.cumsum(z, dim=1), (1, 0))
    ccnt = F.pad(torch.cumsum(counts_raw, dim=1), (1, 0))
    ends = torch.arange(T, device=token_logits.device) + 1
    starts = (ends - M).clamp_min(0)
    sums = csum[:, ends] - csum[:, starts]
    counts = (ccnt[:, ends] - ccnt[:, starts]).clamp_min(1.0)
    zbar = sums / counts
    lengths = mask.long().sum(dim=1)
    pos = torch.arange(T, device=token_logits.device).unsqueeze(0).expand(B, T)
    full_window_positions = pos >= (M - 1)
    final_positions_for_short = pos == (lengths.clamp_min(1) - 1).unsqueeze(1)
    score_mask = mask & torch.where(lengths.unsqueeze(1) >= M, full_window_positions, final_positions_for_short)
    return zbar, score_mask


def ema_sequence_score(token_logits: torch.Tensor, mask: torch.Tensor, window_size: int, ema_decay: float | None):
    B, T = token_logits.shape
    decay = float(ema_decay) if ema_decay is not None else (max(int(window_size), 1) - 1) / max(int(window_size), 1)
    decay = min(max(decay, 0.0), 0.9999)
    ema = torch.zeros(B, device=token_logits.device, dtype=token_logits.dtype)
    seen = torch.zeros(B, device=token_logits.device, dtype=torch.bool)
    best = torch.full((B,), -torch.inf, device=token_logits.device, dtype=token_logits.dtype)
    mask = mask.bool()
    for t in range(T):
        active = mask[:, t]
        z_t = token_logits[:, t]
        ema = torch.where(seen, decay * ema + (1.0 - decay) * z_t, z_t)
        seen = seen | active
        best = torch.where(active, torch.maximum(best, ema), best)
    return best


class ConstitutionalProbeScorer:
    def __init__(self, ckpt: dict):
        cfg = ckpt["config"]
        sd = ckpt["state_dict"]
        self.weight = sd["weight"].float().cpu()  # (L,D)
        self.bias = sd["bias"].float().cpu().reshape(())
        self.config = ConstitutionalConfig(
            n_layers=int(cfg.get("n_layers", self.weight.shape[0])),
            d_model=int(cfg.get("d_model", self.weight.shape[1])),
            window_size=int(cfg.get("window_size", 16)),
            score_mode=str(cfg.get("score_mode", "max_swim")),
            ema_decay=cfg.get("ema_decay", None),
            source_layer_idxs=list(cfg["source_layer_idxs"]) if cfg.get("source_layer_idxs") is not None else None,
            layer_positions=list(cfg["layer_positions"]) if cfg.get("layer_positions") is not None else None,
        )
        self.task = ckpt.get("task")
        self.model_key = ckpt.get("model_key")
        self.kind = "constitutional_streaming_linear_probe_v1"

    def _select_residual_layers(self, residuals: torch.Tensor, ex_layer_idxs: list[int] | None):
        # Prefer mapping by original source layer ids, so refusal extracts can have a different
        # tensor order as long as they include the trained source layers.
        if self.config.source_layer_idxs is not None and ex_layer_idxs is not None:
            pos_by_layer = {int(layer): i for i, layer in enumerate(ex_layer_idxs)}
            missing = [l for l in self.config.source_layer_idxs if int(l) not in pos_by_layer]
            if missing:
                raise ValueError(
                    f"extract missing trained source layers {missing}; extract has layer_idxs={ex_layer_idxs}. "
                    "Regenerate refusal activations with the same layer spec used for cyber training."
                )
            positions = [pos_by_layer[int(l)] for l in self.config.source_layer_idxs]
        elif self.config.layer_positions is not None:
            positions = self.config.layer_positions
        else:
            positions = list(range(self.weight.shape[0]))
        return residuals.index_select(0, torch.as_tensor(positions, dtype=torch.long))

    def score_extract(self, ex: dict) -> dict:
        r = residual_tensor(ex)
        ex_layer_idxs = ex.get("layer_idxs")
        if ex_layer_idxs is not None:
            ex_layer_idxs = [int(x) for x in ex_layer_idxs]
        r = self._select_residual_layers(r, ex_layer_idxs)
        if r.shape[0] != self.weight.shape[0] or r.shape[-1] != self.weight.shape[-1]:
            raise ValueError(f"shape mismatch: residual={tuple(r.shape)} weight={tuple(self.weight.shape)}")
        T = r.shape[1]
        mask = get_mask(ex, T)
        # token layer contributions: (L,T)
        contrib = torch.einsum("ltd,ld->lt", r.float(), self.weight.float())
        token_logits = contrib.sum(dim=0) + self.bias
        tl = token_logits.unsqueeze(0)
        m = mask.unsqueeze(0)
        if self.config.score_mode == "max_ema":
            seq_logit = ema_sequence_score(tl, m, self.config.window_size, self.config.ema_decay)[0]
            zbar, score_mask = sliding_window_mean(tl, m, self.config.window_size)
        else:
            zbar, score_mask = sliding_window_mean(tl, m, self.config.window_size)
            if self.config.score_mode == "final_swim":
                valid = zbar[0, score_mask[0]]
                seq_logit = valid[-1] if valid.numel() else token_logits[mask][-1]
            else:
                seq_logit = zbar.masked_fill(~score_mask, -torch.inf).max(dim=1).values[0]
        valid_logits = token_logits[mask]
        valid_zbar = zbar[0, score_mask[0]]
        return {
            "score_logit": float(seq_logit.item()),
            "score_prob": float(torch.sigmoid(seq_logit).item()),
            "final_token_logit": float(valid_logits[-1].item()) if valid_logits.numel() else float("nan"),
            "mean_token_logit": float(valid_logits.mean().item()) if valid_logits.numel() else float("nan"),
            "max_token_logit": float(valid_logits.max().item()) if valid_logits.numel() else float("nan"),
            "mean_swim_logit": float(valid_zbar.mean().item()) if valid_zbar.numel() else float("nan"),
            "n_tokens": int(mask.sum().item()),
        }


# ---------------- Starter/default attention checkpoint scoring ----------------

class StarterAttentionScorer:
    """Scores checkpoints saved by starter_code/train_probe.py.

    That script's get_full_tokens(ex) chose residuals[0] when extracts contained many
    layers, and the saved attention checkpoint does not record a layer id. We mirror that.
    """
    def __init__(self, ckpt: dict, layer_position: int = 0):
        self.state = ckpt["state"]
        self.q = self.state["q"].float().cpu()
        self.head_weight = self.state["head.weight"].float().cpu().view(-1)
        self.head_bias = self.state["head.bias"].float().cpu().view(-1)[0]
        self.layer_position = int(layer_position)
        self.task = ckpt.get("task")
        self.model_key = ckpt.get("model_key")
        self.kind = "starter_attention_probe"

    def score_extract(self, ex: dict) -> dict:
        r = residual_tensor(ex)
        if self.layer_position >= r.shape[0]:
            raise ValueError(f"requested layer_position={self.layer_position}, but residual has {r.shape[0]} layers")
        x = r[self.layer_position].float()  # (T,D)
        T = x.shape[0]
        mask = get_mask(ex, T)
        d = x.shape[-1]
        if d != self.q.numel():
            raise ValueError(f"shape mismatch: residual d={d}, q d={self.q.numel()}")
        attn_logits = (x @ self.q) / math.sqrt(d)
        attn_logits = attn_logits.masked_fill(~mask, -torch.inf)
        alpha = torch.softmax(attn_logits, dim=0)
        pooled = torch.einsum("t,td->d", alpha, x)
        score = torch.dot(pooled, self.head_weight) + self.head_bias
        return {
            "score_logit": float(score.item()),
            "score_prob": float(torch.sigmoid(score).item()),
            "final_token_logit": float((x[mask][-1] @ self.head_weight + self.head_bias).item()) if mask.any() else float("nan"),
            "mean_token_logit": float(((x[mask] @ self.head_weight + self.head_bias).mean()).item()) if mask.any() else float("nan"),
            "max_token_logit": float(((x[mask] @ self.head_weight + self.head_bias).max()).item()) if mask.any() else float("nan"),
            "mean_swim_logit": float("nan"),
            "n_tokens": int(mask.sum().item()),
        }


def load_scorer(path: Path, ckpt_type: str, starter_layer_position: int):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if ckpt_type == "auto":
        if isinstance(ckpt, dict) and ckpt.get("format") == "constitutional_streaming_linear_probe_v1":
            ckpt_type = "constitutional"
        elif isinstance(ckpt, dict) and "state" in ckpt and "q" in ckpt.get("state", {}):
            ckpt_type = "starter_attention"
        else:
            raise ValueError(
                "Could not infer checkpoint type. Pass --ckpt_type constitutional or starter_attention."
            )
    if ckpt_type == "constitutional":
        return ConstitutionalProbeScorer(ckpt)
    if ckpt_type == "starter_attention":
        return StarterAttentionScorer(ckpt, layer_position=starter_layer_position)
    raise ValueError(f"unsupported ckpt_type: {ckpt_type}")


# ---------------- Metrics, output, plots ----------------


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y.astype(int).tolist())) < 2:
            return float("nan")
        return float(roc_auc_score(y.astype(int), scores))
    except Exception:
        return float("nan")


def safe_average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
        if len(set(y.astype(int).tolist())) < 2:
            return float("nan")
        return float(average_precision_score(y.astype(int), scores))
    except Exception:
        return float("nan")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_plots(rows: list[dict], out_dir: Path):
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except Exception as e:
        print(f"[warn] skipping plots: {e}", file=sys.stderr)
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray([r["is_refusal"] for r in rows], dtype=int)
    scores = np.asarray([r["score_logit"] for r in rows], dtype=float)
    probs = np.asarray([r["score_prob"] for r in rows], dtype=float)
    lengths = np.asarray([r["n_tokens"] for r in rows], dtype=float)

    plt.figure(figsize=(8, 5))
    plt.hist(scores[y == 0], bins=30, alpha=0.6, label="complied / non-refusal")
    plt.hist(scores[y == 1], bins=30, alpha=0.6, label="refused")
    plt.xlabel("frozen cyber probe score logit")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "score_hist_by_refusal_label.png", dpi=180)
    plt.close()

    if len(set(y.tolist())) > 1:
        fpr, tpr, _ = roc_curve(y, scores)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("FPR predicting is_refusal from cyber score")
        plt.ylabel("TPR")
        plt.tight_layout()
        plt.savefig(fig_dir / "roc_refusal_from_cyber_score.png", dpi=180)
        plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(lengths[y == 0], scores[y == 0], s=14, alpha=0.65, label="complied / non-refusal")
    plt.scatter(lengths[y == 1], scores[y == 1], s=14, alpha=0.65, label="refused")
    plt.xlabel("prompt tokens")
    plt.ylabel("frozen cyber probe score logit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "score_vs_length.png", dpi=180)
    plt.close()

    # Quadrant counts as a bar chart if thresholded.
    quadrants = [r.get("quadrant", "") for r in rows]
    if all(q for q in quadrants):
        names = [
            "high_cyber_refusal",
            "high_cyber_compliance",
            "low_cyber_refusal",
            "low_cyber_compliance",
        ]
        vals = [quadrants.count(n) for n in names]
        plt.figure(figsize=(9, 4))
        plt.bar(names, vals)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(fig_dir / "quadrant_counts.png", dpi=180)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe_ckpt", required=True, help="Cyber-trained checkpoint path")
    ap.add_argument("--ckpt_type", default="auto", choices=["auto", "constitutional", "starter_attention"],
                    help="Checkpoint format. auto usually works.")
    ap.add_argument("--starter_layer_position", type=int, default=0,
                    help="For starter_attention checkpoints, layer position to use. Old trainer used 0.")
    ap.add_argument("--refusal_extracts_dir", required=True,
                    help="Dir or comma-separated dirs of refusal_gemma4_31b .pt extracts")
    ap.add_argument("--out_dir", required=True, help="Output analysis directory")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Cyber score logit threshold for high/low quadrants. Default 0.0.")
    ap.add_argument("--top_k", type=int, default=50,
                    help="Rows to save in each mismatch CSV")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Optional cap for smoke tests")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = load_scorer(Path(args.probe_ckpt), args.ckpt_type, args.starter_layer_position)
    dirs = parse_dirs(args.refusal_extracts_dir)

    rows: list[dict] = []
    errors: list[dict] = []
    for i, pt in enumerate(iter_pt_files(dirs)):
        if args.max_samples and len(rows) >= args.max_samples:
            break
        try:
            ex = load_extract(pt)
            raw_label = ex.get("label")
            if raw_label is None:
                raise ValueError("extract missing `label`; refusal extracts should store is_refusal")
            is_refusal = bool_label(raw_label)
            s = scorer.score_extract(ex)
            high = s["score_logit"] >= args.threshold
            if high and is_refusal:
                quadrant = "high_cyber_refusal"
            elif high and not is_refusal:
                quadrant = "high_cyber_compliance"
            elif (not high) and is_refusal:
                quadrant = "low_cyber_refusal"
            else:
                quadrant = "low_cyber_compliance"
            rows.append({
                "sample_id": str(ex.get("sample_id", pt.stem)),
                "path": str(pt),
                "raw_refusal_label": raw_label,
                "is_refusal": int(is_refusal),
                "score_logit": s["score_logit"],
                "score_prob": s["score_prob"],
                "final_token_logit": s["final_token_logit"],
                "mean_token_logit": s["mean_token_logit"],
                "max_token_logit": s["max_token_logit"],
                "mean_swim_logit": s["mean_swim_logit"],
                "n_tokens": s["n_tokens"],
                "quadrant": quadrant,
            })
        except Exception as e:
            errors.append({"path": str(pt), "error": f"{type(e).__name__}: {str(e)}"})
        if (i + 1) % 100 == 0:
            print(f"scanned {i+1} files | scored={len(rows)} errors={len(errors)}", flush=True)

    if not rows:
        raise SystemExit(f"No rows scored. First errors: {errors[:3]}")

    scores = np.asarray([r["score_logit"] for r in rows], dtype=float)
    probs = np.asarray([r["score_prob"] for r in rows], dtype=float)
    y = np.asarray([r["is_refusal"] for r in rows], dtype=int)
    refused = scores[y == 1]
    complied = scores[y == 0]
    quadrants = {q: int(sum(1 for r in rows if r["quadrant"] == q)) for q in sorted({r["quadrant"] for r in rows})}
    metrics = {
        "probe_ckpt": str(Path(args.probe_ckpt)),
        "probe_kind": getattr(scorer, "kind", "unknown"),
        "probe_task": getattr(scorer, "task", None),
        "probe_model_key": getattr(scorer, "model_key", None),
        "refusal_extracts_dir": args.refusal_extracts_dir,
        "n_scored": len(rows),
        "n_errors": len(errors),
        "n_refusal": int(y.sum()),
        "n_compliance": int((1 - y).sum()),
        "threshold_logit": args.threshold,
        "auc_predicting_is_refusal_from_cyber_score": safe_auc(y, scores),
        "average_precision_predicting_is_refusal_from_cyber_score": safe_average_precision(y, scores),
        "mean_score_refusal": float(refused.mean()) if len(refused) else float("nan"),
        "mean_score_compliance": float(complied.mean()) if len(complied) else float("nan"),
        "median_score_refusal": float(np.median(refused)) if len(refused) else float("nan"),
        "median_score_compliance": float(np.median(complied)) if len(complied) else float("nan"),
        "cohens_d_refusal_minus_compliance": cohens_d(refused, complied),
        "pearson_score_vs_length": float(np.corrcoef(scores, np.asarray([r["n_tokens"] for r in rows], dtype=float))[0, 1]) if len(rows) > 1 else float("nan"),
        "quadrants": quadrants,
    }

    write_csv(out_dir / "transfer_scores.csv", rows)
    # Mismatch 1: high cyber score but complied/non-refusal. Sort descending by cyber score.
    high_cyber_compliance = [r for r in rows if r["quadrant"] == "high_cyber_compliance"]
    high_cyber_compliance.sort(key=lambda r: r["score_logit"], reverse=True)
    write_csv(out_dir / "mismatch_high_cyber_compliance.csv", high_cyber_compliance[: args.top_k])
    # Mismatch 2: low cyber score but refused. Sort ascending by cyber score.
    low_cyber_refusal = [r for r in rows if r["quadrant"] == "low_cyber_refusal"]
    low_cyber_refusal.sort(key=lambda r: r["score_logit"])
    write_csv(out_dir / "mismatch_low_cyber_refusal.csv", low_cyber_refusal[: args.top_k])
    if errors:
        write_csv(out_dir / "errors.csv", errors)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    make_plots(rows, out_dir)

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
