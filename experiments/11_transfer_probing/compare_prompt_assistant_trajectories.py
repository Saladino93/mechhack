"""Compare prompt-side and assistant-side probe scores for full-exchange extracts.

Use after `extract_exchange_activations.py`. The script computes three scores:

  * prompt_final: score at the final prompt token before assistant response
  * assistant_early: max SWiM score over the first N assistant tokens
  * assistant_full: max SWiM score over all assistant tokens

This helps separate failure modes:

  A. prompt already predicts the final behavior
  B. prompt ambiguous, assistant trajectory reveals behavior early
  C. prompt looks refusal-like, but assistant drifts compliance-like
  D. prompt looks compliance-like, but assistant flips refusal-like

Example:
    python experiments/04_failure_analysis/compare_prompt_assistant_trajectories.py \
      --probe_ckpt ./probes/constitutional_refusal_gemma/constitutional_probe.pt \
      --extracts_dir ./extracts/gemma4_31b_refusal_full_exchange \
      --dataset refusal_gemma4_31b \
      --task refusal_gemma4_31b \
      --out_dir ./analysis/refusal_prompt_vs_assistant
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Reuse helpers from depth_token_maps.py in the same folder.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from depth_token_maps import (  # type: ignore
    causal_swim_1d,
    get_binary_label,
    iter_pt_files,
    load_checkpoint,
    parse_dirs,
    residual_tensor,
    resolve_layer_positions,
    span_indices,
)


def token_logits_for_extract(ex: dict, ckpt: dict) -> np.ndarray:
    weight = ckpt["state_dict"]["weight"].float()
    bias = float(ckpt["state_dict"]["bias"].float().item())
    r_all = residual_tensor(ex)
    positions, _ = resolve_layer_positions(ex, ckpt)
    r = r_all[torch.tensor(positions, dtype=torch.long)]
    if r.shape[0] != weight.shape[0] or r.shape[-1] != weight.shape[-1]:
        raise ValueError(f"shape mismatch residual={tuple(r.shape)} weight={tuple(weight.shape)}")
    return (torch.einsum("ltd,ld->t", r.float(), weight.float()).numpy().astype(np.float32) + bias)


def max_swim_score(z: np.ndarray, idx: list[int], window_size: int) -> float:
    if not idx:
        return float("nan")
    local = z[idx]
    swim = causal_swim_1d(local, window_size)
    return float(np.max(swim)) if swim.size else float("nan")


def last_score(z: np.ndarray, idx: list[int]) -> float:
    if not idx:
        return float("nan")
    return float(z[idx[-1]])


def classify_case(prompt_score: float, early_score: float, full_score: float, threshold: float) -> str:
    p = prompt_score >= threshold
    e = early_score >= threshold
    f = full_score >= threshold
    if p == f:
        return "A_prompt_already_matches_full"
    if not p and e == f:
        return "B_assistant_reveals_early"
    if p and not f:
        return "C_refusal_like_prompt_drifts_compliance_like"
    if not p and f:
        return "D_compliance_like_prompt_flips_refusal_like"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe_ckpt", required=True)
    ap.add_argument("--extracts_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--window_size", type=int, default=None)
    ap.add_argument("--early_tokens", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint(Path(args.probe_ckpt))
    window_size = int(args.window_size or (ckpt.get("config", {}) or {}).get("window_size", 16))

    rows = []
    counts = {}
    for p in iter_pt_files(parse_dirs(args.extracts_dir)):
        if args.max_samples and len(rows) >= args.max_samples:
            break
        try:
            ex = torch.load(str(p), map_location="cpu", weights_only=False)
            y = get_binary_label(ex, args.dataset, args.task)
            if y is None:
                continue
            z = token_logits_for_extract(ex, ckpt)
            mask = ex.get("attention_mask", torch.ones(len(z), dtype=torch.bool)).bool()
            prompt_idx = span_indices(ex, "prompt", mask)
            assistant_idx = span_indices(ex, "assistant", mask)
            if not assistant_idx:
                # Not a full-exchange extract; skip rather than silently treating all tokens as assistant.
                continue
            early_idx = assistant_idx[: max(args.early_tokens, 1)]
            prompt_final = last_score(z, prompt_idx)
            assistant_early = max_swim_score(z, early_idx, window_size)
            assistant_full = max_swim_score(z, assistant_idx, window_size)
            case = classify_case(prompt_final, assistant_early, assistant_full, args.threshold)
            counts[case] = counts.get(case, 0) + 1
            rows.append({
                "sample_id": str(ex.get("sample_id", p.stem)),
                "label": int(y),
                "raw_label": ex.get("label"),
                "prompt_final": prompt_final,
                "assistant_early": assistant_early,
                "assistant_full": assistant_full,
                "case": case,
                "n_prompt_tokens": len(prompt_idx),
                "n_assistant_tokens": len(assistant_idx),
                "path": str(p),
            })
        except Exception as e:
            print(f"[warn] skipping {p.name}: {type(e).__name__}: {str(e)[:180]}", flush=True)

    csv_path = out_dir / "prompt_assistant_scores.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "sample_id", "label", "raw_label", "prompt_final", "assistant_early", "assistant_full",
            "case", "n_prompt_tokens", "n_assistant_tokens", "path"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "n": len(rows),
        "case_counts": counts,
        "threshold": args.threshold,
        "early_tokens": args.early_tokens,
        "window_size": window_size,
        "csv": str(csv_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Simple scatter plots.
    try:
        import matplotlib.pyplot as plt
        if rows:
            labels = np.array([r["label"] for r in rows])
            x = np.array([r["prompt_final"] for r in rows])
            y = np.array([r["assistant_full"] for r in rows])
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(x[labels == 0], y[labels == 0], alpha=0.55, label="label 0")
            ax.scatter(x[labels == 1], y[labels == 1], alpha=0.55, label="label 1")
            ax.axvline(args.threshold, linestyle="--", linewidth=1)
            ax.axhline(args.threshold, linestyle="--", linewidth=1)
            ax.set_xlabel("prompt final token logit")
            ax.set_ylabel("assistant full max-SWiM logit")
            ax.set_title("Prompt vs assistant probe scores")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "prompt_vs_assistant_full.png", dpi=160)
            plt.close(fig)
    except Exception as e:
        print(f"[warn] plotting failed: {type(e).__name__}: {str(e)[:160]}", flush=True)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
