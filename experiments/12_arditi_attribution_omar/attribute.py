"""Level 2 Step 1 — per-token Arditi attribution for the 81 attribution_eval prompts.

For each refusal-attack prompt in datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl:

  1. Load the pre-extracted residuals at layer 32 (the layer the Arditi direction lives in).
  2. Compute per-token contribution to refusal: contrib[i] = residual[i] @ arditi_direction.
     Higher score → token i pushes the model toward refusal (per Arditi 2024).
  3. Save the per-token scores + token strings + the [POS|SCORE]token-marked prompt
     (the format `iterative_edit_agent.py`'s editor LLM expects).

The marker format is `[<position>|<+score>]<token>` with two-decimal scores. We
mark the top-K tokens by absolute attribution (default K=8, configurable).

Outputs (in this experiment's dir):
  - attributions.jsonl : one row per prompt with {sample_id, n_tokens, top_tokens,
                          marked_prompt, attribution_min/max, mean_abs_attribution}
  - summary.json       : aggregate stats across the 81 prompts
  - top_tokens.tsv     : flat TSV (sample_id, position, score, token) for inspection

CPU only.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARDITI_PATH = REPO_ROOT / "experiments" / "11_refusal_probe_omar" / "arditi_direction.npy"
EXTRACTS_DIR = Path("/home/ubuntu/extracts/gemma_refusal_L32/gemma4_31b")
EVAL_PATH = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attribution_eval.jsonl"
MODEL_PATH = Path("/home/ubuntu/models/Gemma-4-31B-it")
OUT_DIR = Path(__file__).parent

TOP_K = 8


def load_arditi():
    arr = np.load(ARDITI_PATH)
    print(f"  arditi direction shape: {arr.shape}, norm: {np.linalg.norm(arr):.4f}", flush=True)
    return arr.astype(np.float32)


def attribute_one(p, arditi_dir, tok):
    """Return (per_token_scores, decoded_tokens, n_kept) for one .pt extract."""
    ex = torch.load(str(p), weights_only=False)
    residuals = ex["residuals"]  # (1, n_tok, d) fp16
    if residuals.dim() == 3 and residuals.shape[0] == 1:
        residuals = residuals.squeeze(0)
    residuals = residuals.float().numpy()  # (n_tok, d)
    mask = ex["attention_mask"].bool().squeeze().numpy()
    input_ids = ex["input_ids"].squeeze().numpy() if "input_ids" in ex else None

    scores = residuals @ arditi_dir  # (n_tok,)
    # Zero out non-attended positions for clarity
    scores[~mask] = 0.0
    n_kept = int(mask.sum())

    decoded = None
    if input_ids is not None and tok is not None:
        try:
            decoded = [tok.decode([int(i)]) for i in input_ids]
        except Exception:
            decoded = None

    return scores, decoded, mask, n_kept


def make_marked_prompt(decoded_tokens, scores, mask, top_k):
    """Build the [pos|score]token format the editor LLM expects.

    Mark only the top-K positions by absolute score. Other positions are
    rendered as plain decoded tokens (concatenated).
    """
    masked_idxs = np.where(mask)[0]
    if len(masked_idxs) == 0:
        return ""
    abs_scores = np.abs(scores[masked_idxs])
    k = min(top_k, len(masked_idxs))
    top_local = np.argsort(-abs_scores)[:k]
    top_global = set(int(masked_idxs[j]) for j in top_local)

    parts = []
    for i in masked_idxs:
        i = int(i)
        tok = decoded_tokens[i] if decoded_tokens is not None else f"[{i}]"
        if i in top_global:
            parts.append(f"[{i}|{scores[i]:+.2f}]{tok}")
        else:
            parts.append(tok)
    return "".join(parts)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[Arditi attribution] computing per-token contributions for 81 attribution_eval prompts", flush=True)

    arditi = load_arditi()

    # Tokenizer for decoding (same as Gemma's)
    print("  loading tokenizer...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        print(f"  tokenizer ok: {type(tok).__name__}", flush=True)
    except Exception as e:
        print(f"  [warn] tokenizer load failed: {e}; will use position labels only", flush=True)
        tok = None

    # Load attribution_eval samples to get sample_ids in order
    samples = [json.loads(line) for line in EVAL_PATH.read_text().splitlines() if line.strip()]
    print(f"  loaded {len(samples)} attribution_eval samples", flush=True)

    rows = []
    top_tsv_lines = ["sample_id\tposition\tscore\ttoken"]
    skipped = 0
    for s in samples:
        sid = s["sample_id"]
        p = EXTRACTS_DIR / f"{sid}.pt"
        if not p.exists():
            print(f"  [skip] {sid}: extract missing", flush=True)
            skipped += 1
            continue

        scores, decoded, mask, n_kept = attribute_one(p, arditi, tok)
        if not np.isfinite(scores).all():
            print(f"  [skip] {sid}: non-finite attribution scores (fp16 overflow)", flush=True)
            skipped += 1
            continue

        marked = make_marked_prompt(decoded, scores, mask, TOP_K)

        masked_idxs = np.where(mask)[0]
        abs_in_scope = np.abs(scores[masked_idxs])
        top_local = np.argsort(-abs_in_scope)[:TOP_K]
        top_tokens = []
        for j in top_local:
            i = int(masked_idxs[j])
            tok_str = decoded[i] if decoded is not None else f"[{i}]"
            top_tokens.append({
                "position": i,
                "score": float(scores[i]),
                "token": tok_str,
            })
            top_tsv_lines.append(f"{sid}\t{i}\t{scores[i]:+.4f}\t{tok_str!r}")

        rows.append({
            "sample_id": sid,
            "is_refusal": s["is_refusal"],
            "n_tokens_in_scope": n_kept,
            "attribution_min": float(scores[mask].min()),
            "attribution_max": float(scores[mask].max()),
            "mean_abs_attribution": float(abs_in_scope.mean()),
            "top_tokens": top_tokens,
            "marked_prompt": marked,
        })

    # Write outputs
    out_jsonl = OUT_DIR / "attributions.jsonl"
    with out_jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {out_jsonl} ({len(rows)} rows, {skipped} skipped)", flush=True)

    out_tsv = OUT_DIR / "top_tokens.tsv"
    out_tsv.write_text("\n".join(top_tsv_lines))
    print(f"  wrote {out_tsv}", flush=True)

    # Summary
    if rows:
        all_max = np.array([r["attribution_max"] for r in rows])
        all_min = np.array([r["attribution_min"] for r in rows])
        all_mean = np.array([r["mean_abs_attribution"] for r in rows])
        n_refusal = sum(1 for r in rows if r["is_refusal"])
        n_complied = len(rows) - n_refusal
        summary = {
            "n_processed": len(rows),
            "n_skipped": skipped,
            "n_refusal": n_refusal,
            "n_complied": n_complied,
            "top_k_per_prompt": TOP_K,
            "attribution_max_overall_mean": float(all_max.mean()),
            "attribution_max_overall_max": float(all_max.max()),
            "attribution_min_overall_mean": float(all_min.mean()),
            "attribution_min_overall_min": float(all_min.min()),
            "mean_abs_attribution_across_prompts": float(all_mean.mean()),
            "arditi_direction_path": str(ARDITI_PATH),
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
        print("\n  === SUMMARY ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
