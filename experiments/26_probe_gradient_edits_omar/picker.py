"""Probe-gradient edit picker — generate edits aware of the FULL probe direction.

For each prompt in attribution_eval.jsonl:
  1. Load per-token residuals at a fixed layer L (e.g. 40 for mean L40 LR).
  2. For a given fitted probe (coef, bias) at layer L:
       per_token_score = (1 / n_tokens) * coef · residual[t]
     Tokens with the largest per_token_score are pulling the probe toward
     refusal. (For mean-pool features, the probe score = sum of
     per_token_scores; deleting a high-score token directly drops the
     pooled activation and therefore the probe score.)
  3. Map top-K tokens → containing words (or sentences).
  4. Generate 3 minimal-edit candidates per (prompt, probe):
       - delete_top1_word_grad
       - delete_top3_words_grad
       - delete_top1_sentence_grad
  5. Save to gradient_edits.jsonl.

Difference from exp 19 (Arditi-attribution edits): exp 19 used the Arditi
direction = mean(refused) - mean(complied). The probe coef is a different
direction, optimized for discrimination — uses the WHOLE probe gradient,
not the linear-classifier-friendly mean-difference. For probes that beat
Arditi (LR last-tok L45, multi-layer concat), this should produce more
targeted edits.

Usage:
    python picker.py --probe lr_mean_L40   # default
    python picker.py --probe lr_last_L45
    python picker.py --probe lr_multi_concat
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)
REPO_ROOT = HERE.parent.parent

PROBES_NPZ = REPO_ROOT / "experiments" / "24_robustness_omar" / "fitted_probes.npz"
PROBE_SPECS = REPO_ROOT / "experiments" / "24_robustness_omar" / "probe_specs.json"
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
EVAL_SET = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attribution_eval.jsonl"
REFUSAL_EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_13layers/gemma4_31b")
MODEL_PATH = Path("/home/ubuntu/models/Gemma-4-31B-it")

LAYER_IDXS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def load_probe(probe_name: str):
    z = np.load(PROBES_NPZ, allow_pickle=True)
    coef = z[f"coef_{probe_name}"].astype(np.float32)
    bias = float(z[f"bias_{probe_name}"])
    specs = json.loads(PROBE_SPECS.read_text())
    spec = next(s for s in specs["probes"] if s["name"] == probe_name)
    return coef, bias, spec


def per_token_scores(probe_name, coef, spec, ex):
    """Returns (per_token_score: 1d ndarray of length n_tokens, layer_for_word_align: int)."""
    residuals = ex["residuals"].float()        # (n_layers, n_tok, d) fp32
    mask = ex["attention_mask"].bool().squeeze()  # (n_tok,)
    n = int(mask.sum().item())
    if n < 2:
        return None, None
    layer_idxs = list(ex["layer_idxs"])
    if spec["kind"] == "lr_single_layer":
        if spec["pooling"] != "mean":
            return None, None  # last-tok probes don't decompose token-by-token
        L = spec["layer"]
        li = layer_idxs.index(L)
        # contribution of token t = (1/n) * coef · h_t,L
        scores = (residuals[li].numpy() @ coef) / n   # (n_tok,)
        # zero-out padding
        scores = np.where(mask.numpy(), scores, -1e9)
        return scores, L
    elif spec["kind"] == "lr_multi_layer":
        # coef shape (13*d,) — sum over layers
        d_model = residuals.shape[2]
        coef_per_layer = coef.reshape(len(layer_idxs), d_model)
        # per-token contribution = sum_L (1/n) * coef_L · h_t,L
        scores = np.zeros(residuals.shape[1], dtype=np.float32)
        for li in range(len(layer_idxs)):
            scores += (residuals[li].numpy() @ coef_per_layer[li]) / n
        scores = np.where(mask.numpy(), scores, -1e9)
        return scores, layer_idxs[len(layer_idxs)//2]  # middle layer for word-align
    elif spec["kind"] == "lr_aggregate":
        # mean_of_layers / max_of_layers — each token contributes to mean across layers
        # For mean_of_layers: scores_t = (1/n) * (1/13) * sum_L coef · h_t,L
        d_model = residuals.shape[2]
        scores = np.zeros(residuals.shape[1], dtype=np.float32)
        for li in range(len(layer_idxs)):
            scores += (residuals[li].numpy() @ coef) / (n * len(layer_idxs))
        scores = np.where(mask.numpy(), scores, -1e9)
        return scores, layer_idxs[len(layer_idxs)//2]
    else:
        return None, None


def find_word_span(prompt: str, token_str: str) -> tuple[int, int]:
    needle = token_str.strip()
    if not needle: return (-1, -1)
    matches = list(re.finditer(re.escape(needle), prompt))
    if not matches: return (-1, -1)
    s, e = matches[0].span()
    while s > 0 and not prompt[s - 1].isspace():
        s -= 1
    while e < len(prompt) and not prompt[e].isspace() and prompt[e] not in ".,;:!?\"'":
        e += 1
    return (s, e)


def find_sentence_span(prompt: str, char_pos: int) -> tuple[int, int]:
    if char_pos < 0 or char_pos >= len(prompt): return (-1, -1)
    s = char_pos
    while s > 0:
        if prompt[s - 1] in ".!?\n": break
        s -= 1
    while s < len(prompt) and prompt[s].isspace(): s += 1
    e = char_pos
    while e < len(prompt):
        if prompt[e] in ".!?\n":
            e += 1
            break
        e += 1
    return (s, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="lr_mean_L40",
                    help="Probe name from fitted_probes.npz (default: lr_mean_L40)")
    args = ap.parse_args()

    print(f"[probe-gradient edits] probe={args.probe}", flush=True)
    coef, bias, spec = load_probe(args.probe)
    print(f"  coef shape: {coef.shape}  spec: {spec}", flush=True)

    # Load eval prompts + originals
    eval_ids = [json.loads(l)["sample_id"] for l in EVAL_SET.read_text().splitlines() if l.strip()]
    originals = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            originals[r["sample_id"]] = r["attack_prompt"]
    eval_ids = [s for s in eval_ids if s in originals]
    print(f"  {len(eval_ids)} attribution_eval prompts with originals", flush=True)

    print("  loading tokenizer...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    OUT = HERE / f"gradient_edits_{args.probe}.jsonl"
    candidates = []
    for sid in eval_ids:
        orig = originals[sid]
        ext_path = REFUSAL_EXTRACTS / f"{sid}.pt"
        if not ext_path.exists():
            continue
        ex = torch.load(str(ext_path), weights_only=False, map_location="cpu", mmap=True)
        scores, L_for_align = per_token_scores(args.probe, coef, spec, ex)
        if scores is None:
            continue
        input_ids = ex["input_ids"].squeeze().tolist()
        decoded = [tok.decode([i]) for i in input_ids]
        # Top-K positions by score (positive contribution = pulling toward class 1 = refusal)
        # We want to DELETE tokens that pull TOWARD refusal — so high positive scores.
        topK_positions = np.argsort(-scores)[:5]
        top_tokens = [(int(p), decoded[p].strip(), float(scores[p])) for p in topK_positions
                      if decoded[p].strip()]
        if not top_tokens:
            continue

        # delete_top1_word_grad
        for label, top_n in [("delete_top1_word_grad", 1), ("delete_top3_words_grad", 3)]:
            deletions = []
            for pos, tok_str, sc in top_tokens[:top_n]:
                ws, we = find_word_span(orig, tok_str)
                if 0 <= ws < we:
                    deletions.append((ws, we))
            deletions = sorted(set(deletions))
            if deletions:
                edited = orig
                deleted_parts = []
                for ws, we in sorted(deletions, key=lambda x: -x[0]):
                    deleted_parts.append(edited[ws:we])
                    edited = edited[:ws] + edited[we:]
                candidates.append({
                    "sample_id": sid, "edit_kind": label, "probe": args.probe,
                    "deleted_text": " | ".join(reversed(deleted_parts)),
                    "n_chars_deleted": sum(we - ws for ws, we in deletions),
                    "n_words_deleted": len(deletions),
                    "edited_prompt": edited,
                    "original_chars": len(orig), "edited_chars": len(edited),
                    "top_token_scores": [{"pos": p, "tok": t, "score": s} for p, t, s in top_tokens[:top_n]],
                })

        # delete_top1_sentence_grad
        pos, tok_str, sc = top_tokens[0]
        # Need char position of the top token in the prompt
        matches = list(re.finditer(re.escape(tok_str), orig))
        if matches:
            char_pos = matches[0].start()
            ss, se = find_sentence_span(orig, char_pos)
            if 0 <= ss < se:
                edited = orig[:ss] + orig[se:]
                candidates.append({
                    "sample_id": sid, "edit_kind": "delete_top1_sentence_grad",
                    "probe": args.probe,
                    "deleted_text": orig[ss:se],
                    "n_chars_deleted": se - ss,
                    "n_words_deleted": len(orig[ss:se].split()),
                    "edited_prompt": edited,
                    "original_chars": len(orig), "edited_chars": len(edited),
                    "top_token_scores": [{"pos": pos, "tok": tok_str, "score": sc}],
                })

    with OUT.open("w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    print(f"\nwrote {OUT}: {len(candidates)} candidates over {len(eval_ids)} prompts", flush=True)

    by_kind = {}
    for c in candidates:
        by_kind.setdefault(c["edit_kind"], []).append(c)
    for k, lst in by_kind.items():
        char_med = sorted([c["n_chars_deleted"] for c in lst])[len(lst)//2]
        word_med = sorted([c["n_words_deleted"] for c in lst])[len(lst)//2]
        print(f"  {k:>30}: n={len(lst)}, median chars={char_med}, words={word_med}", flush=True)


if __name__ == "__main__":
    main()
