#!/usr/bin/env python3
"""Create prompt-pair data for a first-sentence causal ablation experiment.

This script does NOT run the model. It creates a JSONL containing original and
ablated prompts, plus metadata from transfer_scores.csv. You then feed the JSONL
through the hackathon rollout/refusal verifier, and analyze flips.

Example:
  python make_first_sentence_ablation.py \
    --transfer_scores_csv ./analysis/cyber3_on_refusal_gemma_fixed/transfer_scores.csv \
    --dataset refusal_gemma4_31b \
    --split test \
    --quadrants high_cyber_refusal,high_cyber_compliance \
    --out_jsonl ./analysis/first_sentence_ablation/prompts.jsonl
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path.cwd()
# Allow running from repo root or copied into experiments/06_causal_ablation.
sys.path.insert(0, str(Path.cwd() / "experiments" / "02_extract_activations"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
try:
    from data import load_dataset
except Exception as e:
    raise SystemExit(
        "Could not import experiments/02_extract_activations/data.py. "
        "Run from repo root or add that directory to PYTHONPATH."
    ) from e

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'`({\[])|\n+")


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter robust enough for ablation generation."""
    parts = [p.strip() for p in _SENT_BOUNDARY.split(text.strip()) if p.strip()]
    if len(parts) <= 1:
        # Fallback: split on first newline, then first period-like punctuation.
        if "\n" in text.strip():
            first, rest = text.strip().split("\n", 1)
            return [first.strip(), rest.strip()] if rest.strip() else [first.strip()]
        m = re.search(r"[.!?]", text)
        if m and m.end() < len(text):
            return [text[:m.end()].strip(), text[m.end():].strip()]
    return parts


def remove_first_sentence(text: str, n: int = 1) -> Tuple[str, str, int]:
    sents = split_sentences(text)
    if len(sents) <= n:
        return "", text.strip(), len(sents)
    removed = " ".join(sents[:n]).strip()
    kept = " ".join(sents[n:]).strip()
    return kept, removed, len(sents)


def remove_random_sentence(text: str, rng: random.Random, protected_first: bool = True) -> Tuple[str, str, int, int]:
    sents = split_sentences(text)
    if len(sents) <= 1:
        return text.strip(), "", len(sents), -1
    choices = list(range(1 if protected_first else 0, len(sents)))
    if not choices:
        return text.strip(), "", len(sents), -1
    idx = rng.choice(choices)
    removed = sents[idx]
    kept = " ".join(s for i, s in enumerate(sents) if i != idx).strip()
    return kept, removed, len(sents), idx


def load_samples_by_id(dataset: str, split: str) -> Dict[str, dict]:
    if split == "all":
        splits = ["train", "test"]
        out = {}
        for sp in splits:
            for s in load_dataset(dataset, split=sp):
                out[s["sample_id"]] = s
        try:
            for s in load_dataset(dataset, split="full"):
                out.setdefault(s["sample_id"], s)
        except Exception:
            pass
        return out
    return {s["sample_id"]: s for s in load_dataset(dataset, split=split)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transfer_scores_csv", required=True)
    ap.add_argument("--dataset", default="refusal_gemma4_31b")
    ap.add_argument("--split", default="test", help="Dataset split to recover prompts from: train, test, full, eval, or all")
    ap.add_argument("--quadrants", default="high_cyber_refusal,high_cyber_compliance",
                    help="Comma-separated quadrants to include, or 'all'.")
    ap.add_argument("--top_k_per_quadrant", type=int, default=0,
                    help="If >0, keep top-k by score_logit within each selected quadrant.")
    ap.add_argument("--remove_first_n", type=int, default=1)
    ap.add_argument("--include_random_control", action="store_true",
                    help="Also add a control variant removing one non-first random sentence.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.transfer_scores_csv)
    if args.quadrants != "all":
        wanted = {q.strip() for q in args.quadrants.split(",") if q.strip()}
        df = df[df["quadrant"].isin(wanted)].copy()

    if args.top_k_per_quadrant and args.top_k_per_quadrant > 0:
        df = (df.sort_values("score_logit", ascending=False)
                .groupby("quadrant", group_keys=False)
                .head(args.top_k_per_quadrant)
                .copy())

    samples_by_id = load_samples_by_id(args.dataset, args.split)
    rng = random.Random(args.seed)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_missing = 0
    n_written = 0
    with out_path.open("w") as f:
        for _, row in df.iterrows():
            sid = row["sample_id"]
            sample = samples_by_id.get(sid)
            if sample is None:
                n_missing += 1
                continue
            prompt = sample["prompt"]
            ablated, removed, n_sents = remove_first_sentence(prompt, n=args.remove_first_n)
            base_meta = {
                "sample_id": sid,
                "dataset": args.dataset,
                "source_split": sample.get("split", args.split),
                "raw_label": sample.get("label"),
                "original_is_refusal": int(row.get("is_refusal", -1)),
                "original_quadrant": row.get("quadrant"),
                "original_score_logit": float(row.get("score_logit", float("nan"))),
                "original_n_tokens": int(row.get("n_tokens", -1)),
                "n_sentences_est": int(n_sents),
            }
            variants = [
                ("original", prompt, ""),
                (f"remove_first_{args.remove_first_n}_sentence", ablated, removed),
            ]
            if args.include_random_control:
                ctrl, ctrl_removed, _, ctrl_idx = remove_random_sentence(prompt, rng=rng, protected_first=True)
                variants.append(("remove_random_nonfirst_sentence", ctrl, ctrl_removed))
            for variant, text, removed_text in variants:
                rec = {
                    **base_meta,
                    "variant": variant,
                    "prompt": text,
                    "removed_text": removed_text,
                    "empty_after_ablation": not bool(text.strip()),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"wrote {n_written} prompt variants to {out_path}")
    print(f"missing sample_ids: {n_missing}")
    print(df["quadrant"].value_counts().to_string())


if __name__ == "__main__":
    main()
