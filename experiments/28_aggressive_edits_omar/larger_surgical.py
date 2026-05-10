"""Larger surgical-edit buckets — extend exp 19's 3 minimal-deletes with
more aggressive variants:
    delete_top5_words      — 5 highest-Arditi words
    delete_top7_words      — 7 highest-Arditi words
    delete_top2_sentences  — 2 highest-Arditi sentences
    delete_top3_sentences  — 3 highest-Arditi sentences

Goal: increase Pr(f|edit) from the 1.4% we saw with min-1-word edits, while
still keeping edits intent-preserving (judge will check).

CPU only, runs in seconds. Output: aggressive_surgical.jsonl
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ATTRS = REPO_ROOT / "experiments" / "12_arditi_attribution_omar" / "attributions.jsonl"
EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_L32/gemma4_31b")
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
MODEL_PATH = Path("/home/ubuntu/models/Gemma-4-31B-it")
OUT = Path(__file__).parent / "aggressive_surgical.jsonl"


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
    return [p for p in parts if p.strip()]


def find_word_span(prompt: str, token_str: str, hint: tuple[int, int] | None = None):
    needle = token_str.strip()
    if not needle: return (-1, -1)
    matches = [m.span() for m in re.finditer(re.escape(needle), prompt)]
    if not matches: return (-1, -1)
    if hint is None: ms = matches[0]
    else:
        c = (hint[0] + hint[1]) // 2
        ms = min(matches, key=lambda s: abs((s[0] + s[1]) // 2 - c))
    s, e = ms
    while s > 0 and not prompt[s - 1].isspace():
        s -= 1
    while e < len(prompt) and not prompt[e].isspace() and prompt[e] not in ".,;:!?\"'":
        e += 1
    return (s, e)


def find_sentence_span(prompt: str, char_pos: int):
    if char_pos < 0 or char_pos >= len(prompt): return (-1, -1)
    s = char_pos
    while s > 0:
        if prompt[s - 1] in ".!?\n": break
        s -= 1
    while s < len(prompt) and prompt[s].isspace(): s += 1
    e = char_pos
    while e < len(prompt):
        if prompt[e] in ".!?\n":
            e += 1; break
        e += 1
    return (s, e)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in ATTRS.read_text().splitlines() if l.strip()]
    originals = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            originals[r["sample_id"]] = r["attack_prompt"]

    print("loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    candidates = []
    for r in rows:
        sid = r["sample_id"]
        if sid not in originals: continue
        orig = originals[sid]
        n_tot = r["n_tokens_in_scope"]

        ext_path = EXTRACTS / f"{sid}.pt"
        decoded_by_pos = None
        if ext_path.exists():
            ex = torch.load(str(ext_path), weights_only=False)
            input_ids = ex["input_ids"].squeeze().tolist()
            decoded_by_pos = [tok.decode([i]) for i in input_ids]

        top_sorted = sorted(r["top_tokens"], key=lambda t: -t["score"])

        # delete_topN_words for N in {5, 7}
        for N in [5, 7]:
            deletions = []
            for t in top_sorted[:N]:
                tok_str = decoded_by_pos[t["position"]] if decoded_by_pos else t["token"]
                # Estimate char position by proportional mapping
                needle = tok_str.strip()
                if not needle: continue
                matches = [m.span() for m in re.finditer(re.escape(needle), orig)]
                if not matches: continue
                if len(matches) == 1:
                    char_pos = matches[0][0]
                else:
                    target = int(len(orig) * (t["position"] / max(1, n_tot)))
                    char_pos = min(matches, key=lambda s: abs(s[0] - target))[0]
                ws, we = find_word_span(orig, needle,
                                         hint=(char_pos, char_pos + len(needle)))
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
                    "sample_id": sid, "edit_kind": f"delete_top{N}_words",
                    "deleted_text": " | ".join(reversed(deleted_parts)),
                    "n_chars_deleted": sum(we - ws for ws, we in deletions),
                    "n_words_deleted": len(deletions),
                    "n_sentences_deleted": 0,
                    "edited_prompt": edited,
                    "original_chars": len(orig),
                    "edited_chars": len(edited),
                })

        # delete_topN_sentences for N in {2, 3}
        for N in [2, 3]:
            sentence_spans = []
            seen_starts = set()
            for t in top_sorted:
                if len(sentence_spans) >= N: break
                tok_str = decoded_by_pos[t["position"]] if decoded_by_pos else t["token"]
                needle = tok_str.strip()
                if not needle: continue
                matches = [m.span() for m in re.finditer(re.escape(needle), orig)]
                if not matches: continue
                if len(matches) == 1:
                    char_pos = matches[0][0]
                else:
                    target = int(len(orig) * (t["position"] / max(1, n_tot)))
                    char_pos = min(matches, key=lambda s: abs(s[0] - target))[0]
                ss, se = find_sentence_span(orig, char_pos)
                if 0 <= ss < se and ss not in seen_starts:
                    sentence_spans.append((ss, se))
                    seen_starts.add(ss)
            if sentence_spans:
                edited = orig
                deleted_parts = []
                for ss, se in sorted(sentence_spans, key=lambda x: -x[0]):
                    deleted_parts.append(edited[ss:se])
                    edited = edited[:ss] + edited[se:]
                candidates.append({
                    "sample_id": sid, "edit_kind": f"delete_top{N}_sentences",
                    "deleted_text": " | ".join(reversed(deleted_parts)),
                    "n_chars_deleted": sum(se - ss for ss, se in sentence_spans),
                    "n_words_deleted": len(" | ".join(deleted_parts).split()),
                    "n_sentences_deleted": len(sentence_spans),
                    "edited_prompt": edited,
                    "original_chars": len(orig),
                    "edited_chars": len(edited),
                })

    with OUT.open("w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    print(f"\nwrote {OUT}: {len(candidates)} candidates")
    by_kind = {}
    for c in candidates:
        by_kind.setdefault(c["edit_kind"], []).append(c)
    for k, lst in by_kind.items():
        char_med = sorted([c["n_chars_deleted"] for c in lst])[len(lst)//2]
        word_med = sorted([c["n_words_deleted"] for c in lst])[len(lst)//2]
        print(f"  {k:>22}: n={len(lst)}, median chars deleted={char_med}, words={word_med}")


if __name__ == "__main__":
    main()
