"""Surgical-edit candidate generator (Arditi-attribution-driven, minimal-edit).

For each of the 81 attribution_eval prompts, produce a small bouquet of MINIMAL
edit candidates — far smaller than the full-paraphrase rewrites_k7.json — by
deleting Arditi-high-attribution units at three granularities:

  * delete_top1_word     — drop the single word containing the top-1 token
  * delete_top3_words    — drop the words containing the top-3 tokens
  * delete_top1_sentence — drop the sentence containing the top-1 token

These are the smallest interventions you can do that still target the
refusal-direction signal. Naturalness is preserved by deleting whole words /
sentences (not random characters), so they pass the "no GCG gibberish" rule by
construction.

This script ONLY produces candidate prompts. It does NOT run Gemma forward.
Scoring + behavior verification are deferred to a follow-up `score_candidates.py`
that needs GPU access.

Output: `surgical_candidates.jsonl` — one row per (sample_id, edit_kind):
  {sample_id, edit_kind, original_chars, edited_chars, n_chars_deleted,
   n_words_deleted, n_sentences_deleted, deleted_text, edited_prompt}

CPU only.
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
EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_L32/gemma4_31b")  # has input_ids
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
MODEL_PATH = Path("/home/ubuntu/models/Gemma-4-31B-it")
OUT = Path(__file__).parent / "surgical_candidates.jsonl"


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
    return [p for p in parts if p.strip()]


def find_word_span_around_pos(prompt: str, token_str: str, hint_window: tuple[int, int] | None = None) -> tuple[int, int]:
    """Find the (start_char, end_char) of the WORD containing the given token
    string in `prompt`. We search the whole prompt for the token's stripped
    form; if there are multiple matches we use the closest one to the
    optional hint window (start_char, end_char range).
    """
    needle = token_str.strip()
    if not needle:
        return (-1, -1)
    # All matches of the needle in the prompt
    matches = [m.span() for m in re.finditer(re.escape(needle), prompt)]
    if not matches:
        return (-1, -1)
    if hint_window is None:
        ms = matches[0]
    else:
        # Pick match closest to the hint window centre
        centre = (hint_window[0] + hint_window[1]) // 2
        ms = min(matches, key=lambda s: abs((s[0] + s[1]) // 2 - centre))
    # Expand to whole word (left+right while non-whitespace + non-punct)
    s, e = ms
    while s > 0 and not prompt[s - 1].isspace():
        s -= 1
    while e < len(prompt) and not prompt[e].isspace() and not prompt[e] in ".,;:!?\"'":
        e += 1
    return (s, e)


def find_sentence_span(prompt: str, char_pos: int) -> tuple[int, int]:
    """Find (start, end) of the sentence containing char_pos."""
    if char_pos < 0 or char_pos >= len(prompt):
        return (-1, -1)
    # Walk left to start of sentence (after a previous . ! ? newline-or-space)
    s = char_pos
    while s > 0:
        # If we just walked past sentence-end punct, stop
        if prompt[s - 1] in ".!?\n":
            break
        s -= 1
    # Skip whitespace at start
    while s < len(prompt) and prompt[s].isspace():
        s += 1
    # Walk right to next sentence-end
    e = char_pos
    while e < len(prompt):
        if prompt[e] in ".!?\n":
            e += 1
            break
        e += 1
    return (s, e)


def estimate_token_position_in_prompt(prompt: str, decoded_tok: str, tok_pos_in_full: int,
                                       n_total_tokens: int) -> int:
    """Approximate where in `prompt` the token at index `tok_pos_in_full` sits.
    The .pt input_ids include chat-template prefix, so the prompt content starts
    after a fixed offset. Estimate via the decoded_tok and a proportional
    guess from tok_pos / n_total."""
    # Try to find the token text directly in the prompt; if multiple matches,
    # use proportional position.
    needle = decoded_tok.strip()
    if not needle:
        return -1
    matches = [m.span() for m in re.finditer(re.escape(needle), prompt)]
    if not matches:
        return -1
    if len(matches) == 1:
        return matches[0][0]
    # Multiple matches → use the one closest to the proportional position.
    # Proportion: treat input_ids as roughly aligned with prompt chars (rough!)
    if n_total_tokens == 0:
        return matches[0][0]
    target = int(len(prompt) * (tok_pos_in_full / max(1, n_total_tokens)))
    return min(matches, key=lambda s: abs(s[0] - target))[0]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in ATTRS.read_text().splitlines() if l.strip()]
    originals = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            originals[r["sample_id"]] = r["attack_prompt"]

    print(f"loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    candidates = []
    skipped = 0
    for r in rows:
        sid = r["sample_id"]
        if sid not in originals:
            skipped += 1; continue
        orig = originals[sid]
        n_tot = r["n_tokens_in_scope"]

        # Decode the input_ids if available — gives us the literal token strings
        ext_path = EXTRACTS / f"{sid}.pt"
        decoded_by_pos = None
        if ext_path.exists():
            ex = torch.load(str(ext_path), weights_only=False)
            input_ids = ex["input_ids"].squeeze().tolist()
            decoded_by_pos = [tok.decode([i]) for i in input_ids]

        top_sorted = sorted(r["top_tokens"], key=lambda t: -t["score"])

        # ---- delete_top1_word ----
        if top_sorted:
            t = top_sorted[0]
            tok_str = decoded_by_pos[t["position"]] if decoded_by_pos else t["token"]
            char_pos = estimate_token_position_in_prompt(orig, tok_str, t["position"], n_tot)
            if char_pos >= 0:
                ws, we = find_word_span_around_pos(orig, tok_str.strip(),
                                                    hint_window=(char_pos, char_pos + len(tok_str.strip())))
                if 0 <= ws < we:
                    edited = orig[:ws] + orig[we:]
                    candidates.append({
                        "sample_id": sid, "edit_kind": "delete_top1_word",
                        "deleted_text": orig[ws:we],
                        "n_chars_deleted": we - ws,
                        "n_words_deleted": 1,
                        "n_sentences_deleted": 0,
                        "edited_prompt": edited,
                        "original_chars": len(orig),
                        "edited_chars": len(edited),
                    })

        # ---- delete_top3_words ----
        deletions = []
        for t in top_sorted[:3]:
            tok_str = decoded_by_pos[t["position"]] if decoded_by_pos else t["token"]
            char_pos = estimate_token_position_in_prompt(orig, tok_str, t["position"], n_tot)
            if char_pos < 0:
                continue
            ws, we = find_word_span_around_pos(orig, tok_str.strip(),
                                                hint_window=(char_pos, char_pos + len(tok_str.strip())))
            if 0 <= ws < we:
                deletions.append((ws, we))
        # Apply deletions right-to-left (preserves earlier offsets) and dedupe overlaps
        deletions = sorted(set(deletions))
        if deletions:
            edited = orig
            total_deleted_chars = 0
            deleted_text_parts = []
            for ws, we in sorted(deletions, key=lambda x: -x[0]):
                deleted_text_parts.append(edited[ws:we])
                edited = edited[:ws] + edited[we:]
                total_deleted_chars += we - ws
            candidates.append({
                "sample_id": sid, "edit_kind": "delete_top3_words",
                "deleted_text": " | ".join(reversed(deleted_text_parts)),
                "n_chars_deleted": total_deleted_chars,
                "n_words_deleted": len(deletions),
                "n_sentences_deleted": 0,
                "edited_prompt": edited,
                "original_chars": len(orig),
                "edited_chars": len(edited),
            })

        # ---- delete_top1_sentence ----
        if top_sorted:
            t = top_sorted[0]
            tok_str = decoded_by_pos[t["position"]] if decoded_by_pos else t["token"]
            char_pos = estimate_token_position_in_prompt(orig, tok_str, t["position"], n_tot)
            if char_pos >= 0:
                ss, se = find_sentence_span(orig, char_pos)
                if 0 <= ss < se:
                    edited = orig[:ss] + orig[se:]
                    candidates.append({
                        "sample_id": sid, "edit_kind": "delete_top1_sentence",
                        "deleted_text": orig[ss:se],
                        "n_chars_deleted": se - ss,
                        "n_words_deleted": len(orig[ss:se].split()),
                        "n_sentences_deleted": 1,
                        "edited_prompt": edited,
                        "original_chars": len(orig),
                        "edited_chars": len(edited),
                    })

    with OUT.open("w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    print(f"\nwrote {OUT}: {len(candidates)} candidates over {len(rows)-skipped} prompts")
    by_kind = {}
    for c in candidates:
        by_kind.setdefault(c["edit_kind"], []).append(c)
    for k, lst in by_kind.items():
        char_med = sorted([c["n_chars_deleted"] for c in lst])[len(lst)//2]
        word_med = sorted([c["n_words_deleted"] for c in lst])[len(lst)//2]
        print(f"  {k:>22}: n={len(lst)}, median chars deleted={char_med}, "
              f"median words deleted={word_med}")


if __name__ == "__main__":
    main()
