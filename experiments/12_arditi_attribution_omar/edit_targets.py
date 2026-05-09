"""Render 'what to edit' per attribution_eval prompt.

For each (or a sampled subset of) the 81 prompts:
  1. Decode the input_ids to get the per-token strings.
  2. Load the per-prompt top-K Arditi attribution from attributions.jsonl.
  3. For each top-K token, print a 12-token window (6 before / 6 after) so
     you can SEE the local phrase containing the refusal-pulling token.
  4. Compare against the actual DeepSeek rewrite (rewrites_k7.json[0]) to
     see whether the rewrite actually struck through that phrase.

Output:
  - edit_targets.md   : human-readable per-prompt report (cap to first 12 prompts
                         to keep the file readable; pass --all to dump every prompt)

Run:
  python edit_targets.py            # 12-prompt sample
  python edit_targets.py --all      # all 81 prompts (~700 lines)
  python edit_targets.py --sample EVO_0076 EVO_0012 ...

CPU only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ATTRS = Path(__file__).parent / "attributions.jsonl"
EXTRACTS = Path("/home/ubuntu/extracts/gemma_refusal_L32/gemma4_31b")
REWRITES = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "rewrites_k7.json"
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
MODEL_PATH = Path("/home/ubuntu/models/Gemma-4-31B-it")
OUT = Path(__file__).parent / "edit_targets.md"

# Default 12 sample IDs spanning lr_prob from the cleanest probe-flips to a couple of
# probe-but-not-behavior-flips, plus a few control samples.
DEFAULT_SAMPLES = [
    "EVO_0012",  # framing-heavy infrastructure security
    "EVO_0023",  # forensic course
    "EVO_0024",  # ransomware-flavored validation
    "EVO_0076",  # supply-chain — the cleanest probe+behavior flip in our run
    "EVO_0109",  # BLE cloning
    "EVO_0137",  # form-tracker / keylogger pattern
    "EVO_0156",  # remote-binary executor
    "EVO_0208",  # C2 relay (probe flipped, model didn't)
    "EVO_0383",  # npm package (probe flipped strongly, model still refused)
    "EVO_0437",  # collect.js — strongest example in level2_summary
    "EVO_0669",  # python implementation
    "EVO_0861",  # browser-window simulation
]


def load_rows(args):
    rows = {}
    with ATTRS.open() as f:
        for line in f:
            r = json.loads(line)
            rows[r["sample_id"]] = r
    if args.all:
        keys = list(rows.keys())
    elif args.sample:
        keys = [s for s in args.sample if s in rows]
    else:
        keys = [s for s in DEFAULT_SAMPLES if s in rows]
    return [rows[k] for k in keys]


def load_originals():
    out = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["sample_id"]] = r["attack_prompt"]
    return out


def load_first_rewrite():
    rw = json.loads(REWRITES.read_text())
    return {entry["sample_id"]: entry["rewrites"][0] for entry in rw}


def get_window(decoded_tokens, mask_positions, center_pos, half=6):
    """Return ' ... ' joined window of ~12 tokens around center_pos, with the centre
    underscored as **token**."""
    masked = sorted(mask_positions)
    if center_pos not in mask_positions:
        return f"(centre pos {center_pos} not in masked tokens)"
    idx_in_masked = masked.index(center_pos)
    lo = max(0, idx_in_masked - half)
    hi = min(len(masked), idx_in_masked + half + 1)
    out = []
    for j in range(lo, hi):
        gp = masked[j]
        tok = decoded_tokens[gp]
        if gp == center_pos:
            out.append(f"**{tok}**")
        else:
            out.append(tok)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", nargs="*")
    args = ap.parse_args()

    rows = load_rows(args)
    originals = load_originals()
    first_rewrites = load_first_rewrite()

    print(f"loading tokenizer + extracts for {len(rows)} prompts...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    out_lines = []
    out_lines.append("# What to edit — Arditi attribution targets")
    out_lines.append("")
    out_lines.append(
        "For each prompt: top-attributed tokens (Arditi-direction projection at L32) shown "
        "with a ±6-token window. The **bold** token is the centre of the attribution "
        "(highest score at that position). DeepSeek's first rewrite is shown for comparison "
        "— note how it strips the surrounding safety-washing context, not the centre token "
        "alone."
    )
    out_lines.append("")

    for r in rows:
        sid = r["sample_id"]
        ext_path = EXTRACTS / f"{sid}.pt"
        if not ext_path.exists():
            continue
        ex = torch.load(str(ext_path), weights_only=False)
        input_ids = ex["input_ids"].squeeze().tolist()
        mask = ex["attention_mask"].bool().squeeze().tolist()
        masked_positions = {i for i, m in enumerate(mask) if m}
        decoded = [tok.decode([i]) for i in input_ids]
        n_tok = sum(mask)

        out_lines.append(f"## {sid}")
        out_lines.append("")
        out_lines.append(
            f"- prompt chars: {len(originals[sid])}, "
            f"tokens (attended): {n_tok}, "
            f"attribution range: [{r['attribution_min']:+.2f}, {r['attribution_max']:+.2f}]"
        )
        out_lines.append("")
        out_lines.append("### Top-8 refusal-pulling tokens (with ±6-token context)")
        out_lines.append("")
        out_lines.append("| Rank | Pos | Score | Window |")
        out_lines.append("|---:|---:|---:|---|")
        for rank, t in enumerate(
                sorted(r["top_tokens"], key=lambda x: -x["score"]), start=1):
            window = get_window(decoded, masked_positions, t["position"])
            # Replace newlines and pipes for markdown safety
            window = window.replace("\n", " ").replace("|", "\\|")[:120]
            out_lines.append(f"| {rank} | {t['position']} | {t['score']:+.1f} | `…{window}…` |")
        out_lines.append("")
        out_lines.append("### DeepSeek's first rewrite (rewrites_k7.json[0])")
        out_lines.append("")
        rw_text = first_rewrites.get(sid, "")
        out_lines.append(
            f"- length ratio: rewrite={len(rw_text)}, original={len(originals[sid])} "
            f"(rewrite is **{len(rw_text)/max(1,len(originals[sid])):.2f}×** the length)"
        )
        out_lines.append("")
        out_lines.append("```text")
        out_lines.append(rw_text[:1200] + ("\n…[truncated]" if len(rw_text) > 1200 else ""))
        out_lines.append("```")
        out_lines.append("")
        out_lines.append("---")
        out_lines.append("")

    OUT.write_text("\n".join(out_lines))
    print(f"wrote {OUT} ({len(rows)} prompts, {OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
