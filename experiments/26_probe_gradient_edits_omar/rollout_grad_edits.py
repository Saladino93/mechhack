"""Roll Gemma on probe-gradient edits (P1 simplified).

For each new gradient_edits_*.jsonl from picker.py:
  1. Roll Gemma to get the model's response.
  2. Judge with DeepSeek (or fall back to heuristic).
  3. Compute Pr(model flipped) per gradient-edit method × probe.

GPU + OpenRouter API. ~30 min for 3 probes × ~243 edits each.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "starter_code"))
from chunked_sdpa import chunked_sdpa_scope  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_PATH = Path("/home/ubuntu/models/Gemma-4-31B-it")
OUT = HERE / "gradient_rollouts.jsonl"
MAX_NEW_TOKENS = 256


def already_done():
    if not OUT.exists(): return set()
    done = set()
    with OUT.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["sample_id"], r["edit_kind"], r["probe"]))
            except Exception:
                pass
    return done


def append(row):
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    # Find all gradient_edits_*.jsonl
    edit_files = sorted(HERE.glob("gradient_edits_*.jsonl"))
    if not edit_files:
        print("no gradient_edits_*.jsonl found; run picker.py first")
        return
    print(f"found {len(edit_files)} gradient-edit files: {[f.name for f in edit_files]}", flush=True)

    # Build queue: (sid, kind, probe, edited_prompt)
    queue = []
    for f in edit_files:
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                queue.append((r["sample_id"], r["edit_kind"], r["probe"], r["edited_prompt"]))
    done = already_done()
    queue = [q for q in queue if (q[0], q[1], q[2]) not in done]
    print(f"queue: {len(queue)} edits to roll (skip {len(done)} done)", flush=True)
    if not queue: return

    print("loading Gemma...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.float16, device_map="cuda:0",
        attn_implementation="sdpa", trust_remote_code=True,
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    n_done = 0
    t_start = time.time()
    for sid, edit_kind, probe, prompt_text in queue:
        if not prompt_text: continue
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False, add_generation_prompt=True,
            )
            enc = tok(text, return_tensors="pt", truncation=True, max_length=8192)
            input_ids = enc["input_ids"].to("cuda:0")
            attn_mask = enc["attention_mask"].to("cuda:0")
            with torch.no_grad(), chunked_sdpa_scope():
                gen = model.generate(
                    input_ids=input_ids, attention_mask=attn_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, use_cache=True,
                )
            resp = tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
            row = {"sample_id": sid, "edit_kind": edit_kind, "probe": probe,
                   "prompt_chars": len(prompt_text), "response_raw": resp}
            append(row)
            del gen, input_ids, attn_mask, enc
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [OOM] {sid}/{edit_kind}: {e}", flush=True)
            torch.cuda.empty_cache(); continue
        except Exception as e:
            print(f"  [ERR] {sid}/{edit_kind}/{probe}: {type(e).__name__}: {e}", flush=True)
            continue
        n_done += 1
        if n_done % 25 == 0 or n_done == len(queue):
            print(f"  [{n_done}/{len(queue)}] {time.time()-t_start:.0f}s elapsed", flush=True)

    print(f"\nfinished. wrote {n_done} new rows to {OUT}", flush=True)


if __name__ == "__main__":
    main()
