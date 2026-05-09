"""Extract Gemma activations at 13 layers for the missing cyber TEST split.

Currently extracted: 4,189 cyber prompts (3,266 train + 923 test).
Missing test: 1,257 prompts.

Per prompt:
  - tokenize with chat template
  - forward, output_hidden_states=True
  - take layers [0,5,10,15,20,25,30,35,40,45,50,55,60]
  - save to /home/ubuntu/extracts/cyber_all_omar/<sample_id>.pt
    schema: residuals (13, n_tok, 5376) fp16, input_ids, attention_mask,
            n_tokens, layer_idxs, label (category), sample_id

Resume support: skip if .pt exists.

Wall-time estimate: 1,257 prompts × ~6s = ~2.1 hours on H100.
"""
from __future__ import annotations
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "starter_code"))
from chunked_sdpa import chunked_sdpa_scope  # noqa: E402

TEST_PATH = REPO_ROOT / "datasets" / "cyber_probes" / "test.jsonl"
EXTRACTS_DIR = Path("/home/ubuntu/extracts/cyber_all_omar")
MODEL_PATH = Path("/home/ubuntu/models/Gemma-4-31B-it")
LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
MAX_TOKENS = 8192


def main():
    EXTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with TEST_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
    print(f"loaded {len(rows)} cyber test prompts", flush=True)

    queue = []
    for r in rows:
        sid = r["sample_id"]
        out = EXTRACTS_DIR / f"{sid}.pt"
        if out.exists():
            continue
        queue.append(r)
    print(f"queue: {len(queue)} prompts to extract (skipping {len(rows)-len(queue)} already done)",
          flush=True)
    if not queue:
        print("nothing to do."); return

    print("loading Gemma...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.float16, device_map="cuda:0",
        attn_implementation="sdpa", trust_remote_code=True,
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    n_done = 0; n_err = 0; t_start = time.time()
    for r in queue:
        sid = r["sample_id"]
        prompt = r["prompt"]
        out_path = EXTRACTS_DIR / f"{sid}.pt"
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
            input_ids = enc["input_ids"].to("cuda:0")
            attn_mask = enc["attention_mask"].to("cuda:0")
            with torch.no_grad(), chunked_sdpa_scope():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            use_cache=False, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states
            stacked = torch.stack([hs[L][0] for L in LAYERS], dim=0)  # (13, n_tok, d)
            payload = {
                "residuals": stacked.to("cpu", dtype=torch.float16).contiguous(),
                "input_ids": input_ids[0].to("cpu", dtype=torch.int32),
                "attention_mask": attn_mask[0].to("cpu", dtype=torch.bool),
                "n_tokens": int(stacked.shape[1]),
                "layer_idxs": LAYERS,
                "label": r.get("category"),
                "sample_id": sid,
                "split": "test",
            }
            torch.save(payload, str(out_path))
            del out, hs, stacked, input_ids, attn_mask, enc, payload
            torch.cuda.empty_cache(); gc.collect()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [OOM] {sid}: {e}", flush=True)
            torch.cuda.empty_cache()
            n_err += 1
            continue
        except Exception as e:
            print(f"  [ERR] {sid}: {type(e).__name__}: {e}", flush=True)
            n_err += 1
            continue
        n_done += 1
        if n_done % 50 == 0 or n_done == len(queue):
            elapsed = time.time() - t_start
            rate = n_done / elapsed
            eta = (len(queue) - n_done) / rate if rate > 0 else 0
            print(f"  [{n_done}/{len(queue)}] err={n_err}  {elapsed:.0f}s elapsed, "
                  f"eta {eta:.0f}s ({eta/60:.1f}m)", flush=True)

    print(f"\nfinished. ok={n_done}  err={n_err}  total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
