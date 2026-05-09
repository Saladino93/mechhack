"""Score DeepSeek edits + surgical candidates on GPU + LR probe.

Reads two sources of edited prompts:
  - experiments/19_surgical_edit_omar/surgical_candidates.jsonl
       (edit_kind ∈ {delete_top1_word, delete_top3_words, delete_top1_sentence})
  - experiments/20_deepseek_iterative_omar/deepseek_edits.jsonl
       (edit_kind = "deepseek_single_round")

Plus the original prompt as a baseline (re-scored under the same setup).

Per edited prompt:
  1. Tokenize with Gemma chat template (apply_chat_template).
  2. Run Gemma forward, output_hidden_states=True, take L32.
  3. Mean-pool over attended tokens.
  4. Score with the LR probe (lr_probe.npz from exp 13) AND the Arditi
     direction (arditi_direction.npy from exp 11).
  5. Save: probe scores per (sample_id, edit_kind, rewrite_label).

Output: edits_scored.jsonl with one row per edited prompt.

Resume support: skip rows already in output.
"""
from __future__ import annotations
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "starter_code"))
from chunked_sdpa import chunked_sdpa_scope  # noqa: E402

ARDITI_PATH = REPO_ROOT / "experiments" / "11_refusal_probe_omar" / "arditi_direction.npy"
LR_PROBE_PATH = REPO_ROOT / "experiments" / "13_pre_rewrites_omar" / "lr_probe.npz"
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
SURGICAL = REPO_ROOT / "experiments" / "19_surgical_edit_omar" / "surgical_candidates.jsonl"
DEEPSEEK = REPO_ROOT / "experiments" / "20_deepseek_iterative_omar" / "deepseek_edits.jsonl"
EVAL_SET = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attribution_eval.jsonl"
MODEL_PATH = Path("/home/ubuntu/models/Gemma-4-31B-it")
OUT = Path(__file__).parent / "edits_scored.jsonl"
TARGET_LAYER = 32


def load_eval_ids():
    return [json.loads(l)["sample_id"] for l in EVAL_SET.read_text().splitlines() if l.strip()]


def load_originals():
    out = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["sample_id"]] = r["attack_prompt"]
    return out


def load_lr_probe():
    z = np.load(LR_PROBE_PATH, allow_pickle=True)
    return z["coef"].astype(np.float32), float(z["bias"])


def load_arditi():
    return np.load(ARDITI_PATH).astype(np.float32)


def already_done():
    if not OUT.exists():
        return set()
    done = set()
    with OUT.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["sample_id"], r["edit_kind"]))
            except Exception:
                pass
    return done


def append(row):
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


def queue_edits():
    """Return list of (sample_id, edit_kind, rewrite_label, prompt_text)."""
    eval_ids = set(load_eval_ids())
    originals = load_originals()
    out = []

    # 1. originals
    for sid in eval_ids:
        if sid in originals:
            out.append((sid, "original", "original", originals[sid]))

    # 2. surgical candidates
    if SURGICAL.exists():
        with SURGICAL.open() as f:
            for line in f:
                r = json.loads(line)
                if r["sample_id"] not in eval_ids:
                    continue
                out.append((r["sample_id"], r["edit_kind"], r["edit_kind"], r["edited_prompt"]))

    # 3. deepseek edits
    if DEEPSEEK.exists():
        with DEEPSEEK.open() as f:
            for line in f:
                r = json.loads(line)
                if r["sample_id"] not in eval_ids:
                    continue
                if "edited_prompt" not in r:
                    continue
                out.append((r["sample_id"], "deepseek_single_round", "deepseek_single_round",
                            r["edited_prompt"]))
    return out


def main():
    coef, bias = load_lr_probe()
    arditi = load_arditi()
    print(f"loaded LR probe (|coef|_2={float(np.linalg.norm(coef)):.2f}, bias={bias:.4f})", flush=True)
    print(f"loaded Arditi direction (norm={float(np.linalg.norm(arditi)):.4f})", flush=True)

    queue = queue_edits()
    done = already_done()
    queue = [q for q in queue if (q[0], q[1]) not in done]
    print(f"queue: {len(queue)} edits to score (skipped {len(done)} already done)", flush=True)
    if not queue:
        print("nothing to do.")
        return

    print(f"loading Gemma...", flush=True)
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
    for sid, edit_kind, label, prompt_text in queue:
        if not prompt_text:
            continue
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False, add_generation_prompt=True,
            )
            enc = tok(text, return_tensors="pt", truncation=True, max_length=8192)
            input_ids = enc["input_ids"].to("cuda:0")
            attn_mask = enc["attention_mask"].to("cuda:0")
            with torch.no_grad(), chunked_sdpa_scope():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            use_cache=False, output_hidden_states=True, return_dict=True)
            res = out.hidden_states[TARGET_LAYER]  # (1, n_tok, d)
            mask = attn_mask.squeeze(0).bool().cpu()
            n = int(mask.sum().item())
            if n < 2:
                continue
            r = res.squeeze(0).float().cpu()  # (n_tok, d)
            m = mask.float().unsqueeze(-1)
            feat = ((r * m).sum(dim=0) / n).numpy()  # (d,)
            if not np.isfinite(feat).all():
                continue
            arditi_score = float(feat @ arditi)
            logit = float(feat @ coef + bias)
            lr_prob = 1.0 / (1.0 + np.exp(-logit))
            row = {
                "sample_id": sid,
                "edit_kind": edit_kind,
                "rewrite_label": label,
                "n_tokens": n,
                "lr_prob": float(lr_prob),
                "arditi_score": arditi_score,
                "prompt_chars": len(prompt_text),
            }
            append(row)
            del res, r; gc.collect(); torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [OOM] {sid}/{edit_kind}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        n_done += 1
        if n_done % 25 == 0 or n_done == len(queue):
            print(f"  [{n_done}/{len(queue)}]  {time.time()-t_start:.0f}s elapsed", flush=True)

    print(f"\nfinished. wrote {n_done} new rows to {OUT}")


if __name__ == "__main__":
    main()
