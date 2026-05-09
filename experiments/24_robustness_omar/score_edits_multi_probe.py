"""Re-score 403 edited prompts under EVERY fitted probe.

Pipeline:
  1. For each (sample_id, edit_kind, edited_prompt) in score_edits_unified inputs:
       - run Gemma forward (output_hidden_states=True, all 13 layers)
       - mean-pool every layer + last-token every layer (cached as one tensor)
  2. Apply every fitted probe in fitted_probes.npz to each edit's activations.
  3. Save edits_scored_multi.jsonl with one row per (sample_id, edit_kind),
     one column per probe.

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

HERE = Path(__file__).resolve().parent
PROBES = HERE / "fitted_probes.npz"
SPECS = HERE / "probe_specs.json"
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
SURGICAL = REPO_ROOT / "experiments" / "19_surgical_edit_omar" / "surgical_candidates.jsonl"
DEEPSEEK = REPO_ROOT / "experiments" / "20_deepseek_iterative_omar" / "deepseek_edits.jsonl"
EVAL_SET = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attribution_eval.jsonl"
MODEL_PATH = Path("/home/ubuntu/models/Gemma-4-31B-it")
OUT = HERE / "edits_scored_multi.jsonl"
LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def load_probes():
    z = np.load(PROBES, allow_pickle=True)
    specs = json.loads(SPECS.read_text())
    probes = {}
    for s in specs["probes"]:
        n = s["name"]
        probes[n] = {"spec": s,
                     "coef": z[f"coef_{n}"].astype(np.float32),
                     "bias": float(z[f"bias_{n}"])}
    return probes


def queue_edits():
    eval_ids = {json.loads(l)["sample_id"] for l in EVAL_SET.read_text().splitlines() if l.strip()}
    originals = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            originals[r["sample_id"]] = r["attack_prompt"]
    out = []
    for sid in eval_ids:
        if sid in originals:
            out.append((sid, "original", originals[sid]))
    if SURGICAL.exists():
        with SURGICAL.open() as f:
            for line in f:
                r = json.loads(line)
                if r["sample_id"] in eval_ids:
                    out.append((r["sample_id"], r["edit_kind"], r["edited_prompt"]))
    if DEEPSEEK.exists():
        with DEEPSEEK.open() as f:
            for line in f:
                r = json.loads(line)
                if r["sample_id"] in eval_ids and "edited_prompt" in r:
                    out.append((r["sample_id"], "deepseek_single_round", r["edited_prompt"]))
    return out


def already_done():
    if not OUT.exists(): return set()
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


def score_features(probes, mean_per_layer, last_per_layer):
    """mean_per_layer: (13, d). last_per_layer: (13, d)."""
    scores = {}
    n_layers, d = mean_per_layer.shape
    flat_mean_concat = mean_per_layer.reshape(-1)  # (13*d,)
    mean_of_layers = mean_per_layer.mean(axis=0)
    max_of_layers = mean_per_layer.max(axis=0)
    for name, p in probes.items():
        spec = p["spec"]; coef = p["coef"]; bias = p["bias"]
        if spec["kind"] == "lr_single_layer":
            li = LAYERS.index(spec["layer"])
            feat = mean_per_layer[li] if spec["pooling"] == "mean" else last_per_layer[li]
        elif spec["kind"] == "lr_multi_layer":
            feat = flat_mean_concat
        elif spec["kind"] == "lr_aggregate":
            feat = mean_of_layers if spec["agg"] == "mean_over_layers" else max_of_layers
        else:
            continue
        if feat.shape != coef.shape:
            scores[name] = float("nan")
            continue
        logit = float(feat @ coef + bias)
        prob = 1.0 / (1.0 + np.exp(-logit))
        scores[name] = float(prob)
    return scores


def main():
    probes = load_probes()
    print(f"loaded {len(probes)} probes", flush=True)

    queue = queue_edits()
    done = already_done()
    queue = [q for q in queue if (q[0], q[1]) not in done]
    print(f"queue: {len(queue)} edits to score (skipped {len(done)} already done)", flush=True)
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

    n_done = 0; t_start = time.time()
    for sid, edit_kind, prompt_text in queue:
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
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            use_cache=False, output_hidden_states=True, return_dict=True)
            # hidden_states[i] for i in [0..n_layers]; pull the LAYERS we care about
            hs = out.hidden_states
            mask_cpu = attn_mask.squeeze(0).bool().cpu()
            n = int(mask_cpu.sum().item())
            if n < 2: continue
            last_idx = int(mask_cpu.nonzero().max().item())
            mean_per_layer = []
            last_per_layer = []
            for L in LAYERS:
                r = hs[L].squeeze(0).float().cpu()      # (n_tok, d)
                m = mask_cpu.float().unsqueeze(-1)
                feat_mean = ((r * m).sum(dim=0) / n).numpy()
                feat_last = r[last_idx, :].numpy()
                mean_per_layer.append(feat_mean.astype(np.float32))
                last_per_layer.append(feat_last.astype(np.float32))
            mean_per_layer = np.stack(mean_per_layer)
            last_per_layer = np.stack(last_per_layer)
            if not np.isfinite(mean_per_layer).all() or not np.isfinite(last_per_layer).all():
                continue

            scores = score_features(probes, mean_per_layer, last_per_layer)
            row = {
                "sample_id": sid,
                "edit_kind": edit_kind,
                "n_tokens": n,
                "prompt_chars": len(prompt_text),
                "scores": scores,
            }
            append(row)
            del out, hs, mean_per_layer, last_per_layer
            torch.cuda.empty_cache(); gc.collect()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [OOM] {sid}/{edit_kind}: {e}", flush=True)
            torch.cuda.empty_cache(); continue
        except Exception as e:
            print(f"  [ERR] {sid}/{edit_kind}: {type(e).__name__}: {e}", flush=True)
            continue
        n_done += 1
        if n_done % 25 == 0 or n_done == len(queue):
            print(f"  [{n_done}/{len(queue)}] {time.time()-t_start:.0f}s elapsed", flush=True)

    print(f"\nfinished. wrote {n_done} new rows to {OUT}", flush=True)


if __name__ == "__main__":
    main()
