"""Pleshkov-as-f causality story: re-fit Pleshkov quadratic probe on the
full 832 refusal training set, then apply it to the 403 edited prompts to
compute Pr(f|edit), Pr(model|edit), Pr(model|f flipped) — completing the
"all three high" picture for the most interesting non-linear probe.

Pipeline:
  1. Load per-edit cached features… we don't have them, so re-run Gemma
     forward on each of 403 edits at one fixed layer (L40 mean), save
     activations alongside.
  2. Re-fit Pleshkov (PCA d=16 → degree-2 → Ridge) on the 832 refusal
     features at L40 mean. Single fit on the full training set, no CV.
  3. Score the 403 edits under Pleshkov.
  4. Save per-edit pleshkov_score; downstream `compute_pleshkov_metrics.py`
     joins with judgements and computes Pr metrics.

Output:
  - edits_pleshkov_scored.jsonl : per-edit (sample_id, edit_kind, score)
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

sys.path.insert(0, str(REPO_ROOT / "experiments" / "17_quadratic_probe_omar"))
from probes import QuadraticProbe  # noqa: E402

HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)

ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
EVAL_SET = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attribution_eval.jsonl"
SURGICAL = REPO_ROOT / "experiments" / "19_surgical_edit_omar" / "surgical_candidates.jsonl"
DEEPSEEK = REPO_ROOT / "experiments" / "20_deepseek_iterative_omar" / "deepseek_edits.jsonl"
REFUSAL_CACHE = REPO_ROOT / "experiments" / "17_quadratic_probe_omar" / "cache" / "refusal_13layer_mean.npz"
MODEL_PATH = Path("/home/ubuntu/models/Gemma-4-31B-it")
OUT = HERE / "edits_pleshkov_scored.jsonl"

LAYER = 40                # mean-pool L40 — best linear baseline on refusal
LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
LI = LAYERS.index(LAYER)  # = 8


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
    s = set()
    with OUT.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                s.add((r["sample_id"], r["edit_kind"]))
            except Exception:
                pass
    return s


def append(row):
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    # 1. Re-fit Pleshkov on the 832 refusal training set at L40 mean.
    print("[Pleshkov causality]", flush=True)
    print(f"  re-fitting Pleshkov d=16 on full 832 training set @ L{LAYER}", flush=True)
    z = np.load(REFUSAL_CACHE, allow_pickle=True)
    X_train = z["X"][:, LI, :].astype(np.float32)
    y_train = z["y"].astype(np.int64)
    print(f"    X_train shape {X_train.shape}, n_pos={int((y_train==1).sum())}, n_neg={int((y_train==0).sum())}",
          flush=True)
    t0 = time.time()
    probe = QuadraticProbe(d_pca=16, alpha=10.0, random_state=0).fit(X_train, y_train)
    print(f"    fit in {time.time()-t0:.1f}s", flush=True)
    # Sanity: train AUC
    from sklearn.metrics import roc_auc_score
    train_auc = roc_auc_score(y_train, probe.decision_function(X_train))
    print(f"    train AUC = {train_auc:.4f}", flush=True)

    # 2. Forward Gemma on each edited prompt; pull L40 mean; score.
    queue = queue_edits()
    done = already_done()
    queue = [q for q in queue if (q[0], q[1]) not in done]
    print(f"  queue: {len(queue)} edits to score", flush=True)
    if not queue:
        print("nothing to do."); return

    print("  loading Gemma...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.float16, device_map="cuda:0",
        attn_implementation="sdpa", trust_remote_code=True,
    ).eval()
    print(f"    loaded in {time.time()-t0:.1f}s", flush=True)

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
            r_layer = out.hidden_states[LAYER].squeeze(0).float().cpu()  # (n_tok, d)
            mask = attn_mask.squeeze(0).bool().cpu()
            n = int(mask.sum().item())
            if n < 2: continue
            m = mask.float().unsqueeze(-1)
            feat = ((r_layer * m).sum(dim=0) / n).numpy().astype(np.float32)
            if not np.isfinite(feat).all(): continue
            score = float(probe.decision_function(feat[None, :])[0])
            row = {
                "sample_id": sid, "edit_kind": edit_kind,
                "n_tokens": n, "prompt_chars": len(prompt_text),
                "pleshkov_score": score,
                "pleshkov_d_pca": 16, "pleshkov_layer": LAYER,
            }
            append(row)
            del out, r_layer; torch.cuda.empty_cache(); gc.collect()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [OOM] {sid}/{edit_kind}: {e}", flush=True)
            torch.cuda.empty_cache(); continue
        except Exception as e:
            print(f"  [ERR] {sid}/{edit_kind}: {type(e).__name__}: {e}", flush=True)
            continue
        n_done += 1
        if n_done % 50 == 0 or n_done == len(queue):
            print(f"  [{n_done}/{len(queue)}] {time.time()-t_start:.0f}s", flush=True)

    print(f"\nfinished. wrote {n_done} new rows to {OUT}", flush=True)


if __name__ == "__main__":
    main()
