"""GPU pass: forward Gemma on each rewrite_k7 + original (648 prompts) and
save mean-pool features at all 13 layers + last-token at all 13 layers +
raw per-token at L30 (for Kramar probes). Per-prompt activations needed
for downstream probe scoring.

Output: features_rewrites.npz
  - sample_ids: list[str]
  - rewrite_idxs: list[int]   (-1 for original, 0..6 for rewrites)
  - mean: (N, 13, d)
  - last: (N, 13, d)
  - L30_pertoken: list[Tensor (n_tok, d)]  (saved separately for Kramar probes)
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
HERE.mkdir(parents=True, exist_ok=True)

REWRITES_PATH = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "rewrites_k7.json"
ATTRS_FULL = REPO_ROOT / "datasets" / "refusal_probes" / "gemma4_31b" / "attacks_full.jsonl"
MODEL_PATH = Path("/home/ubuntu/models/Gemma-4-31B-it")
OUT_NPZ = HERE / "features_rewrites.npz"
OUT_PT_DIR = HERE / "L30_pertoken"
OUT_PT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def main():
    rewrites = json.load(REWRITES_PATH.open())
    originals = {}
    with ATTRS_FULL.open() as f:
        for line in f:
            r = json.loads(line)
            originals[r["sample_id"]] = r["attack_prompt"]

    queue = []  # (sample_id, rewrite_idx, prompt)
    for r in rewrites:
        sid = r["sample_id"]
        if sid in originals:
            queue.append((sid, -1, originals[sid]))
        for i, rw in enumerate(r["rewrites"]):
            queue.append((sid, i, rw))
    print(f"queue: {len(queue)} prompts (81 originals + 81×7 rewrites)", flush=True)

    # Resume: skip prompts already in features_rewrites.npz
    done_keys = set()
    if OUT_NPZ.exists():
        try:
            z = np.load(OUT_NPZ, allow_pickle=True)
            for sid, ri in zip(z["sample_ids"], z["rewrite_idxs"]):
                done_keys.add((str(sid), int(ri)))
        except Exception:
            pass
    queue = [q for q in queue if (q[0], q[1]) not in done_keys]
    print(f"after resume filter: {len(queue)} prompts to process", flush=True)
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

    # Pre-allocate accumulators
    sids, ridxs, means, lasts = [], [], [], []
    if OUT_NPZ.exists():
        z = np.load(OUT_NPZ, allow_pickle=True)
        sids = list(z["sample_ids"])
        ridxs = list(z["rewrite_idxs"])
        means = list(z["mean"])
        lasts = list(z["last"])
        print(f"  resumed with {len(sids)} prior entries", flush=True)

    n_done = 0; t_start = time.time()
    save_every = 50
    for sid, ridx, prompt in queue:
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            enc = tok(text, return_tensors="pt", truncation=True, max_length=8192)
            input_ids = enc["input_ids"].to("cuda:0")
            attn_mask = enc["attention_mask"].to("cuda:0")
            with torch.no_grad(), chunked_sdpa_scope():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            use_cache=False, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states
            mask_cpu = attn_mask.squeeze(0).bool().cpu()
            n = int(mask_cpu.sum().item())
            if n < 2: continue
            last_idx = int(mask_cpu.nonzero().max().item())
            mean_per_layer = []
            last_per_layer = []
            for L in LAYERS:
                r_layer = hs[L].squeeze(0).float().cpu()
                m = mask_cpu.float().unsqueeze(-1)
                feat_mean = ((r_layer * m).sum(dim=0) / n).numpy().astype(np.float32)
                feat_last = r_layer[last_idx, :].numpy().astype(np.float32)
                mean_per_layer.append(feat_mean)
                last_per_layer.append(feat_last)
            mean_per_layer = np.stack(mean_per_layer)
            last_per_layer = np.stack(last_per_layer)
            if not np.isfinite(mean_per_layer).all() or not np.isfinite(last_per_layer).all():
                continue

            # Save L30 per-token for Kramar probes (small file per prompt)
            r30 = hs[30].squeeze(0).float().cpu()
            pt_path = OUT_PT_DIR / f"{sid}_r{ridx}.pt"
            torch.save({"residuals": r30.half(), "mask": mask_cpu, "n_tokens": n}, str(pt_path))

            sids.append(sid); ridxs.append(ridx)
            means.append(mean_per_layer); lasts.append(last_per_layer)
            del out, hs, r_layer, r30; torch.cuda.empty_cache(); gc.collect()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [OOM] {sid}/r{ridx}: {e}", flush=True)
            torch.cuda.empty_cache(); continue
        except Exception as e:
            print(f"  [ERR] {sid}/r{ridx}: {type(e).__name__}: {e}", flush=True)
            continue
        n_done += 1
        if n_done % save_every == 0:
            np.savez(OUT_NPZ,
                     sample_ids=np.array(sids, dtype=object),
                     rewrite_idxs=np.array(ridxs, dtype=np.int64),
                     mean=np.stack(means), last=np.stack(lasts))
            print(f"  [{n_done}/{len(queue)}] {time.time()-t_start:.0f}s; saved {OUT_NPZ}", flush=True)

    np.savez(OUT_NPZ,
             sample_ids=np.array(sids, dtype=object),
             rewrite_idxs=np.array(ridxs, dtype=np.int64),
             mean=np.stack(means), last=np.stack(lasts))
    print(f"\nfinished. wrote {OUT_NPZ} ({len(sids)} entries)", flush=True)


if __name__ == "__main__":
    main()
