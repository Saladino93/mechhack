"""Extract Gemma 4-31B-it residuals for cyber + refusal + rewrites_k7 paraphrases.

Loads the model once, walks four datasets sequentially:

  cyber_train  (5079 rows, datasets/cyber_probes/train.jsonl)        -> /home/ubuntu/extracts/gemma_cyber/
  cyber_test   (2180 rows, datasets/cyber_probes/test.jsonl)         -> /home/ubuntu/extracts/gemma_cyber/
  refusal      ( 878 rows, refusal_probes/gemma4_31b/attacks_full)   -> /home/ubuntu/extracts/gemma_refusal/
  rewrites     ( 567 rows, rewrites_k7.json flattened, sample_ids
                 like EVO_0012_rw3)                                    -> /home/ubuntu/extracts/gemma_rewrites/

Layer schedule: 0:65:5 = [0,5,10,...,60] (matches Qwen exp 31).
Resumes by skipping samples whose .pt already exists.
"""
import os, sys, json, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "starter_code"))
from chunked_sdpa import chunked_sdpa_scope  # noqa: E402

MODEL_PATH = "/home/ubuntu/models/Gemma-4-31B-it"
LAYERS_SPEC = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
DTYPE = torch.float16

DATASETS = [
    # Smallest first so any code bugs surface fast
    {"name": "rewrites",
     "jsonl": REPO / "datasets/refusal_probes/gemma4_31b/rewrites_k7.json",
     "out": Path("/home/ubuntu/extracts/gemma_rewrites"),
     "prompt_key": "_rewrite_prompt", "label_key": None,
     "is_rewrites": True},
    {"name": "refusal",
     "jsonl": REPO / "datasets/refusal_probes/gemma4_31b/attacks_full.jsonl",
     "out": Path("/home/ubuntu/extracts/gemma_refusal"),
     "prompt_key": "attack_prompt", "label_key": "is_refusal"},
    {"name": "cyber_test",
     "jsonl": REPO / "datasets/cyber_probes/test.jsonl",
     "out": Path("/home/ubuntu/extracts/gemma_cyber"),
     "prompt_key": "prompt", "label_key": "category"},
    {"name": "cyber_train",
     "jsonl": REPO / "datasets/cyber_probes/train.jsonl",
     "out": Path("/home/ubuntu/extracts/gemma_cyber"),
     "prompt_key": "prompt", "label_key": "category"},
]


def load_samples(spec):
    p = spec["jsonl"]
    if spec.get("is_rewrites"):
        items = json.loads(p.read_text())
        out = []
        for item in items:
            base_id = item["sample_id"]
            for i, rw in enumerate(item["rewrites"]):
                out.append({"sample_id": f"{base_id}_rw{i}",
                             "_rewrite_prompt": rw,
                             "_origin_sample_id": base_id})
        return out
    rows = []
    for line in open(p):
        rows.append(json.loads(line))
    return rows


def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0", trust_remote_code=True)
    model.eval()
    return model, tok


def parse_layers(spec, n_layers):
    return [i for i in spec if 0 <= i <= n_layers]


def extract_one(model, tok, sample, prompt, out_path, layer_idxs):
    enc = tok(prompt, return_tensors="pt").to("cuda:0")
    ids, attn = enc.input_ids, enc.attention_mask
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn,
                    output_hidden_states=True, return_dict=True)
    torch.cuda.synchronize()
    fwd = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    hs = out.hidden_states
    stacked = torch.stack([hs[k][0] for k in layer_idxs], dim=0)
    N = stacked.shape[1]
    stacked = stacked.to("cpu", dtype=DTYPE).contiguous()
    extract = {
        "residuals": stacked,
        "input_ids": ids[0].to("cpu", dtype=torch.int32),
        "attention_mask": attn[0].to("cpu", dtype=torch.bool),
        "n_tokens": int(N),
        "layer_idxs": layer_idxs,
        "fwd_seconds": round(fwd, 3),
        "peak_vram_gb": round(peak, 2),
        "label": sample.get("label_value", -1),
    }
    del out, hs, stacked
    torch.cuda.empty_cache()
    torch.save(extract, str(out_path))
    return fwd, peak, N


def run_dataset(model, tok, layer_idxs, spec, use_ct=True):
    print(f"\n=== {spec['name']} ===  jsonl={spec['jsonl']}  out={spec['out']}", flush=True)
    spec["out"].mkdir(parents=True, exist_ok=True)
    samples = load_samples(spec)
    print(f"  loaded {len(samples)} samples", flush=True)
    metadata_path = spec["out"] / f"{spec['name']}_metadata.json"
    metadata = {"name": spec["name"], "model": "gemma4_31b",
                "n_samples": len(samples), "layer_idxs": layer_idxs,
                "samples": []}
    t_start = time.time()
    n_done = n_skip = n_fail = 0
    for i, s in enumerate(samples):
        sid = s["sample_id"]
        out_path = spec["out"] / f"{sid}.pt"
        if out_path.exists():
            n_skip += 1; continue
        prompt = s.get(spec["prompt_key"])
        if not prompt:
            n_fail += 1; continue
        # tokenize with chat template if it's a refusal-style attack prompt
        txt = (tok.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
               if use_ct else prompt)
        # save the right label
        if spec.get("label_key"):
            v = s.get(spec["label_key"])
            if isinstance(v, bool): s["label_value"] = int(v)
            elif v is None: s["label_value"] = -1
            else: s["label_value"] = v   # strings (cyber category) saved as-is
        try:
            fwd, peak, N = extract_one(model, tok, s, txt, out_path, layer_idxs)
            n_done += 1
            metadata["samples"].append({"sample_id": sid, "n_tokens": int(N),
                                          "fwd_s": round(fwd, 2)})
            if (i+1) % 25 == 0 or (i+1) == len(samples):
                elapsed = time.time() - t_start
                rate = max(n_done, 1) / max(elapsed, 1e-3)
                eta = (len(samples) - (i+1)) / max(rate, 1e-3) / 60
                sz = out_path.stat().st_size / 1024**2
                print(f"  [{i+1}/{len(samples)}] {sid}: N={N} fwd={fwd:.2f}s "
                      f"peak={peak:.1f}GB sz={sz:.0f}MB | {rate:.2f}/s "
                      f"eta={eta:.1f}min  done={n_done} skip={n_skip} fail={n_fail}",
                      flush=True)
        except Exception as e:
            n_fail += 1
            metadata["samples"].append({"sample_id": sid,
                                          "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"  [{i+1}/{len(samples)}] {sid}: FAIL "
                  f"{type(e).__name__}: {str(e)[:150]}", flush=True)
            # Aggressive cleanup so an OOM doesn't poison the next sample
            import gc; gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"=== {spec['name']} done: ok={n_done} skip={n_skip} fail={n_fail} "
          f"in {(time.time()-t_start)/60:.1f}min ===", flush=True)


def main():
    print(f"Loading Gemma...", flush=True)
    # Gemma 4 has interleaved sliding-window (head_dim=256) + global
    # (head_dim=512) attention. text_config head_dim only reports the
    # majority case; chunked_sdpa is required either way for the global
    # layers and to keep long-sequence memory bounded.
    cfg = json.load(open(f"{MODEL_PATH}/config.json"))
    n_layers = cfg["text_config"]["num_hidden_layers"]
    print(f"  config: n_layers={n_layers}  enabling chunked_sdpa "
          f"unconditionally for Gemma", flush=True)

    model, tok = load_model()
    layer_idxs = parse_layers(LAYERS_SPEC, n_layers)
    print(f"Gemma loaded | n_layers={n_layers} | extracting "
          f"{len(layer_idxs)} layers: {layer_idxs}", flush=True)

    with chunked_sdpa_scope():
        for spec in DATASETS:
            run_dataset(model, tok, layer_idxs, spec)

    print("\n=== ALL DATASETS DONE ===", flush=True)


if __name__ == "__main__":
    main()
