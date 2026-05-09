"""Extract residuals at 13 layers for the cyber_1 selection in selection.json.

Uses forward hooks on only the requested transformer blocks, so the model never
materialises the full (n_layers+1)-element `output_hidden_states` tuple.
This saves ~5 GB HBM on long prompts vs. experiment 02's `ActivationExtractor`,
which OOM'd on Gemma 4-31B at >7k tokens.

Sample list is read from selection.json (run select.py first). Resumes by
skipping any sample whose .pt already exists.

Output schema (per .pt):
  residuals       (n_selected_layers, N, d)  fp16
  input_ids       (N,) int32
  attention_mask  (N,) bool
  layer_idxs      list[int]
  n_tokens, fwd_seconds, peak_vram_gb, label, sample_id
"""
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
sys.path.insert(0, str(REPO_ROOT / "starter_code"))

from chunked_sdpa import chunked_sdpa_scope
from data import load_dataset

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
MODEL_PATH = "/home/ubuntu/models/Gemma-4-31B-it"
OUT_DIR = Path("/home/ubuntu/extracts/03_layer_sweep_omar")
SELECTION = Path(__file__).parent / "selection.json"
DEVICE = "cuda:0"


def find_decoder_layers(model):
    """Walk the model graph to find the text-decoder layer ModuleList."""
    candidates = [
        ("model.model.layers", lambda m: m.model.model.layers),
        ("model.language_model.model.layers", lambda m: m.model.language_model.model.layers),
        ("model.model.language_model.layers", lambda m: m.model.language_model.layers),
        ("model.layers", lambda m: m.model.layers),
    ]
    for name, getter in candidates:
        try:
            layers = getter(model)
            if hasattr(layers, "__len__") and len(layers) > 10:
                return name, layers
        except AttributeError:
            continue
    raise RuntimeError("Could not locate decoder layers on model")


def make_hooks(layers, requested_idxs, captured: dict, storage_dtype):
    """Install hooks so that requested_idxs[k] hidden state ends up in captured[k].

    Convention: idx 0 = embedding output (= input to layers[0]); idx i (1 <= i <= n_blocks-1)
    = output of layers[i-1] (= input to layers[i]); idx n_blocks = output of layers[-1].
    """
    n_blocks = len(layers)
    handles = []

    def pre_hook_factory(idx):
        def pre_hook(module, args, kwargs=None):
            # First positional arg is the residual stream (1, N, d)
            x = args[0] if args else (kwargs or {}).get("hidden_states")
            captured[idx] = x.detach().to("cpu", dtype=storage_dtype, copy=True)
        return pre_hook

    def post_hook_factory(idx):
        def post_hook(module, args, output):
            # Block returns either Tensor or tuple where [0] is the residual stream
            x = output[0] if isinstance(output, tuple) else output
            captured[idx] = x.detach().to("cpu", dtype=storage_dtype, copy=True)
        return post_hook

    for k in requested_idxs:
        if k > n_blocks:
            raise ValueError(f"layer idx {k} > n_blocks={n_blocks}")
        if k < n_blocks:
            handles.append(layers[k].register_forward_pre_hook(pre_hook_factory(k), with_kwargs=True))
        else:  # k == n_blocks → take output of last block
            handles.append(layers[-1].register_forward_hook(post_hook_factory(k)))
    return handles


def load_selected_samples():
    """Read selection.json and join against the cyber-train split for prompts."""
    sel = json.loads(SELECTION.read_text())
    train_by_id = {s["sample_id"]: s for s in load_dataset("cyber", split="train")}
    out = []
    for row in sel["samples"]:
        s = train_by_id.get(row["sample_id"])
        if s is None:
            raise SystemExit(f"selection has unknown sample_id {row['sample_id']!r}")
        out.append(s)
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_selected_samples()
    n_du = sum(1 for s in samples if s["label"] == "dual_use")
    n_be = sum(1 for s in samples if s["label"] == "benign")
    print(f"Selected {len(samples)} samples: dual_use={n_du}, benign={n_be}", flush=True)

    print(f"Loading {MODEL_PATH}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()

    layers_path, layers = find_decoder_layers(model)
    n_blocks = len(layers)
    print(f"Decoder layers: {layers_path} (n_blocks={n_blocks})", flush=True)
    print(f"Extracting {len(LAYERS)} layer(s): {LAYERS}", flush=True)

    metadata = {
        "model": "gemma4_31b",
        "model_path": MODEL_PATH,
        "n_blocks": n_blocks,
        "layer_idxs": LAYERS,
        "use_chat_template": True,
        "dtype": "fp16",
        "samples": [],
    }

    storage_dtype = torch.float16
    t_start = time.time()
    n_ok = n_skip = n_err = 0

    with chunked_sdpa_scope():
        for i, s in enumerate(samples):
            sid = s["sample_id"]
            out_path = OUT_DIR / f"{sid}.pt"
            if out_path.exists():
                n_skip += 1
                continue

            prompt = s["prompt"]
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            enc = tok(text, return_tensors="pt").to(DEVICE)
            ids, attn = enc.input_ids, enc.attention_mask

            captured = {}
            handles = make_hooks(layers, LAYERS, captured, storage_dtype)

            torch.cuda.synchronize(); torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=attn,
                          output_hidden_states=False, use_cache=False, return_dict=True)
                torch.cuda.synchronize()
                fwd = time.time() - t0
                peak = torch.cuda.max_memory_allocated() / 1024**3

                # Stack in the order of LAYERS
                stacked = torch.stack([captured[k][0] for k in LAYERS], dim=0)  # (L, N, d)
                N = stacked.shape[1]
                extract = {
                    "residuals": stacked.contiguous(),
                    "input_ids": ids[0].to("cpu", dtype=torch.int32),
                    "attention_mask": attn[0].to("cpu", dtype=torch.bool),
                    "n_tokens": int(N),
                    "layer_idxs": LAYERS,
                    "fwd_seconds": round(fwd, 3),
                    "peak_vram_gb": round(peak, 2),
                    "use_chat_template": True,
                    "label": s["label"],
                    "sample_id": sid,
                }
                torch.save(extract, str(out_path))
                n_ok += 1
                metadata["samples"].append({"sample_id": sid, "n_tokens": int(N),
                                            "fwd_seconds": fwd, "peak_vram_gb": round(peak, 2)})

                if (n_ok + n_skip) % 10 == 0 or (i + 1) == len(samples):
                    elapsed = time.time() - t_start
                    rate = (n_ok + n_skip) / max(elapsed, 1e-3)
                    eta = (len(samples) - (i + 1)) / max(rate, 1e-3) / 60
                    sz = out_path.stat().st_size / 1024**2
                    print(f"  [{i+1}/{len(samples)}] {sid}: N={N} "
                          f"fwd={fwd:.2f}s peak={peak:.1f}GB sz={sz:.0f}MB | "
                          f"{rate:.2f}/s eta={eta:.1f}min", flush=True)
            except Exception as e:
                n_err += 1
                print(f"  [{i+1}/{len(samples)}] {sid}: FAIL "
                      f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                metadata["samples"].append({"sample_id": sid, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            finally:
                for h in handles:
                    h.remove()
                captured.clear()
                torch.cuda.empty_cache()

    metadata["n_ok"] = n_ok
    metadata["n_skipped"] = n_skip
    metadata["n_errors"] = n_err
    (OUT_DIR / "extraction_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n=== DONE ok={n_ok} skipped={n_skip} errors={n_err} | "
          f"{(time.time() - t_start) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
