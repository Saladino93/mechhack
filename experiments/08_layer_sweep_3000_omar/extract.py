"""Extract residuals at 13 layers for the cyber_1 selection in selection.json.

Adapted from experiments/03_layer_sweep_omar/extract.py and
experiments/06_cyber2_extract_omar/extract.py — same hook-based extractor +
chunked_sdpa_scope. Writes into the SHARED extracts dir
/home/ubuntu/extracts/cyber_all_omar/, the convention chosen by the previous
agent.

Resume / re-use logic:
  - For each selection-id, if .pt already exists in cyber_all_omar -> skip.
  - Otherwise, if .pt exists in /home/ubuntu/extracts/03_layer_sweep_omar/
    (the original exp 03 dir) -> create a relative symlink in cyber_all_omar
    pointing to it. No GPU work.
  - Otherwise -> run a fresh GPU extraction.

OOM handling: each forward pass is wrapped in try/except. OOM'd samples are
logged to extraction_metadata.json with `error` field and skipped — no .pt
on disk. train_probes.py auto-excludes anything not on disk.

Output schema (per .pt) — identical to exp 03 / 06 / 07:
  residuals       (n_selected_layers, N, d)  fp16
  input_ids       (N,) int32
  attention_mask  (N,) bool
  layer_idxs      list[int]
  n_tokens, fwd_seconds, peak_vram_gb, label, sample_id
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "02_extract_activations"))
sys.path.insert(0, str(REPO_ROOT / "starter_code"))

from chunked_sdpa import chunked_sdpa_scope
from data import load_dataset

LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
MODEL_PATH = "/home/ubuntu/models/Gemma-4-31B-it"
OUT_DIR = Path("/home/ubuntu/extracts/cyber_all_omar")
SOURCE_DIR_03 = Path("/home/ubuntu/extracts/03_layer_sweep_omar")
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
    """Install hooks so requested_idxs[k] hidden state ends up in captured[k]."""
    n_blocks = len(layers)
    handles = []

    def pre_hook_factory(idx):
        def pre_hook(module, args, kwargs=None):
            x = args[0] if args else (kwargs or {}).get("hidden_states")
            captured[idx] = x.detach().to("cpu", dtype=storage_dtype, copy=True)
        return pre_hook

    def post_hook_factory(idx):
        def post_hook(module, args, output):
            x = output[0] if isinstance(output, tuple) else output
            captured[idx] = x.detach().to("cpu", dtype=storage_dtype, copy=True)
        return post_hook

    for k in requested_idxs:
        if k > n_blocks:
            raise ValueError(f"layer idx {k} > n_blocks={n_blocks}")
        if k < n_blocks:
            handles.append(layers[k].register_forward_pre_hook(pre_hook_factory(k), with_kwargs=True))
        else:
            handles.append(layers[-1].register_forward_hook(post_hook_factory(k)))
    return handles


def load_selected_samples():
    sel = json.loads(SELECTION.read_text())
    by_id = {}
    for split in ("train", "test"):
        for s in load_dataset("cyber", split=split):
            by_id[s["sample_id"]] = s
    out = []
    for row in sel["samples"]:
        s = by_id.get(row["sample_id"])
        if s is None:
            raise SystemExit(f"selection has unknown sample_id {row['sample_id']!r}")
        s["_split"] = row["split"]
        out.append(s)
    return out


def reuse_via_symlink(samples):
    """For any selection-id whose .pt is missing in OUT_DIR but present in
    SOURCE_DIR_03, create a relative symlink. Returns a list of newly linked IDs.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    linked = []
    for s in samples:
        sid = s["sample_id"]
        dst = OUT_DIR / f"{sid}.pt"
        if dst.exists() or dst.is_symlink():
            continue
        src = SOURCE_DIR_03 / f"{sid}.pt"
        if not src.exists():
            continue
        # Use absolute path target — matches the convention already in OUT_DIR.
        os.symlink(str(src), str(dst))
        linked.append(sid)
    return linked


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_selected_samples()
    label_counts = Counter(s["label"] for s in samples)
    split_counts = Counter(s["_split"] for s in samples)
    print(f"Selected {len(samples)} samples: {dict(label_counts)} | "
          f"split={dict(split_counts)}", flush=True)

    # 1) Symlink anything we already have in 03_layer_sweep_omar.
    linked = reuse_via_symlink(samples)
    print(f"Symlinked {len(linked)} pre-existing extracts from {SOURCE_DIR_03}", flush=True)

    # 2) Whatever is still missing needs fresh extraction.
    todo = [s for s in samples if not (OUT_DIR / f"{s['sample_id']}.pt").exists()]
    n_already = len(samples) - len(todo)
    print(f"On disk: {n_already}; need fresh extraction: {len(todo)}", flush=True)

    if not todo:
        print("Nothing to extract. Done.", flush=True)
        # Still write metadata so downstream knows the layer schema.
        meta = {
            "model": "gemma4_31b", "model_path": MODEL_PATH, "layer_idxs": LAYERS,
            "use_chat_template": True, "dtype": "fp16", "task": "cyber_1",
            "selection_file": str(SELECTION), "out_dir": str(OUT_DIR),
            "n_selected": len(samples), "n_already_on_disk": n_already,
            "n_symlinked_this_run": len(linked),
            "n_extracted_this_run": 0, "n_errors": 0, "samples": [],
        }
        (Path(__file__).parent / "extraction_metadata.json").write_text(json.dumps(meta, indent=2))
        return

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
        "task": "cyber_1",
        "selection_file": str(SELECTION),
        "out_dir": str(OUT_DIR),
        "n_selected": len(samples),
        "n_already_on_disk_at_start": n_already,
        "n_symlinked_this_run": len(linked),
        "samples": [],
    }

    storage_dtype = torch.float16
    t_start = time.time()
    n_ok = n_err = 0
    peak_overall = 0.0
    oom_ids = []

    with chunked_sdpa_scope():
        for i, s in enumerate(todo):
            sid = s["sample_id"]
            out_path = OUT_DIR / f"{sid}.pt"
            if out_path.exists():
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
                peak_overall = max(peak_overall, peak)

                stacked = torch.stack([captured[k][0] for k in LAYERS], dim=0)
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
                                            "fwd_seconds": round(fwd, 3),
                                            "peak_vram_gb": round(peak, 2),
                                            "label": s["label"]})

                if (n_ok + n_err) % 10 == 0 or (i + 1) == len(todo):
                    elapsed = time.time() - t_start
                    rate = (n_ok + n_err) / max(elapsed, 1e-3)
                    eta = (len(todo) - (i + 1)) / max(rate, 1e-3) / 60
                    sz = out_path.stat().st_size / 1024**2
                    print(f"  [{i+1}/{len(todo)}] {sid}: N={N} "
                          f"fwd={fwd:.2f}s peak={peak:.1f}GB sz={sz:.0f}MB | "
                          f"{rate:.2f}/s eta={eta:.1f}min", flush=True)
            except torch.cuda.OutOfMemoryError as e:
                n_err += 1
                oom_ids.append(sid)
                print(f"  [{i+1}/{len(todo)}] {sid}: OOM "
                      f"prompt_chars={len(prompt)} :: {str(e)[:120]}", flush=True)
                metadata["samples"].append({"sample_id": sid, "label": s["label"],
                                            "prompt_chars": len(prompt),
                                            "error": f"OutOfMemoryError: {str(e)[:200]}"})
            except Exception as e:
                n_err += 1
                print(f"  [{i+1}/{len(todo)}] {sid}: FAIL "
                      f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                metadata["samples"].append({"sample_id": sid, "label": s["label"],
                                            "error": f"{type(e).__name__}: {str(e)[:200]}"})
            finally:
                for h in handles:
                    h.remove()
                captured.clear()
                torch.cuda.empty_cache()

    metadata["n_ok"] = n_ok
    metadata["n_errors"] = n_err
    metadata["oom_ids"] = oom_ids
    metadata["wall_minutes"] = round((time.time() - t_start) / 60, 2)
    metadata["peak_vram_gb_overall"] = round(peak_overall, 2)

    META_OUT = Path(__file__).parent / "extraction_metadata.json"
    META_OUT.write_text(json.dumps(metadata, indent=2))
    print(f"\n=== DONE ok={n_ok} errors={n_err} oom={len(oom_ids)} | "
          f"{metadata['wall_minutes']:.1f} min | "
          f"peak VRAM {peak_overall:.1f} GB ===", flush=True)


if __name__ == "__main__":
    main()
