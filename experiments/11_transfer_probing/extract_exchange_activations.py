"""Extract residual activations for full user+assistant exchanges.

`experiments/02_extract_activations/run.py` extracts prompt-side activations by
wrapping a single user message with `add_generation_prompt=True`. That is enough
for prompt-side refusal/compliance analysis, but it cannot answer questions like:

  * when does the generated assistant response become refusal-like?
  * do compliance failures drift during generation?

This script extracts activations for a full exchange:

    user prompt + assistant response

and saves token spans so downstream analysis can focus on prompt tokens,
assistant tokens, or all tokens.

Input is a JSONL file. Each row should contain at least:
  * sample_id
  * prompt field, e.g. `prompt` or `attack_prompt`
  * response field, e.g. `response`, `completion`, `assistant_response`
  * label field, e.g. `label` or `is_refusal`

Example:
    python experiments/04_failure_analysis/extract_exchange_activations.py \
      --input_jsonl ./datasets/refusal_probes/gemma4_31b/attacks_full.jsonl \
      --model_key gemma4_31b \
      --layers "0:65:4" \
      --prompt_field attack_prompt \
      --response_field model_response \
      --label_field is_refusal \
      --out_dir ./extracts/gemma4_31b_refusal_full_exchange
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents, Path.cwd(), *Path.cwd().parents]:
        if (p / "experiments" / "02_extract_activations" / "extractor.py").exists():
            return p
    return Path.cwd()


def import_extractor():
    root = repo_root()
    ext_dir = root / "experiments" / "02_extract_activations"
    if not (ext_dir / "extractor.py").exists():
        raise ImportError("Could not find experiments/02_extract_activations/extractor.py")
    sys.path.insert(0, str(ext_dir))
    from extractor import ActivationExtractor  # type: ignore
    return ActivationExtractor


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_present(row: dict, candidates: list[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in row:
            raise KeyError(f"Field {explicit!r} not present. Available keys: {sorted(row)}")
        return explicit
    for c in candidates:
        if c in row:
            return c
    raise KeyError(f"None of {candidates} present. Available keys: {sorted(row)}")


def as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def build_chat_text_and_spans(tokenizer, prompt: str, response: str, add_generation_prompt_for_prefix: bool = True):
    """Return `(full_text, input_ids, attention_mask, spans)` for user+assistant.

    We estimate `assistant_start` by tokenizing the user-only chat with
    `add_generation_prompt=True`; this is the point at which assistant response
    tokens begin in most chat templates.
    """
    # Full exchange.
    messages_full = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    full_text = tokenizer.apply_chat_template(
        messages_full,
        tokenize=False,
        add_generation_prompt=False,
    )
    enc_full = tokenizer(full_text, return_tensors="pt")

    # Prefix up to assistant response start.
    messages_prefix = [{"role": "user", "content": prompt}]
    prefix_text = tokenizer.apply_chat_template(
        messages_prefix,
        tokenize=False,
        add_generation_prompt=add_generation_prompt_for_prefix,
    )
    enc_prefix = tokenizer(prefix_text, return_tensors="pt")
    assistant_start = min(enc_prefix.input_ids.shape[1], enc_full.input_ids.shape[1])

    spans = {
        "prompt": [0, assistant_start],
        "user": [0, assistant_start],
        "assistant": [assistant_start, int(enc_full.input_ids.shape[1])],
        "assistant_start": assistant_start,
        "assistant_end": int(enc_full.input_ids.shape[1]),
        "n_tokens_full": int(enc_full.input_ids.shape[1]),
    }
    return full_text, enc_full.input_ids, enc_full.attention_mask, spans


def extract_from_ids(extractor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    """Run extractor.model on already-tokenized IDs and return extract dict."""
    assert extractor.model is not None, "Call extractor.load_model() first"
    device = extractor.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    if extractor._needs_chunked_sdpa():
        from chunked_sdpa import chunked_sdpa_scope
        ctx = chunked_sdpa_scope()
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        with torch.no_grad():
            out = extractor.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fwd_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

    hidden_states = out.hidden_states
    stacked = torch.stack([hidden_states[k][0] for k in extractor.layer_idxs], dim=0)
    result = {
        "residuals": stacked.to("cpu", dtype=extractor.storage_dtype).contiguous(),
        "input_ids": input_ids[0].to("cpu", dtype=torch.int32),
        "attention_mask": attention_mask[0].to("cpu", dtype=torch.bool),
        "n_tokens": int(stacked.shape[1]),
        "layer_idxs": list(extractor.layer_idxs),
        "fwd_seconds": round(fwd_time, 3),
        "peak_vram_gb": round(peak_vram, 2),
    }
    del out, hidden_states, stacked
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_key", default="gemma4_31b", choices=["gemma4_31b", "qwen36"])
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--layers", default="0:65:4")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt_field", default=None)
    ap.add_argument("--response_field", default=None)
    ap.add_argument("--label_field", default=None)
    ap.add_argument("--sample_id_field", default="sample_id")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ActivationExtractor = import_extractor()
    rows = load_jsonl(Path(args.input_jsonl).expanduser())
    if args.limit > 0:
        rows = rows[: args.limit]

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = ActivationExtractor(
        model_key=args.model_key,
        layers=args.layers,
        dtype=args.dtype,
        use_chat_template=False,  # we build the full chat text ourselves
        model_path=args.model_path,
        device=args.device,
    )
    extractor.load_model()

    metadata = extractor.get_metadata()
    metadata.update({
        "input_jsonl": str(Path(args.input_jsonl).expanduser()),
        "mode": "full_exchange_user_assistant",
        "n_rows_requested": len(rows),
        "samples": [],
    })

    n_ok = n_skip = n_err = 0
    t_start = time.time()
    for i, row in enumerate(rows):
        try:
            sid = str(row.get(args.sample_id_field) or row.get("id") or f"row_{i:06d}")
            out_path = out_dir / f"{sid}.pt"
            if out_path.exists() and not args.overwrite:
                n_skip += 1
                continue
            p_field = first_present(row, ["prompt", "attack_prompt", "user", "question"], args.prompt_field)
            r_field = first_present(row, ["response", "completion", "assistant_response", "model_response", "answer"], args.response_field)
            l_field = first_present(row, ["label", "is_refusal", "category"], args.label_field)
            prompt = as_text(row[p_field])
            response = as_text(row[r_field])
            label = row[l_field]

            _, input_ids, attention_mask, spans = build_chat_text_and_spans(extractor.tokenizer, prompt, response)
            result = extract_from_ids(extractor, input_ids, attention_mask)
            result.update({
                "sample_id": sid,
                "label": label,
                "spans": spans,
                "fields": {
                    "prompt_field": p_field,
                    "response_field": r_field,
                    "label_field": l_field,
                },
                "metadata": {k: v for k, v in row.items() if k not in {p_field, r_field}},
            })
            torch.save(result, str(out_path))
            n_ok += 1
            metadata["samples"].append({
                "sample_id": sid,
                "n_tokens": result["n_tokens"],
                "assistant_tokens": spans["assistant_end"] - spans["assistant_start"],
                "fwd_seconds": result["fwd_seconds"],
            })
            if (n_ok + n_skip) % 10 == 0 or i + 1 == len(rows):
                elapsed = time.time() - t_start
                print(
                    f"[{i+1}/{len(rows)}] ok={n_ok} skip={n_skip} err={n_err} "
                    f"last={sid} N={result['n_tokens']} assistant={spans['assistant_end'] - spans['assistant_start']} "
                    f"elapsed={elapsed/60:.1f}m",
                    flush=True,
                )
        except Exception as e:
            n_err += 1
            print(f"[{i+1}/{len(rows)}] FAIL {type(e).__name__}: {str(e)[:240]}", flush=True)
            metadata["samples"].append({"row": i, "error": f"{type(e).__name__}: {str(e)[:240]}"})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metadata.update({
        "n_ok": n_ok,
        "n_skipped": n_skip,
        "n_errors": n_err,
        "total_seconds": round(time.time() - t_start, 1),
    })
    meta_path = out_dir / "exchange_extraction_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Done: ok={n_ok} skipped={n_skip} errors={n_err}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
