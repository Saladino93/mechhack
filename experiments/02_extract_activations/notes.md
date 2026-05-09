# Experiment 02: Extract Residual Activations

## Goal
Extract residual stream activations from Gemma 4-31B-it and Qwen 3.6-27B for all datasets (cyber probes + refusal probes). These activations are the input to all Level-1 probe methods.

## How to run

### List available datasets
```bash
python experiments/02_extract_activations/run.py --list
```

### Extract activations (GPU required)
```bash
# Gemma refusal probes — middle layer, all samples
python experiments/02_extract_activations/run.py \
    --dataset refusal_gemma4_31b \
    --model_key gemma4_31b \
    --layers middle

# Cyber probes — multiple layers, first 100 samples
python experiments/02_extract_activations/run.py \
    --dataset cyber \
    --model_key gemma4_31b \
    --layers "10,30,50" \
    --limit 100

# Qwen refusal probes — custom output dir
python experiments/02_extract_activations/run.py \
    --dataset refusal_qwen36 \
    --model_key qwen36 \
    --out_dir ./extracts/qwen36_refusal

# Quick smoke test (2 samples)
python experiments/02_extract_activations/run.py \
    --dataset refusal_gemma4_31b \
    --model_key gemma4_31b \
    --limit 2
```

### Examples (no GPU needed for data demo)
```bash
# Data loading demo — see how datasets are loaded
python experiments/02_extract_activations/example.py --demo data

# Extraction demo — needs GPU
python experiments/02_extract_activations/example.py --demo extract \
    --model_key gemma4_31b --limit 2

# Inspect a saved .pt file
python experiments/02_extract_activations/example.py --demo inspect \
    --pt_file ./extracts/gemma4_31b_refusal_gemma4_31b/ref_gemma4_31b_pos_0001.pt
```

### Using as a library in other experiments
```python
import sys; sys.path.insert(0, "experiments/02_extract_activations")
from data import load_dataset, get_label_for_task
from extractor import ActivationExtractor

# Load samples
samples = load_dataset("refusal_gemma4_31b", split="train")

# Create extractor and load model
ext = ActivationExtractor(model_key="gemma4_31b", layers="middle")
ext.load_model()

# Extract single sample
result = ext.extract_single(samples[0]["prompt"])
print(result["residuals"].shape)  # (1, n_tokens, 3584)
```

## Outputs
- `extracts/<model_key>_<dataset>/` — one `.pt` file per sample
- `extracts/<model_key>_<dataset>/extraction_metadata.json` — run metadata

### .pt file contents
```
residuals       (n_layers, n_tokens, d_model)  fp16 — the activations
input_ids       (n_tokens,)                    int32
attention_mask  (n_tokens,)                    bool
n_tokens        int
layer_idxs      list[int]
fwd_seconds     float
peak_vram_gb    float
label           raw label (bool or str)
sample_id       str
```

## Module structure
- `data.py` — Dataset loading with uniform schema across cyber/refusal probes
- `extractor.py` — `ActivationExtractor` class: model loading, forward pass, saving
- `run.py` — CLI entry point
- `example.py` — Runnable examples for all three workflows

## Method
Forward pass with `output_hidden_states=True`, saving selected layer activations per sample as `.pt` files. Uses chunked SDPA for Gemma (head_dim=512 incompatible with flash-attn). Auto-resolves model path: cluster `/data/`, local `./models/`, or HF cache. Resumes automatically (skips existing `.pt` files).

## Results
(To be filled after running on cluster)

## Takeaways
(To be filled)
