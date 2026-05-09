# Experiment 02: Extract Residual Activations

## Goal
Extract residual stream activations from Gemma 4-31B-it and Qwen 3.6-27B for all datasets (cyber probes + refusal probes). These activations are the input to all Level-1 probe methods.

## Method
Forward pass with `output_hidden_states=True`, saving selected layer activations per sample as `.pt` files. Uses chunked SDPA for Gemma (head_dim=512 incompatible with flash-attn).

### Module structure
- `data.py` — Dataset loading with uniform schema across cyber/refusal probes
- `extractor.py` — `ActivationExtractor` class: model loading, forward pass, saving
- `run.py` — CLI entry point

### CLI examples
```bash
# List datasets
python experiments/02_extract_activations/run.py --list

# Extract Gemma refusal probes (middle layer)
python experiments/02_extract_activations/run.py \
    --dataset refusal_gemma4_31b --model_key gemma4_31b --layers middle

# Extract cyber probes with multiple layers
python experiments/02_extract_activations/run.py \
    --dataset cyber --model_key gemma4_31b --layers "10,30,50"

# Quick test (2 samples only)
python experiments/02_extract_activations/run.py \
    --dataset refusal_gemma4_31b --model_key gemma4_31b --limit 2
```

## Results
(To be filled after running on cluster)

## Takeaways
(To be filled)
