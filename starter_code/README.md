# Starter Code

Minimal baselines for each track. **Don't** treat these as the answer — they're floors to beat.

## Files

| File | Track | Purpose |
|---|---|---|
| `extract_residuals.py` | 1 | GPU helper: dump `(n_selected_layers, n_tokens, d_model)` per rollout. Layer choice in `extract_config.json` — default is one middle layer; supports list, range, or `"all"`. |
| `extract_config.json`  | 1 | Config for the extractor (model_key, layer-spec, dtype, output dir) |
| `train_probe.py` | 1 | Linear / attention / all-layer-concat probe trainers |
| `grad_input_baseline.py` | 2 | Reference attribution method (full-LLM grad×input) |
| `iterative_edit_agent.py` | 3 | Reference 5-iteration edit agent |
| `llm_clients.py` | 3 | OpenRouter + AIaaS wrappers with schema forcing |
| `chunked_sdpa.py` | shared | SDPA monkey-patch for Gemma's `head_dim=512` (won't fit flash-attn, won't fit math backend on long sequences) |

## Quick start

```bash
pip install -r ../requirements.txt
export OPENROUTER_KEY=sk-or-v1-...        # if you want OpenRouter editor
export AIAAS_KEY=sk--...                  # if you want EPFL AIaaS

# Extract residuals — defaults to ONE middle layer.
# Edit extract_config.json for a different layer set, or override with the LAYERS env var:
LAYERS="middle"        python extract_residuals.py     # default — single middle layer
LAYERS="10,30,50"      python extract_residuals.py     # 3 explicit layers
LAYERS="0:65:8"        python extract_residuals.py     # every 8th layer (start:stop:step)
LAYERS="all"           python extract_residuals.py     # every layer (~65× the disk)

# Train a baseline probe
python train_probe.py --model_key gemma4_31b --variant middle --task refusal

# Run baseline attribution + edit agent end-to-end
python iterative_edit_agent.py \
    --probe_path ./probe_gemma4_31b_middle.pt \
    --eval_set ../datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \
    --max_iters 5 \
    --editor_endpoint openrouter \
    --editor_model deepseek/deepseek-v4-pro \
    --output ./baseline_results.jsonl
```

## A100-80GB recipe

```python
import os, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, ...)
model.eval()
for p in model.parameters(): p.requires_grad_(False)
model.gradient_checkpointing_enable()        # halves activation memory
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()       # required for grads-on-input-embeddings

# For Gemma 4-31B, needed:
from chunked_sdpa import chunked_sdpa_scope
with chunked_sdpa_scope():
    out = model(inputs_embeds=embs, attention_mask=attn,
                output_hidden_states=True, return_dict=True)
```

## Ideas your baseline doesn't try

- Probe ensembles
- Distilled probes
- Counterfactual editing (Wachter et al.)
- Beam search across edit trajectories
- RL fine-tuning of the edit agent against the probe-flip reward (careful — must keep intent judge as a hard gate)

Make us beat by 30%.
