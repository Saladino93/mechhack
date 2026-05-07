# Starter Code

Minimal baselines for each level. Treat these as floors to build past, not as the answer.

## Files

| File | Level | Purpose |
|---|---|---|
| `download_models.py` | shared | Pulls `google/gemma-4-31B-it` and `Qwen/Qwen3.6-27B` from HF Hub into `./models/` |
| `extract_residuals.py` | 1 | GPU helper: dump `(n_selected_layers, n_tokens, d_model)` per rollout. Layer choice in `extract_config.json` — default is one middle layer; supports list, range, or `"all"`. CLI: `--model_path --layers --out_dir --samples_file`. |
| `extract_config.json`  | 1 | Default config for the extractor (model_key, layer-spec, dtype, output dir). CLI args > env > config file. |
| `train_probe.py` | 1 | Linear / MLP / attention probe trainer. CLI: `--extracts_dir --manifest --out_dir --task`. |
| `grad_input_baseline.py` | 2 | Reference attribution method (full-LLM grad×input). CLI: `--model_path --probe_weights --extracts_dir --out_dir --variant`. |
| `iterative_edit_agent.py` | 2 | Reference 5-iteration edit agent |
| `llm_clients.py` | 2 | OpenRouter + AIaaS wrappers with schema forcing |
| `chunked_sdpa.py` | shared | SDPA monkey-patch for Gemma's `head_dim=512` (won't fit flash-attn, won't fit math backend on long sequences) |

## Quick start

```bash
pip install -r ../requirements.txt

# 1. Download model weights (or skip if already present in ./models/ or HF cache)
export HF_TOKEN=hf_...
python download_models.py --out_dir ../models

# 2. Extract residuals — defaults to ONE middle layer
python extract_residuals.py \
    --model_path ../models/Gemma-4-31B-it \
    --out_dir   ./extracts/gemma4_31b \
    --layers    middle

# Other layer-spec options:
#   --layers "10,30,50"      explicit list
#   --layers "0:65:8"        python-range (start:stop:step)
#   --layers "all"           every layer (~65× the disk)

# 3. Train a probe (consumes the extracts above)
python train_probe.py \
    --extracts_dir ./extracts/gemma4_31b \
    --manifest    ./extracts/gemma4_31b/extraction_metadata.json \
    --out_dir     ./probes \
    --task        refusal_gemma4_31b

# 4. Reference attribution baseline
python grad_input_baseline.py \
    --model_path    ../models/Gemma-4-31B-it \
    --probe_weights ./probes/weights/refusal_gemma4_31b_attention.pt \
    --extracts_dir  ./extracts/gemma4_31b \
    --out_dir       ./edit_eval

# 5. Iterative edit agent (Level 2 baseline)
export OPENROUTER_KEY=sk-or-v1-...        # or AIAAS_KEY for EPFL AIaaS
python iterative_edit_agent.py \
    --probe_path  ./probes/weights/refusal_gemma4_31b_attention.pt \
    --eval_set    ../datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \
    --max_iters   5 \
    --editor_endpoint openrouter \
    --editor_model    deepseek/deepseek-v4-pro \
    --output      ./baseline_results.jsonl
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
