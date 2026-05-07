# Level 1 — Probe Training Rules

## What you're training

Five binary classifiers over residual-stream activations of a frozen LLM:

### Cyber probes (graded harm levels)

| Probe | Positive (label=1) | Negative (label=0) |
|---|---|---|
| **Probe-1** (low risk) | `dual_use` | `benign` |
| **Probe-2** (mid risk) | `high_risk_dual_use` | `dual_use` ∪ `benign` |
| **Probe-3** (high risk) | `prohibited` | `high_risk_dual_use` ∪ `dual_use` ∪ `benign` |

Each is a separate probe with its own train/test set drawn from the master split.

### Refusal probes (per LLM target)

| Probe | Positive | Negative |
|---|---|---|
| **Refusal-Gemma** | model output classified `refusal` | classified `compliance` |
| **Refusal-Qwen** | same, on Qwen 3.6-27B | same |

## Data

`datasets/cyber_probes/` and `datasets/refusal_probes/{gemma4_31b,qwen36}/` contain the rollouts. Each row has:

```jsonc
{
  "sample_id": "...",
  "model_key": "gemma4_31b" | "qwen36",
  "prompt": "...",                   // the attack text
  "category": "benign" | "dual_use" | "high_risk_dual_use" | "prohibited",  // cyber only
  "label": 0 | 1,                    // refusal only (1=refusal, 0=compliance)
  "split": "train" | "test"          // FIXED. Do not change.
}
```

**Splits are deterministic and immutable.** Each rollout has `split: "train" | "test"` baked in. Computed once via `hash(sample_id) mod 100 < 70 → train, else test`. Do NOT redefine.

For each probe:
- Train data = rollouts in `split=="train"` whose category matches the probe's positive or negative set.
- Test data = same filter on `split=="test"`.

Different probes therefore have different test sets — that's intentional. Each probe's AUC is reported on its own test set.

## Architectural freedom

You choose:
- **Layer** to attach to (early / middle / late / all-layer-concat / mixture)
- **Probe arch** (linear, MLP, attention probe, learned-pool, …)
- **Hyperparameters**
- **Optimizer & schedule**
- **Number of training steps**

You **must** keep the model frozen — the LLM weights cannot be fine-tuned.

## Training data extraction (provided)

We provide a `starter_code/extract_residuals.py` GPU script that:
- Loads Gemma 4-31B-it / Qwen 3.6-27B from `--model_path` (download via `starter_code/download_models.py`, or auto-resolve from `./models/<repo-name>` / HF cache).
- For each rollout, runs `forward(input_ids, output_hidden_states=True)`.
- Saves per-sample `.pt` files containing `residuals: (n_selected_layers, n_tokens, d_model)` at fp16, plus `input_ids`, `attention_mask`, `layer_idxs`, `label`.

**Layer selection is configurable** via `starter_code/extract_config.json` (or the `LAYERS` env var). Default is **one layer at the middle of the stack** (`"middle"`). Other valid forms:

| Spec               | Resolves to                                       |
|--------------------|---------------------------------------------------|
| `"middle"`         | `[n_layers // 2]` — 1 layer (default)            |
| `"early"`          | `[n_layers // 4]` — 1 layer                       |
| `"late"`           | `[3 * n_layers // 4]` — 1 layer                   |
| `"32"`             | single explicit index                             |
| `"10,30,50"`       | explicit list                                     |
| `"0:65:8"`         | python-range `range(start, stop, step)`           |
| `"all"`            | every layer (embedding output + each block)       |

Storage scales linearly with `len(layer_idxs)`. The default 1-layer config is ~10 GB per model for 1k rollouts at fp16. Going to `"all"` 65× the disk — only do that if your probe genuinely needs all-layer info.

Alternatively, run extraction on demand inside your probe training loop — slower per epoch but no disk cost.

## Suggested probe interface

A small `predict.py` of this shape makes your probe runnable by anyone (and by `scoring/score_probes.py`):

```python
def load_probe() -> Any:
    """Load your probe weights (e.g., torch.load). Return arbitrary object."""

def predict(probe, residuals: np.ndarray, attention_mask: np.ndarray | None = None) -> float:
    """
    residuals: (n_selected_layers, n_tokens, d_model) per-sample, fp16
                — same layer subset your probe was trained on (you pick which layers
                  to extract via extract_config.json; the scorer extracts the same set)
    attention_mask: (n_tokens,) bool — True for real tokens

    Returns: float in [0, 1] — probability the sample is positive class
    """
```

If your probe needs a different layer subset, expose a `layers` constant or function so it's clear which residuals to extract before calling `predict`.

Plus a probe weights file (`probe.pt`).

`scoring/score_probes.py` iterates the test split, runs `predict` per row, and computes AUC per probe.

## Suggested metric to track

**Mean AUC across the 5 probes**:

```
mean_auc = (auc_probe1 + auc_probe2 + auc_probe3 + auc_refusal_gemma + auc_refusal_qwen) / 5
```

Variance across the 5 is also worth reporting — a low-variance probe family that ties on mean is a more robust result than a specialist.

## Constraints

- No external data: train only on the provided `train` rows.
- Frozen LLM: you may not change Gemma/Qwen weights.
- Reproducible: your `predict.py` should be deterministic given the input residuals.
- A100-80GB-friendly: probe forward should fit in ≤ 2 GB VRAM (residuals are pre-computed).

## What's interesting / open

- **Qwen vs Gemma**: Qwen has a hybrid attention architecture (mixed full-attention + linear-attention layers). Whether refusal signal lives at the same depth and same diffusion as in Gemma is an open question. Linear probes on a single layer's final-token may underperform across layers; layer-aware methods might help.
- **Cyber probes**: the graded ladder (`benign → dual_use → high_risk_dual_use → prohibited`) implies the upper levels are easier to discriminate against the bottom (`prohibited` vs everything-else, P3) but harder to separate from neighbors (`high_risk_dual_use` vs `dual_use`). Expect P3 > P1 > P2 in difficulty; methods that share representations across the 3 probes may help on P2.
- **All-layer methods vs single-layer**: more capacity, harder to optimize, more disk for extracts.

Bring your favorite approach: linear, MLP, attention probe, all-layer-concat, mixture-of-layers, distilled, ensembled, etc.
